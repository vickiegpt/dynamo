// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

use super::TransferError;
use crate::block_manager::block::{BlockDataProvider, BlockDataProviderMut};
use anyhow::Result;
use cudarc::driver::result as cuda_result;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::OnceLock;

type CudaMemcpyFnPtr = unsafe fn(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError>;

fn cuda_memcpy_fn_ptr(strategy: &TransferStrategy) -> Result<CudaMemcpyFnPtr, TransferError> {
    match strategy {
        TransferStrategy::CudaAsyncH2D => Ok(cuda_memcpy_h2d),
        TransferStrategy::CudaAsyncD2H => Ok(cuda_memcpy_d2h),
        TransferStrategy::CudaAsyncD2D => Ok(cuda_memcpy_d2d),
        _ => Err(TransferError::ExecutionError(
            "Unsupported copy strategy for CUDA memcpy async".into(),
        )),
    }
}

#[derive(Clone, Debug)]
struct CudaMemcpyRequest {
    src_ptr: u64,
    dst_ptr: u64,
    size: usize,
}

static CUDA_MEMCPY_REQUESTS: OnceLock<Mutex<Vec<CudaMemcpyRequest>>> = OnceLock::new();

fn global_cuda_memcpy_requests() -> &'static Mutex<Vec<CudaMemcpyRequest>> {
    CUDA_MEMCPY_REQUESTS.get_or_init(|| Mutex::new(Vec::new()))
}

fn add_cuda_memcpy_request(request: CudaMemcpyRequest) {
    let requests = global_cuda_memcpy_requests();
    let mut guard = requests.lock().unwrap();
    guard.push(request);
}

fn batch_cuda_memcpy(stream: &CudaStream, memcpy_fn: CudaMemcpyFnPtr) -> Result<(), TransferError> {
    let requests_mutex = global_cuda_memcpy_requests();
    let mut requests = requests_mutex.lock().unwrap();

    if requests.is_empty() {
        return Ok(());
    }

    // Sort requests by src_ptr and dst_ptr to help merging contiguous requests
    requests.sort_by(|a, b| {
        (a.src_ptr, a.dst_ptr)
            .cmp(&(b.src_ptr, b.dst_ptr))
    });

    let mut merged_requests: Vec<CudaMemcpyRequest> = Vec::new();

    let mut curr = requests.first().unwrap().clone();
    for req in requests.iter().skip(1) {
        let curr_src_end = curr.src_ptr + curr.size as u64;
        let curr_dst_end = curr.dst_ptr + curr.size as u64;
        if req.src_ptr == curr_src_end && req.dst_ptr == curr_dst_end {
             // Merge contiguous requests
            curr.size += req.size;
        } else {
            merged_requests.push(curr);
            curr = req.clone();
        }
    }
    requests.clear();
    merged_requests.push(curr);

    tracing::info!("ziqif sending out {} merged requests", merged_requests.len());
    for req in merged_requests {
        unsafe {
            memcpy_fn(
                req.src_ptr as *const u8,
                req.dst_ptr as *mut u8,
                req.size,
                stream,
            )?;
        }
    }

    Ok(())
}

pub fn copy_blocks_with_customized_kernel<'a, Source, Destination>(
    sources: &'a [Source],
    destinations: &'a mut [Destination],
    stream: &CudaStream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    // Use custom copy_kernel_KV for multiple H2D blocks only
    if sources.len() > 1 {
        // Check if blocks are fully contiguous (required for our simple kernel)
        let src_data = sources[0].block_data();
        let is_fully_contiguous = src_data.is_fully_contiguous();

        if is_fully_contiguous {
            tracing::info!("Using copy_kernel_KV (FATBIN) for {} H2D blocks [FullyContiguous]", sources.len());

            let block_size = src_data.block_view()?.size();
            let mut src_ptrs = Vec::with_capacity(sources.len());
            let mut dst_ptrs = Vec::with_capacity(sources.len());

            for (src, dst) in sources.iter().zip(destinations.iter()) {
                let src_view = src.block_data().block_view()?;
                let dst_view = dst.block_data().block_view()?;

                unsafe {
                    src_ptrs.push(src_view.as_ptr() as u64);
                    dst_ptrs.push(dst_view.as_ptr() as u64);
                }
            }

            tracing::info!("Copy size: {} bytes, {} pairs [FullyContiguous blocks]", block_size, sources.len());

            // For now, skip pooled buffers to avoid CUDA context issues during initialization
            // TODO: Implement proper CUDA context-aware buffer pool initialization
            let use_pooled = false; // sources.len() <= MAX_PREALLOCATED_BLOCKS;
            let buffer_guard = None::<TransferBufferGuard>;

            if let Some(ref guard) = buffer_guard {
                // Use pooled buffer - fast path
                let buffer = guard.get();

                unsafe {
                    // Bind CUDA context to current thread for pooled operations
                    let _context_guard = stream.context().bind_to_thread();

                    // Copy pointer arrays to pooled device memory
                    cuda_result::memcpy_htod_async(buffer.src_ptrs, &src_ptrs, stream.cu_stream())
                        .map_err(|e| TransferError::ExecutionError(format!("Failed to copy source pointers: {}", e)))?;
                    cuda_result::memcpy_htod_async(buffer.dst_ptrs, &dst_ptrs, stream.cu_stream())
                        .map_err(|e| TransferError::ExecutionError(format!("Failed to copy dest pointers: {}", e)))?;

                    // Launch kernel with pooled buffers
                    let kernel = get_copy_kernel()?;

                    // Optimal grid sizing for grid-stride kernel
                    let threads_per_block = 256u32;
                    let max_blocks = 1024u32; // Reasonable limit, let grid-stride handle the rest
                    let blocks_needed = std::cmp::min(max_blocks, sources.len() as u32);

                    let grid_dim = (blocks_needed, 1, 1);
                    let block_dim = (threads_per_block, 1, 1);

                    // cuLaunchKernel expects pointers to parameter values
                    let src_ptr_param = buffer.src_ptrs;
                    let dst_ptr_param = buffer.dst_ptrs;
                    let size_param = block_size;
                    let num_pairs_param = sources.len() as i32; // Match kernel's int num_pairs

                    let params = [
                        &src_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                        &dst_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                        &size_param as *const _ as *mut std::ffi::c_void,       // Pointer to size value
                        &num_pairs_param as *const _ as *mut std::ffi::c_void,  // Pointer to num_pairs (int)
                    ];

                    let result = cudarc::driver::sys::cuLaunchKernel(
                        kernel,
                        grid_dim.0, grid_dim.1, grid_dim.2,  // grid dimensions
                        block_dim.0, block_dim.1, block_dim.2,  // block dimensions
                        0, // shared memory
                        stream.cu_stream(),
                        params.as_ptr() as *mut *mut std::ffi::c_void,
                        std::ptr::null_mut(), // extra
                    );
                    if result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(TransferError::ExecutionError(format!("Kernel launch failed: {:?}", result)));
                    }
                }

                let pool = get_transfer_buffer_pool()?;
                tracing::info!("copy_kernel_KV launched with pooled buffer {} (available: {}/{})",
                        buffer.id, pool.len(), pool.total_buffers);
            } else {
                // Fall back to malloc/memcpy/free approach using pinned host memory
                // Note: This path has potential memory leak due to async completion
                if !use_pooled {
                    tracing::info!("Block count {} exceeds buffer capacity {}, using malloc fallback",
                            sources.len(), MAX_PREALLOCATED_BLOCKS);
                } else {
                    tracing::info!("No pooled buffers available, using malloc fallback");
                }

                // Collect all source and destination addresses for each layer's K/V components
                let mut src_addresses = Vec::new();
                let mut dst_addresses = Vec::new();

                let num_layers = sources[0].block_data().num_layers();
                for (src_block, dst_block) in sources.iter().zip(destinations.iter()) {
                    for layer_idx in 0..num_layers {
                        // K cache addresses (outer_dim = 0)
                        if let (Ok(src_k), Ok(dst_k)) = (
                            src_block.block_data().layer_view(layer_idx, 0),
                            dst_block.block_data().layer_view(layer_idx, 0)
                        ) {
                            unsafe {
                                src_addresses.push(src_k.as_ptr() as u64);
                                dst_addresses.push(dst_k.as_ptr() as u64);
                            }
                        }
                        // V cache addresses (outer_dim = 1)
                        if let (Ok(src_v), Ok(dst_v)) = (
                            src_block.block_data().layer_view(layer_idx, 1),
                            dst_block.block_data().layer_view(layer_idx, 1)
                        ) {
                            unsafe {
                                src_addresses.push(src_v.as_ptr() as u64);
                                dst_addresses.push(dst_v.as_ptr() as u64);
                            }
                        }
                    }
                }

                let layer_size = sources[0].block_data().layer_view(0, 0)?.size();
                tracing::info!("Copy size: {} bytes, {} transfers [FullyContiguous fallback: {} src/dst pairs]",
                        layer_size, src_addresses.len(), src_addresses.len());

                let ptr_array_size = src_addresses.len() * std::mem::size_of::<u64>();
                let mut d_src_ptrs = 0u64;
                let mut d_dst_ptrs = 0u64;

                unsafe {
                    // Bind CUDA context to current thread for allocations
                    let _context_guard = stream.context().bind_to_thread();

                    // Allocate device memory for pointer arrays
                    d_src_ptrs = cuda_result::malloc_sync(ptr_array_size)
                        .map_err(|e| TransferError::ExecutionError(format!("Failed to allocate device source pointers: {}", e)))?;
                    d_dst_ptrs = cuda_result::malloc_sync(ptr_array_size)
                        .map_err(|e| TransferError::ExecutionError(format!("Failed to allocate device dest pointers: {}", e)))?;

                    // Copy pointer arrays to device (direct from Vec)
                    cuda_result::memcpy_htod_async(d_src_ptrs, &src_addresses, stream.cu_stream())
                        .map_err(|e| TransferError::ExecutionError(format!("Failed to copy source pointers: {}", e)))?;
                    cuda_result::memcpy_htod_async(d_dst_ptrs, &dst_addresses, stream.cu_stream())
                        .map_err(|e| TransferError::ExecutionError(format!("Failed to copy dest pointers: {}", e)))?;

                    // Launch kernel using fallback malloc approach
                    let kernel = get_copy_kernel()?;

                    // Optimal grid sizing for grid-stride kernel (not one block per transfer!)
                    let threads_per_block = 256u32;
                    let max_blocks = 1024u32; // Reasonable limit, let grid-stride handle the rest
                    let blocks_needed = std::cmp::min(max_blocks, src_addresses.len() as u32);

                    let grid_dim = (blocks_needed, 1, 1);
                    let block_dim = (threads_per_block, 1, 1);

                    tracing::info!("Grid-stride kernel: {} pairs, {} blocks × {} threads (grid-stride handles overflow)",
                            src_addresses.len(), blocks_needed, threads_per_block);

                    // cuLaunchKernel expects pointers to parameter values
                    let src_ptr_param = d_src_ptrs;
                    let dst_ptr_param = d_dst_ptrs;
                    let size_param = layer_size;  // Size per layer component (32KB)
                    let num_pairs_param = src_addresses.len() as i32; // Match kernel's int num_pairs

                    let params = [
                        &src_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                        &dst_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                        &size_param as *const _ as *mut std::ffi::c_void,       // Pointer to size value
                        &num_pairs_param as *const _ as *mut std::ffi::c_void,  // Pointer to num_pairs (int)
                    ];

                    let result = cudarc::driver::sys::cuLaunchKernel(
                        kernel,
                        grid_dim.0, grid_dim.1, grid_dim.2,  // grid dimensions
                        block_dim.0, block_dim.1, block_dim.2,  // block dimensions
                        0, // shared memory
                        stream.cu_stream(),
                        params.as_ptr() as *mut *mut std::ffi::c_void,
                        std::ptr::null_mut(), // extra
                    );
                    if result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(TransferError::ExecutionError(format!("FullyContiguous kernel launch failed: {:?}", result)));
                    }

                    // Note: Memory cleanup deferred - will be cleaned up when stream completes
                    // TODO: Implement proper async cleanup or use pooled buffers to avoid this issue
                    // let _ = cuda_result::free_sync(d_src_ptrs);
                    // let _ = cuda_result::free_sync(d_dst_ptrs);
                }

                tracing::info!("copy_kernel_KV launched successfully (FullyContiguous malloc fallback)");
            }
        } else {
            // LayerSeparate layout detected - collect all layer addresses for scatter kernel
            let num_layers = src_data.num_layers();
            let num_outer_dims = src_data.num_outer_dims();
            let layer_size = src_data.layer_view(0, 0)?.size();
            let total_size = layer_size * num_layers * num_outer_dims;

            tracing::info!("Using copy_kernel_KV (FATBIN) for {} LayerSeparate blocks [{}L×{}O×{}B]",
                     sources.len(), num_layers, num_outer_dims, layer_size);

            // Collect all source and destination addresses for each layer's K/V components
            let mut src_addresses = Vec::new();
            let mut dst_addresses = Vec::new();

            for (src_block, dst_block) in sources.iter().zip(destinations.iter()) {
                for layer_idx in 0..num_layers {
                    // K cache addresses (outer_dim = 0)
                    if let (Ok(src_k), Ok(dst_k)) = (
                        src_block.block_data().layer_view(layer_idx, 0),
                        dst_block.block_data().layer_view(layer_idx, 0)
                    ) {
                        unsafe {
                            src_addresses.push(src_k.as_ptr() as u64);
                            dst_addresses.push(dst_k.as_ptr() as u64);
                        }
                    }
                    // V cache addresses (outer_dim = 1)
                    if let (Ok(src_v), Ok(dst_v)) = (
                        src_block.block_data().layer_view(layer_idx, 1),
                        dst_block.block_data().layer_view(layer_idx, 1)
                    ) {
                        unsafe {
                            src_addresses.push(src_v.as_ptr() as u64);
                            dst_addresses.push(dst_v.as_ptr() as u64);
                        }
                    }
                }
            }

            tracing::info!("Copy size: {} bytes, {} transfers [LayerSeparate: {} src/dst pairs]",
                     layer_size, src_addresses.len(), src_addresses.len());

            // Launch kernel with collected LayerSeparate addresses
            let ptr_array_size = src_addresses.len() * std::mem::size_of::<u64>();
            let mut d_src_ptrs = 0u64;
            let mut d_dst_ptrs = 0u64;

            unsafe {
                // Bind CUDA context to current thread for allocations
                let _context_guard = stream.context().bind_to_thread();

                // Allocate device memory for pointer arrays
                d_src_ptrs = cuda_result::malloc_sync(ptr_array_size)
                    .map_err(|e| TransferError::ExecutionError(format!("Failed to allocate LayerSeparate source pointers: {}", e)))?;
                d_dst_ptrs = cuda_result::malloc_sync(ptr_array_size)
                    .map_err(|e| TransferError::ExecutionError(format!("Failed to allocate LayerSeparate dest pointers: {}", e)))?;

                // Copy pointer arrays to device
                cuda_result::memcpy_htod_async(d_src_ptrs, &src_addresses, stream.cu_stream())
                    .map_err(|e| TransferError::ExecutionError(format!("Failed to copy LayerSeparate source pointers: {}", e)))?;
                cuda_result::memcpy_htod_async(d_dst_ptrs, &dst_addresses, stream.cu_stream())
                    .map_err(|e| TransferError::ExecutionError(format!("Failed to copy LayerSeparate dest pointers: {}", e)))?;

                // Launch kernel for LayerSeparate transfers
                let kernel = get_copy_kernel()?;

                // Optimal grid sizing for grid-stride kernel (not one block per transfer!)
                let threads_per_block = 256u32;
                let max_blocks = 1024u32; // Reasonable limit, let grid-stride handle the rest
                let blocks_needed = std::cmp::min(max_blocks, src_addresses.len() as u32);

                let grid_dim = (blocks_needed, 1, 1);
                let block_dim = (threads_per_block, 1, 1);

                tracing::info!(" LayerSeparate grid-stride: {} pairs, {} blocks × {} threads",
                         src_addresses.len(), blocks_needed, threads_per_block);

                // cuLaunchKernel expects pointers to parameter values
                let src_ptr_param = d_src_ptrs;
                let dst_ptr_param = d_dst_ptrs;
                let size_param = layer_size;  // Size per layer component (32KB)
                let num_pairs_param = src_addresses.len() as i32; // Match kernel's int num_pairs

                let params = [
                    &src_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                    &dst_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                    &size_param as *const _ as *mut std::ffi::c_void,       // Pointer to size value
                    &num_pairs_param as *const _ as *mut std::ffi::c_void,  // Pointer to num_pairs (int)
                ];

                let result = cudarc::driver::sys::cuLaunchKernel(
                    kernel,
                    grid_dim.0, grid_dim.1, grid_dim.2,  // grid dimensions
                    block_dim.0, block_dim.1, block_dim.2,  // block dimensions
                    0, // shared memory
                    stream.cu_stream(),
                    params.as_ptr() as *mut *mut std::ffi::c_void,
                    std::ptr::null_mut(), // extra
                );
                if result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(TransferError::ExecutionError(format!("LayerSeparate kernel launch failed: {:?}", result)));
                }

                // Note: Memory cleanup deferred - will be cleaned up when stream completes
                // TODO: Implement proper async cleanup or use pooled buffers to avoid this issue
                // let _ = cuda_result::free_sync(d_src_ptrs);
                // let _ = cuda_result::free_sync(d_dst_ptrs);
            }

            tracing::info!("copy_kernel_KV launched successfully for LayerSeparate layout");
        }
    } else {
        // Fall back to individual copy for single H2D blocks
        for (src, dst) in sources.iter().zip(destinations.iter_mut()) {
            cuda::copy_block(src, dst, stream, strategy)?;
        }
    }
    Ok(())
}

/// Copy blocks from sources to destinations using CUDA memcpy in batch
pub fn copy_blocks<'a, Source, Destination>(
    sources: &'a [Source],
    destinations: &'a mut [Destination],
    stream: &CudaStream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let memcpy_fn = cuda_memcpy_fn_ptr(&strategy)?;

    for (src, dst) in sources.iter().zip(destinations.iter_mut()) {
        let src_data = src.block_data();
        let dst_data = dst.block_data_mut();

        assert!(!src_data.is_fully_contiguous());
        assert!(!dst_data.is_fully_contiguous());
        assert_eq!(src_data.num_layers(), dst_data.num_layers());

        for layer_idx in 0..src_data.num_layers() {
            for outer_idx in 0..src_data.num_outer_dims() {
                let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

                debug_assert_eq!(src_view.size(), dst_view.size());

                unsafe {
                    add_cuda_memcpy_request(CudaMemcpyRequest {
                        src_ptr: src_view.as_ptr() as u64,
                        dst_ptr: dst_view.as_mut_ptr() as u64,
                        size: src_view.size(),
                    });
                }
            }
        }
    }

    batch_cuda_memcpy(stream, memcpy_fn)?;

    Ok(())
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn copy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: &CudaStream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = cuda_memcpy_fn_ptr(&strategy)?;

    #[cfg(debug_assertions)]
    {
        let expected_strategy =
            expected_strategy::<Source::StorageType, Destination::StorageType>();
        assert_eq!(strategy, expected_strategy);
    }

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;

        debug_assert_eq!(src_view.size(), dst_view.size());

        unsafe {
            memcpy_fn(
                src_view.as_ptr(),
                dst_view.as_mut_ptr(),
                src_view.size(),
                stream,
            )?;
        }
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        copy_layers(
            0..src_data.num_layers(),
            sources,
            destinations,
            stream,
            strategy,
        )?;
    }
    Ok(())
}

/// Copy a range of layers from a source to a destination using CUDA memcpy
pub fn copy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: &CudaStream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = cuda_memcpy_fn_ptr(&strategy)?;

    #[cfg(debug_assertions)]
    {
        let expected_strategy =
            expected_strategy::<Source::StorageType, Destination::StorageType>();
        assert_eq!(strategy, expected_strategy);
    }

    for layer_idx in layer_range {
        for outer_idx in 0..src_data.num_outer_dims() {
            let src_view = src_data.layer_view(layer_idx, outer_idx)?;
            let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

            debug_assert_eq!(src_view.size(), dst_view.size());

            unsafe {
                memcpy_fn(
                    src_view.as_ptr(),
                    dst_view.as_mut_ptr(),
                    src_view.size(),
                    stream,
                )?;
            }
        }
    }
    Ok(())
}


/// H2D Implementation
#[inline(always)]
unsafe fn cuda_memcpy_h2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {

    // ADD INDIVIDUAL TRANSFER LOGGING
    tracing::info!(" H2D Transfer: 0x{:x} → 0x{:x} ({} bytes)",
        src_ptr as usize, dst_ptr as usize, size);

    debug_assert!(!src_ptr.is_null(), "Source host pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");

    unsafe {
        let src_slice = std::slice::from_raw_parts(src_ptr, size);
        cuda_result::memcpy_htod_async(dst_ptr as u64, src_slice, stream.cu_stream())
            .map_err(|e| TransferError::ExecutionError(format!("CUDA H2D memcpy failed: {}", e)))?
    };
    Ok(())
}

/// D2H Implementation
#[inline(always)]
unsafe fn cuda_memcpy_d2h(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    tracing::info!(" D2H Transfer: 0x{:x} → 0x{:x} ({} bytes)",
        src_ptr as usize, dst_ptr as usize, size);

    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination host pointer is null");
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination device memory regions must not overlap for D2D copy"
    );

    unsafe {
        let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, size);
        cuda_result::memcpy_dtoh_async(dst_slice, src_ptr as u64, stream.cu_stream())
            .map_err(|e| TransferError::ExecutionError(format!("CUDA D2H memcpy failed: {}", e)))?;
    }
    Ok(())
}

/// D2D Implementation
#[inline(always)]
unsafe fn cuda_memcpy_d2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination device memory regions must not overlap for D2D copy"
    );

    unsafe {
        cuda_result::memcpy_dtod_async(dst_ptr as u64, src_ptr as u64, size, stream.cu_stream())
            .map_err(|e| TransferError::ExecutionError(format!("CUDA D2D memcpy failed: {}", e)))?
    };
    Ok(())
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;
    use crate::block_manager::storage::{
        DeviceAllocator, PinnedAllocator, StorageAllocator, StorageMemset,
    };

    #[test]
    fn test_memset_and_transfer() {
        // Create allocators
        let device_allocator = DeviceAllocator::default();
        let pinned_allocator = PinnedAllocator::default();

        let ctx = device_allocator.ctx().clone();

        // Create CUDA stream
        let stream = ctx.new_stream().unwrap();

        // Allocate host and device memory
        let mut host = pinned_allocator.allocate(1024).unwrap();
        let mut device = device_allocator.allocate(1024).unwrap();

        // Set a pattern in host memory
        StorageMemset::memset(&mut host, 42, 0, 1024).unwrap();

        // Verify host memory was set correctly
        unsafe {
            let ptr = host.as_ptr();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 42));
        }

        // Copy host to device
        unsafe {
            cuda_memcpy_h2d(host.as_ptr(), device.as_mut_ptr(), 1024, stream.as_ref()).unwrap();
        }

        // Synchronize to ensure H2D copy is complete
        stream.synchronize().unwrap();

        // Clear host memory
        StorageMemset::memset(&mut host, 0, 0, 1024).unwrap();

        // Verify host memory was cleared
        unsafe {
            let ptr = host.as_ptr();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 0));
        }

        // Copy back from device to host
        unsafe {
            cuda_memcpy_d2h(device.as_ptr(), host.as_mut_ptr(), 1024, stream.as_ref()).unwrap();
        }

        // Synchronize to ensure D2H copy is complete before verifying
        stream.synchronize().unwrap();

        // Verify the original pattern was restored
        unsafe {
            let ptr = host.as_ptr();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 42));
        }
    }
}
