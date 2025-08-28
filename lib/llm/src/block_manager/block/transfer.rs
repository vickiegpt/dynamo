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

mod context;
mod cuda;
mod memcpy;
mod nixl;
mod strategy;

use super::*;

use crate::block_manager::storage::{
    nixl::{NixlRegisterableStorage, NixlStorage},
    DeviceStorage, DiskStorage, PinnedStorage, SystemStorage, StorageType,
};

use cudarc::driver::CudaStream;

use nixl_sys::NixlDescriptor;
use nixl_sys::XferOp::{Read, Write};
use std::ops::Range;
use tokio::sync::oneshot;
use std::time::Duration;


// Removed unused imports for zero-copy scatter kernel implementation
use cudarc::driver::result as cuda_result;
use std::sync::{Mutex, OnceLock};
use std::collections::VecDeque;

// Pre-allocated device buffers for kernel pointer arrays
const MAX_PREALLOCATED_BLOCKS: usize = 1024;
const TRANSFER_BUFFER_POOL_SIZE: usize = 8;  // Number of concurrent buffers

struct TransferBuffer {
    id: usize,
    src_ptrs: u64,  // Device pointer to u64 array
    dst_ptrs: u64,  // Device pointer to u64 array
    capacity: usize,
}

struct TransferBufferPool {
    available: Mutex<VecDeque<TransferBuffer>>,
    total_buffers: usize,
}

impl TransferBufferPool {
    fn acquire(&self) -> Option<TransferBuffer> {
        let mut available = self.available.lock().unwrap();
        available.pop_front()
    }

    fn release(&self, buffer: TransferBuffer) {
        let mut available = self.available.lock().unwrap();
        available.push_back(buffer);
    }

    fn len(&self) -> usize {
        let available = self.available.lock().unwrap();
        available.len()
    }
}

// Global storage for transfer buffer pool and kernel function
// Note: Store CUDA pointers as usize to avoid Send/Sync issues with raw pointers
static TRANSFER_BUFFER_POOL: OnceLock<Result<TransferBufferPool, String>> = OnceLock::new();
static COPY_KERNEL_MODULE: Mutex<Option<usize>> = Mutex::new(None);
static COPY_KERNEL_FUNCTION: Mutex<Option<usize>> = Mutex::new(None);

// Load the copy_kernel_KV module from FATBIN
fn get_copy_kernel_module() -> Result<cudarc::driver::sys::CUmodule, TransferError> {
    let mut module_guard = COPY_KERNEL_MODULE.lock().unwrap();

    if let Some(module_ptr) = *module_guard {
        return Ok(module_ptr as cudarc::driver::sys::CUmodule);
    }

    // Load the module on first access
    let module = if let Ok(module) = load_embedded_fatbin() {
        module
    } else if let Ok(module) = load_runtime_fatbin() {
        module
    } else {
        return Err(TransferError::ExecutionError("No copy_kernel_KV FATBIN found (tried embedded and runtime paths)".to_string()));
    };

    let module_ptr = module as usize;
    *module_guard = Some(module_ptr);
    Ok(module as cudarc::driver::sys::CUmodule)
}

// Get the copy_kernel_KV function
fn get_copy_kernel() -> Result<cudarc::driver::sys::CUfunction, TransferError> {
    let mut func_guard = COPY_KERNEL_FUNCTION.lock().unwrap();

    if let Some(func_ptr) = *func_guard {
        return Ok(func_ptr as cudarc::driver::sys::CUfunction);
    }

    // Load the function on first access
    let module = get_copy_kernel_module()?;
    let func = unsafe {
        let mut func = std::ptr::null_mut();
        let func_name = std::ffi::CString::new("copy_kernel_KV").unwrap();
        let result = cudarc::driver::sys::cuModuleGetFunction(&mut func, module, func_name.as_ptr());
        if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            func
        } else {
            return Err(TransferError::ExecutionError(format!("Failed to get kernel function: {:?}", result)));
        }
    };

    let func_ptr = func as usize;
    *func_guard = Some(func_ptr);
    Ok(func as cudarc::driver::sys::CUfunction)
}

// Get or initialize pre-allocated transfer buffer pool
fn get_transfer_buffer_pool() -> Result<&'static TransferBufferPool, TransferError> {
    let result = TRANSFER_BUFFER_POOL.get_or_init(|| {
        let ptr_array_size = MAX_PREALLOCATED_BLOCKS * std::mem::size_of::<u64>();
        let mut buffers = VecDeque::with_capacity(TRANSFER_BUFFER_POOL_SIZE);

        for i in 0..TRANSFER_BUFFER_POOL_SIZE {
            let (src_ptrs, dst_ptrs) = unsafe {
                let src_ptrs = match cuda_result::malloc_sync(ptr_array_size) {
                    Ok(ptr) => ptr,
                    Err(e) => return Err(format!("Failed to allocate transfer src buffer {}: {}", i, e)),
                };
                let dst_ptrs = match cuda_result::malloc_sync(ptr_array_size) {
                    Ok(ptr) => ptr,
                    Err(e) => return Err(format!("Failed to allocate transfer dst buffer {}: {}", i, e)),
                };
                (src_ptrs, dst_ptrs)
            };

            buffers.push_back(TransferBuffer {
                id: i,
                src_ptrs,
                dst_ptrs,
                capacity: MAX_PREALLOCATED_BLOCKS,
            });
        }

        println!("📦 Allocated {} transfer buffers, {} blocks per buffer ({} KB each)",
                 TRANSFER_BUFFER_POOL_SIZE, MAX_PREALLOCATED_BLOCKS, ptr_array_size / 1024);

        Ok(TransferBufferPool {
            available: Mutex::new(buffers),
            total_buffers: TRANSFER_BUFFER_POOL_SIZE,
        })
    });

    match result {
        Ok(pool) => Ok(pool),
        Err(e) => Err(TransferError::ExecutionError(e.clone())),
    }
}

// RAII wrapper for automatic buffer release
struct TransferBufferGuard {
    buffer: Option<TransferBuffer>,
    pool: &'static TransferBufferPool,
}

impl TransferBufferGuard {
    fn new(buffer: TransferBuffer, pool: &'static TransferBufferPool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    fn get(&self) -> &TransferBuffer {
        self.buffer.as_ref().unwrap()
    }
}

impl Drop for TransferBufferGuard {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

// Try to load embedded FATBIN (compile-time)
fn load_embedded_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    // Check if FATBIN was embedded at compile time
    if option_env!("DYNAMO_FATBIN_AVAILABLE").is_some() {
        // FATBIN was copied to OUT_DIR by build.rs and embedded here
        const FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy_kernel_kv.fatbin"));
        println!("📦 Loading embedded FATBIN ({} bytes)", FATBIN.len());
        unsafe {
            let mut module = std::ptr::null_mut();
            let result = cudarc::driver::sys::cuModuleLoadData(&mut module, FATBIN.as_ptr() as *const std::ffi::c_void);
            if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Ok(module);
            }
        }
    }

    Err(cudarc::driver::DriverError(cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND))
}

// Try to load FATBIN from filesystem (runtime)
fn load_runtime_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    // 1. Check runtime environment variable first
    if let Ok(runtime_path) = std::env::var("DYNAMO_FATBIN_PATH") {
        if let Ok(fatbin_data) = std::fs::read(&runtime_path) {
            println!("📁 Loading FATBIN from runtime env var: {}", runtime_path);
            unsafe {
                let mut module = std::ptr::null_mut();
                let result = cudarc::driver::sys::cuModuleLoadData(&mut module, fatbin_data.as_ptr() as *const std::ffi::c_void);
                if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Ok(module);
                }
            }
        }
    }

    // 2. Check standard runtime locations (priority order)
    let runtime_paths = [
        "./src/block_manager/block/transfer/kernels/copy_kernel_kv.fatbin",  // Primary: Next to transfer module
        "./kernels/copy_kernel_kv.fatbin",                                   // Working directory kernels
        "./copy_kernel_kv.fatbin",                                           // Current directory
        "/usr/local/lib/dynamo/kernels/copy_kernel_kv.fatbin",               // System install
        "/opt/dynamo/kernels/copy_kernel_kv.fatbin",                         // Alternative system
    ];

    for path in &runtime_paths {
        if let Ok(fatbin_data) = std::fs::read(path) {
            println!("📁 Loading FATBIN from runtime path: {}", path);
            unsafe {
                let mut module = std::ptr::null_mut();
                let result = cudarc::driver::sys::cuModuleLoadData(&mut module, fatbin_data.as_ptr() as *const std::ffi::c_void);
                if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Ok(module);
                }
            }
        }
    }

    Err(cudarc::driver::DriverError(cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND))
}

pub use crate::block_manager::storage::{CudaAccessible, Local, Remote};
pub use async_trait::async_trait;
pub use context::TransferContext;

/// A block that can be the target of a write
pub trait Writable {}

/// A block that can be the source of a read
pub trait Readable {}

pub trait Mutable: Readable + Writable {}

pub trait Immutable: Readable {}

#[derive(Debug)]
pub enum BlockTarget {
    Source,
    Destination,
}

#[derive(Debug, thiserror::Error)]
pub enum TransferError {
    #[error("Builder configuration error: {0}")]
    BuilderError(String),
    #[error("Transfer execution failed: {0}")]
    ExecutionError(String),
    #[error("Incompatible block types provided: {0}")]
    IncompatibleTypes(String),
    #[error("Mismatched source/destination counts: {0} sources, {1} destinations")]
    CountMismatch(usize, usize),
    #[error("Block operation failed: {0}")]
    BlockError(#[from] BlockError),
    // TODO: Add NIXL specific errors
    #[error("No blocks provided")]
    NoBlocksProvided,

    #[error("Mismatched {0:?} block set index: {1} != {2}")]
    MismatchedBlockSetIndex(BlockTarget, usize, usize),

    #[error("Mismatched {0:?} worker ID: {1} != {2}")]
    MismatchedWorkerID(BlockTarget, usize, usize),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixlTransfer {
    Read,
    Write,
}

impl NixlTransfer {
    pub fn as_xfer_op(&self) -> nixl_sys::XferOp {
        match self {
            NixlTransfer::Read => Read,
            NixlTransfer::Write => Write,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    Memcpy,
    CudaAsyncH2D,
    CudaAsyncD2H,
    CudaAsyncD2D,
    CudaBlockingH2D,
    CudaBlockingD2H,
    Nixl(NixlTransfer),
    Invalid,
}

/// Trait for determining the transfer strategy for writing from a local
/// source to a target destination which could be local or remote
pub trait WriteToStrategy<Target> {
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Invalid
    }
}

/// Trait for determining the transfer strategy for reading from a
/// `Source` which could be local or remote into `Self` which must
/// be both local and writable.
pub trait ReadFromStrategy<Source> {
    fn read_from_strategy() -> TransferStrategy {
        TransferStrategy::Invalid
    }
}

impl<RB: ReadableBlock, WB: WritableBlock> WriteToStrategy<WB> for RB
where
    <RB as StorageTypeProvider>::StorageType:
        Local + WriteToStrategy<<WB as StorageTypeProvider>::StorageType>,
{
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        <<RB as StorageTypeProvider>::StorageType as WriteToStrategy<
            <WB as StorageTypeProvider>::StorageType,
        >>::write_to_strategy()
    }
}

impl<WB: WritableBlock, RB: ReadableBlock> ReadFromStrategy<RB> for WB
where
    <RB as StorageTypeProvider>::StorageType: Remote,
    <WB as StorageTypeProvider>::StorageType: NixlRegisterableStorage,
{
    #[inline(always)]
    fn read_from_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Read)
    }
}

pub fn handle_local_transfer<RB, WB>(
    sources: &[RB],
    targets: &mut [WB],
    ctx: Arc<TransferContext>,
) -> Result<oneshot::Receiver<()>, TransferError>
where
    RB: ReadableBlock + WriteToStrategy<WB> + Local,
    WB: WritableBlock,
    <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
{

    let (tx, rx) = oneshot::channel();

    match RB::write_to_strategy() {
        TransferStrategy::Memcpy => {
            for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                // TODO: Unlike all other transfer strategies, this is fully blocking.
                // We probably want some sort of thread pool to handle these.
                memcpy::copy_block(src, dst)?;
            }

            tx.send(()).unwrap();
            Ok(rx)
        }
        TransferStrategy::CudaAsyncH2D
        | TransferStrategy::CudaAsyncD2H
        | TransferStrategy::CudaAsyncD2D => {
            if RB::write_to_strategy() == TransferStrategy::CudaAsyncH2D {
                // Verify H2D: Host -> Device transfer
                if sources.len() > 0 {
                    if let (Ok(first_src), Ok(first_dst)) = (
                        sources[0].block_data().layer_view(0, 0),
                        targets[0].block_data().layer_view(0, 0)
                    ) {
                        unsafe {
                            // Get storage types and block IDs
                            let src_storage = sources[0].block_data().storage_type();
                            let dst_storage = targets[0].block_data().storage_type();
                            let src_block_id = sources[0].block_data().block_id();
                            let dst_block_id = targets[0].block_data().block_id();

                            // Verify H2D direction: source should be Pinned/System, dest should be Device
                            let direction_correct = matches!(src_storage, StorageType::Pinned | StorageType::System) &&
                                                  matches!(dst_storage, StorageType::Device(_));

                            print!("H2D: {} blocks | ", sources.len());
                            print!("src[{}]={:?}@{:p}", src_block_id, src_storage, first_src.as_ptr());
                            if sources.len() > 1 {
                                if let Ok(last_src) = sources[sources.len()-1].block_data().layer_view(0, 0) {
                                    let last_src_id = sources[sources.len()-1].block_data().block_id();
                                    print!("..{}@{:p}", last_src_id, last_src.as_ptr());
                                }
                            }
                            print!(" → dst[{}]={:?}@{:p}", dst_block_id, dst_storage, first_dst.as_ptr());
                            if sources.len() > 1 {
                                if let Ok(last_dst) = targets[targets.len()-1].block_data().layer_view(0, 0) {
                                    let last_dst_id = targets[targets.len()-1].block_data().block_id();
                                    print!("..{}@{:p}", last_dst_id, last_dst.as_ptr());
                                }
                            }

                            if direction_correct {
                                println!(" ✅");
                            } else {
                                println!(" ❌ WRONG_DIRECTION");
                            }
                        }
                    }
                }

                // Launch kernel and get any device pointers that need cleanup
                let cleanup_ptrs = cuda::copy_blocks_with_customized_kernel(sources, targets, ctx.stream().as_ref(), RB::write_to_strategy())?;

                // Record H2D completion event with debug info + cleanup
                let worker_id = if !sources.is_empty() { Some(sources[0].block_data().worker_id()) } else { None };
                ctx.cuda_event_with_cleanup(tx, "H2D".to_string(), worker_id, cleanup_ptrs)?;
                return Ok(rx);
            } else if RB::write_to_strategy() == TransferStrategy::CudaAsyncD2H {
                // Verify D2H: Device -> Host transfer
                if sources.len() > 0 {
                    if let (Ok(first_src), Ok(first_dst)) = (
                        sources[0].block_data().layer_view(0, 0),
                        targets[0].block_data().layer_view(0, 0)
                    ) {
                        unsafe {
                            // Get storage types and block IDs
                            let src_storage = sources[0].block_data().storage_type();
                            let dst_storage = targets[0].block_data().storage_type();
                            let src_block_id = sources[0].block_data().block_id();
                            let dst_block_id = targets[0].block_data().block_id();

                            // Verify D2H direction: source should be Device, dest should be Pinned/System
                            let direction_correct = matches!(src_storage, StorageType::Device(_)) &&
                                                  matches!(dst_storage, StorageType::Pinned | StorageType::System);

                            print!("D2H: {} blocks | ", sources.len());
                            print!("src[{}]={:?}@{:p}", src_block_id, src_storage, first_src.as_ptr());
                            if sources.len() > 1 {
                                if let Ok(last_src) = sources[sources.len()-1].block_data().layer_view(0, 0) {
                                    let last_src_id = sources[sources.len()-1].block_data().block_id();
                                    print!("..{}@{:p}", last_src_id, last_src.as_ptr());
                                }
                            }
                            print!(" → dst[{}]={:?}@{:p}", dst_block_id, dst_storage, first_dst.as_ptr());
                            if sources.len() > 1 {
                                if let Ok(last_dst) = targets[targets.len()-1].block_data().layer_view(0, 0) {
                                    let last_dst_id = targets[targets.len()-1].block_data().block_id();
                                    print!("..{}@{:p}", last_dst_id, last_dst.as_ptr());
                                }
                            }

                            if direction_correct {
                                tracing::info!(" CORRECT DIRECTION");
                            } else {
                                tracing::info!(" WRONG DIRECTION");
                            }
                        }
                    }
                }

                // Launch kernel and get any device pointers that need cleanup
                let cleanup_ptrs = cuda::copy_blocks_with_customized_kernel(sources, targets, ctx.stream().as_ref(), RB::write_to_strategy())?;

                // Record D2H completion event with debug info + cleanup
                let worker_id = if !sources.is_empty() { Some(sources[0].block_data().worker_id()) } else { None };
                ctx.cuda_event_with_cleanup(tx, "D2H".to_string(), worker_id, cleanup_ptrs)?;
                return Ok(rx);
            } else {
                // Fall back to individual copy for single H2D blocks
                for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                    cuda::copy_block(src, dst, ctx.stream().as_ref(), RB::write_to_strategy())?;
                }

                ctx.cuda_event(tx)?;
                return Ok(rx);
            }
        }
        TransferStrategy::Nixl(transfer_type) => {
            let transfer_fut = nixl::write_blocks_to(sources, targets, &ctx, transfer_type)?;

            ctx.async_rt_handle().spawn(async move {
                transfer_fut.await;
                tx.send(()).unwrap();
            });
            Ok(rx)
        }
        _ => Err(TransferError::IncompatibleTypes(format!(
            "Unsupported copy strategy: {:?}",
            RB::write_to_strategy()
        ))),
    }
}

pub trait WriteTo<Target> {
    fn write_to(
        &self,
        dst: &mut Vec<Target>,
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>;
}

impl<RB, WB, L: LocalityProvider> WriteTo<WB> for Vec<RB>
where
    RB: ReadableBlock + WriteToStrategy<WB> + Local,
    <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
    RB: BlockDataProvider<Locality = L>,
    WB: WritableBlock + BlockDataProviderMut<Locality = L>,
{
    fn write_to(
        &self,
        dst: &mut Vec<WB>,
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError> {
        L::handle_transfer(self, dst, ctx)
    }
}

// Add timeout for transfer completion to detect kernel failures
pub async fn handle_local_transfer_with_timeout<RB, WB>(
    sources: Vec<RB>,
    targets: &mut Vec<WB>,
    ctx: Arc<TransferContext>,
) -> Result<(), TransferError>
where
    RB: BlockDataProvider + Send + Sync,
    WB: BlockDataProviderMut + Send + Sync,
    Vec<RB>: WriteTo<WB>,
{
    let completion_receiver = sources.write_to(targets, ctx)?;

    // Add timeout to detect kernel failures
    let timeout_duration = Duration::from_secs(30); // 30 second timeout

    match tokio::time::timeout(timeout_duration, completion_receiver).await {
        Ok(Ok(())) => {
            println!("✅ Transfer completed successfully");
            Ok(())
        }
        Ok(Err(_)) => {
            let error = TransferError::ExecutionError("Transfer completion channel closed".into());
            println!("❌ Transfer failed: channel closed");
            Err(error)
        }
        Err(_) => {
            let error = TransferError::ExecutionError("Transfer timeout - kernel may have crashed".into());
            println!("⏰ Transfer timed out after {:?} - possible kernel crash", timeout_duration);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_to_strategy() {
        // System to ...
        assert_eq!(
            <SystemStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingH2D
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Pinned to ...
        assert_eq!(
            <PinnedStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncH2D
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Device to ...
        assert_eq!(
            <DeviceStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2D
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Nixl to ... should fail to compile
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
    }
}






