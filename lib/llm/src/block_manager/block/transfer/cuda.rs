use super::*;

use super::TransferError;
use crate::block_manager::storage::{CudaCopyable, DeviceStorage, PinnedStorage};
use anyhow::Result;
use cudarc::driver::result as cuda_result;
pub use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaSlice, CudaStream};
use std::ops::Range;

/// Helper function to perform the appropriate CUDA memcpy based on storage types
unsafe fn dispatch_cuda_memcpy<Source: CudaCopyable, Dest: CudaCopyable>(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: CUstream,
) -> Result<(), TransferError> {
    match (
        std::any::TypeId::of::<Source>(),
        std::any::TypeId::of::<Dest>(),
    ) {
        (src, dst)
            if src == std::any::TypeId::of::<PinnedStorage>()
                && dst == std::any::TypeId::of::<DeviceStorage>() =>
        {
            cuda_memcpy_h2d(src_ptr, dst_ptr, size, stream)
        }
        (src, dst)
            if src == std::any::TypeId::of::<DeviceStorage>()
                && dst == std::any::TypeId::of::<PinnedStorage>() =>
        {
            cuda_memcpy_d2h(src_ptr, dst_ptr, size, stream)
        }
        (src, dst)
            if src == std::any::TypeId::of::<DeviceStorage>()
                && dst == std::any::TypeId::of::<DeviceStorage>() =>
        {
            cuda_memcpy_d2d(src_ptr, dst_ptr, size, stream)
        }
        _ => Err(TransferError::ExecutionError(
            "Unsupported storage type combination for CUDA memcpy".into(),
        )),
    }
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn cuda_memcpy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: CUstream,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider + Local,
    Source::StorageType: CudaCopyable,
    Destination: BlockDataProviderMut + Local,
    Destination::StorageType: CudaCopyable,
{
    let src_data = sources.block_data(private::PrivateToken);
    let mut dst_data = destinations.block_data_mut(private::PrivateToken);

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;
        debug_assert_eq!(src_view.size(), dst_view.size());
        unsafe {
            dispatch_cuda_memcpy::<Source::StorageType, Destination::StorageType>(
                src_view.as_ptr(),
                dst_view.as_mut_ptr(),
                src_view.size(),
                stream,
            )?;
        }
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        cuda_memcpy_layers(0..src_data.num_layers(), sources, destinations, stream)?;
    }
    Ok(())
}

/// Copy a range of layers from a source to a destination using CUDA memcpy
pub fn cuda_memcpy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: CUstream,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider + Local,
    Source::StorageType: CudaCopyable,
    Destination: BlockDataProviderMut + Local,
    Destination::StorageType: CudaCopyable,
{
    let src_data = sources.block_data(private::PrivateToken);
    let mut dst_data = destinations.block_data_mut(private::PrivateToken);

    for layer_idx in layer_range {
        let src_view = src_data.layer_view(layer_idx)?;
        let mut dst_view = dst_data.layer_view_mut(layer_idx)?;

        debug_assert_eq!(src_view.size(), dst_view.size());
        unsafe {
            dispatch_cuda_memcpy::<Source::StorageType, Destination::StorageType>(
                src_view.as_ptr(),
                dst_view.as_mut_ptr(),
                src_view.size(),
                stream,
            )?;
        }
    }
    Ok(())
}

/// H2D Implementation
unsafe fn cuda_memcpy_h2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: CUstream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source host pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");

    let src_slice = std::slice::from_raw_parts(src_ptr, size);
    cuda_result::memcpy_htod_async(dst_ptr as u64, src_slice, stream)
        .map_err(|e| TransferError::ExecutionError(format!("CUDA H2D memcpy failed: {}", e)))?;
    Ok(())
}

/// D2H Implementation
unsafe fn cuda_memcpy_d2h(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: CUstream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination host pointer is null");

    let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, size);
    cuda_result::memcpy_dtoh_async(dst_slice, src_ptr as u64, stream)
        .map_err(|e| TransferError::ExecutionError(format!("CUDA D2H memcpy failed: {}", e)))?;
    Ok(())
}

/// D2D Implementation
unsafe fn cuda_memcpy_d2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: CUstream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");

    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination device memory regions must not overlap for D2D copy"
    );

    cuda_result::memcpy_dtod_async(dst_ptr as u64, src_ptr as u64, size, stream)
        .map_err(|e| TransferError::ExecutionError(format!("CUDA D2D memcpy failed: {}", e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    // TODO: Add test cases for:
    // 1. H2D copy (Pinned -> Device)
    // 2. D2H copy (Device -> Pinned)
    // 3. D2D copy (Device -> Device)
    // 4. Error cases (null pointers, invalid sizes)
    // 5. Layer-by-layer copy
    // 6. Contiguous block copy
}
