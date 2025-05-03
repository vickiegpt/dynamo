use super::*;

use super::TransferError;
use crate::block_manager::storage::{CudaCopyable, DeviceStorage, PinnedStorage};
use anyhow::Result;
use cudarc::driver::result as cuda_result;
pub use cudarc::driver::sys::CUstream;
use std::ops::Range;

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
    let dst_data = destinations.block_data_mut(private::PrivateToken);

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
    let dst_data = destinations.block_data_mut(private::PrivateToken);

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

/// Helper function to perform the appropriate CUDA memcpy based on storage types
unsafe fn dispatch_cuda_memcpy<Source: CudaCopyable, Dest: CudaCopyable>(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: CUstream,
) -> Result<(), TransferError> {
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination memory regions must not overlap for copy_nonoverlapping"
    );

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
        let stream_handle = stream.cu_stream();

        // Allocate host and device memory
        let mut host = pinned_allocator.allocate(1024).unwrap();
        let mut device = device_allocator.allocate(1024).unwrap();

        // Set a pattern in host memory
        unsafe {
            StorageMemset::memset(&mut host, 42, 0, 1024).unwrap();
        }

        // Verify host memory was set correctly
        unsafe {
            let ptr = host.as_ptr().unwrap();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 42));
        }

        // Copy host to device
        unsafe {
            dispatch_cuda_memcpy::<PinnedStorage, DeviceStorage>(
                host.as_ptr().unwrap(),
                device.as_mut_ptr().unwrap(),
                1024,
                stream_handle,
            )
            .unwrap();
        }

        // Synchronize to ensure H2D copy is complete
        stream.synchronize().unwrap();

        // Clear host memory
        unsafe {
            StorageMemset::memset(&mut host, 0, 0, 1024).unwrap();
        }

        // Verify host memory was cleared
        unsafe {
            let ptr = host.as_ptr().unwrap();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 0));
        }

        // Copy back from device to host
        unsafe {
            dispatch_cuda_memcpy::<DeviceStorage, PinnedStorage>(
                device.as_ptr().unwrap(),
                host.as_mut_ptr().unwrap(),
                1024,
                stream_handle,
            )
            .unwrap();
        }

        // Synchronize to ensure D2H copy is complete before verifying
        stream.synchronize().unwrap();

        // Verify the original pattern was restored
        unsafe {
            let ptr = host.as_ptr().unwrap();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 42));
        }
    }
}
