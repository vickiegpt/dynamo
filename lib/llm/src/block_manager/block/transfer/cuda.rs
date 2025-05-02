use super::*;

use super::TransferError;
use crate::block_manager::storage::{DeviceStorage, PinnedStorage};
use anyhow::Result;
use cudarc::driver::result as cuda_result;
pub use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};

/// Copy a block from a source to a destination using memcpy
pub fn cuda_memcpy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: CUstream,
) -> Result<()>
where
    Source: BlockDataProvider + Local,
    Source::StorageType: SystemCopyable,
    Destination: BlockDataProviderMut + Local,
    Destination::StorageType: SystemCopyable,
{
    let src_data = sources.block_data(private::PrivateToken);
    let mut dst_data = destinations.block_data_mut(private::PrivateToken);

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;
        debug_assert_eq!(src_view.size(), dst_view.size());
        unsafe {
            memcpy::<Source::StorageType, Destination::StorageType>(
                src_view.as_ptr(),
                dst_view.as_mut_ptr(),
                src_view.size(),
                stream,
            )?;
        }
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        memcpy_layers(0..src_data.num_layers(), sources, destinations, stream)?;
    }
    Ok(())
}

/// Copy a range of layers from a source to a destination using memcpy
pub fn cuda_memcpy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: CUstream,
) -> Result<()>
where
    Source: BlockDataProvider + Local,
    Source::StorageType: SystemCopyable,
    Destination: BlockDataProviderMut + Local,
    Destination::StorageType: SystemCopyable,
{
    let src_data = sources.block_data(private::PrivateToken);
    let mut dst_data = destinations.block_data_mut(private::PrivateToken);

    for layer_idx in layer_range {
        let src_view = src_data.layer_view(layer_idx)?;
        let mut dst_view = dst_data.layer_view_mut(layer_idx)?;

        debug_assert_eq!(src_view.size(), dst_view.size());
        unsafe {
            memcpy::<Source::StorageType, Destination::StorageType>(
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
unsafe fn memcpy<PinnedStorage, DeviceStorage>(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: CUstream,
) -> Result<()> {
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination memory regions must not overlap for copy_nonoverlapping"
    );

    // convert src_ptr + size to a slice &[u8]
    let src_slice = std::slice::from_raw_parts(src_ptr, size);
    Ok(cudarc::driver::result::memcpy_htod_async(
        dst_ptr as u64,
        src_slice,
        stream,
    )?)
}

/// D2H Implementation
unsafe fn memcpy<DeviceStorage, PinnedStorage>(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: CUstream,
) -> Result<()> {
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination memory regions must not overlap for copy_nonoverlapping"
    );

    // convert src_ptr + size to a slice &[u8]
    let src_slice = std::slice::from_raw_parts(src_ptr, size);
    Ok(cudarc::driver::result::memcpy_dtoh_async(
        dst_ptr as u64,
        src_slice,
        stream,
    )?)
}

/// D2D Implementation
unsafe fn memcpy<DeviceStorage, DeviceStorage>(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: CUstream,
) -> Result<()> {
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination memory regions must not overlap for copy_nonoverlapping"
    );

    Ok(cudarc::driver::result::memcpy_dtod_async(
        dst_ptr as u64,
        src_ptr as u64,
        size,
        stream,
    )?)
}

/// H2D Implementation
unsafe fn memcpy_h2d(
    src_host_ptr: *const u8,
    dst_device_ptr: u64,
    size: usize,
    stream: CUstream,
) -> Result<(), TransferError> {
    debug_assert!(!src_host_ptr.is_null(), "Source host pointer is null");
    debug_assert!(dst_device_ptr != 0, "Destination device pointer is null");

    let src_slice = std::slice::from_raw_parts(src_host_ptr, size);
    cuda_result::memcpy_htod_async(dst_device_ptr, src_slice, stream)
        .map_err(|e| TransferError::ExecutionError(format!("CUDA H2D memcpy failed: {}", e)))?;
    Ok(())
}

/// D2H Implementation
unsafe fn memcpy_d2h(
    src_device_ptr: u64,
    dst_host_ptr: *mut u8,
    size: usize,
    stream: CUstream,
) -> Result<(), TransferError> {
    debug_assert!(src_device_ptr != 0, "Source device pointer is null");
    debug_assert!(!dst_host_ptr.is_null(), "Destination host pointer is null");

    let dst_slice = std::slice::from_raw_parts_mut(dst_host_ptr, size);
    cuda_result::memcpy_dtoh_async(dst_slice, src_device_ptr, stream)
        .map_err(|e| TransferError::ExecutionError(format!("CUDA D2H memcpy failed: {}", e)))?;
    Ok(())
}

/// D2D Implementation
unsafe fn memcpy_d2d(
    src_device_ptr: u64,
    dst_device_ptr: u64,
    size: usize,
    stream: CUstream,
) -> Result<(), TransferError> {
    debug_assert!(src_device_ptr != 0, "Source device pointer is null");
    debug_assert!(dst_device_ptr != 0, "Destination device pointer is null");

    debug_assert!(
        (src_device_ptr as usize + size <= dst_device_ptr as usize)
            || (dst_device_ptr as usize + size <= src_device_ptr as usize),
        "Source and destination device memory regions must not overlap for D2D copy"
    );

    cuda_result::memcpy_dtod_async(dst_device_ptr, src_device_ptr, size, stream)
        .map_err(|e| TransferError::ExecutionError(format!("CUDA D2D memcpy failed: {}", e)))?;
    Ok(())
}
