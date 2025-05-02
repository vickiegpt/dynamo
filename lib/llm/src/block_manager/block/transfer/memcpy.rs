use super::*;

/// Copy a block from a source to a destination using memcpy
pub fn memcpy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
) -> Result<(), TransferError>
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
            memcpy(src_view.as_ptr(), dst_view.as_mut_ptr(), src_view.size());
        }
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        memcpy_layers(0..src_data.num_layers(), sources, destinations)?;
    }
    Ok(())
}

/// Copy a range of layers from a source to a destination using memcpy
pub fn memcpy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
) -> Result<(), TransferError>
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
            memcpy(src_view.as_ptr(), dst_view.as_mut_ptr(), src_view.size());
        }
    }
    Ok(())
}

unsafe fn memcpy(src_ptr: *const u8, dst_ptr: *mut u8, size: usize) {
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination memory regions must not overlap for copy_nonoverlapping"
    );

    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
}
