// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Copy a block from a source to a destination using memcpy
pub fn copy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
) -> Result<(), TransferError>
where
    Source: ReadableBlock,
    Destination: WritableBlock,
{
    tracing::info!("copy_block (memcpy) called");

    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        tracing::info!("MEMCPY: Using CONTIGUOUS path - transferring {} bytes as single block",
                      src_data.block_view().map(|v| v.size()).unwrap_or(0));

        // Get storage types before getting mutable views to avoid borrow conflicts
        let src_storage = src_data.storage_type().clone();
        let dst_storage = dst_data.storage_type().clone();

        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;
        debug_assert_eq!(src_view.size(), dst_view.size());

        // Benchmark hook for fully contiguous memcpy transfers (after getting views)
        #[cfg(feature = "block-manager")]
        {
            crate::block_manager::bench::hooks::hook_contiguous_block_transfer_with_size(
                &src_storage,
                &dst_storage,
                src_view.size(),
            );
        }

        unsafe {
            memcpy(src_view.as_ptr(), dst_view.as_mut_ptr(), src_view.size());
        }
    } else {
        tracing::info!("MEMCPY: Using LAYER-BY-LAYER path - transferring {} layers × {} outer_dims",
                      src_data.num_layers(), src_data.num_outer_dims());
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        copy_layers(0..src_data.num_layers(), sources, destinations)?;
    }
    Ok(())
}

/// Copy a range of layers from a source to a destination using memcpy
pub fn copy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
) -> Result<(), TransferError>
where
    Source: ReadableBlock,
    // <Source as ReadableBlock>::StorageType: SystemAccessible + Local,
    Destination: WritableBlock,
    // <Destination as WritableBlock>::StorageType: SystemAccessible + Local,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();

    // Get storage types once before the loop to avoid borrow conflicts
    let src_storage = src_data.storage_type().clone();
    let dst_storage = dst_data.storage_type().clone();

    for layer_idx in layer_range {
        for outer_idx in 0..src_data.num_outer_dims() {
            let src_view = src_data.layer_view(layer_idx, outer_idx)?;
            let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

            debug_assert_eq!(src_view.size(), dst_view.size());

            // Benchmark hook for each layer transfer (after getting views)
            #[cfg(feature = "block-manager")]
            {
                crate::block_manager::bench::hooks::hook_layer_transfer_with_size(
                    &src_storage,
                    &dst_storage,
                    layer_idx,
                    outer_idx,
                    src_view.size(),
                );
            }

            unsafe {
                memcpy(src_view.as_ptr(), dst_view.as_mut_ptr(), src_view.size());
            }
        }
    }
    Ok(())
}

#[inline(always)]
unsafe fn memcpy(src_ptr: *const u8, dst_ptr: *mut u8, size: usize) {
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination memory regions must not overlap for copy_nonoverlapping"
    );

    unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size) };
}
