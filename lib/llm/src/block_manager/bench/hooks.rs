// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Transfer Benchmarking Hooks 🪝
//!
//! This module provides non-intrusive hooks that integrate with existing transfer
//! infrastructure to collect performance metrics without impacting production workloads.

use super::{TransferPath, global_benchmark, record_transfer};
use crate::block_manager::{
    block::{BlockDataProvider, BlockDataProviderMut, data::BlockDataExt},
    layout::LayoutType,
    storage::StorageType,
};

// Default layout type for benchmarking (since we don't track layout variations)
const DEFAULT_LAYOUT: LayoutType = LayoutType::LayerSeparate { outer_contiguous: true };

/// Calculate total transfer size for a batch of blocks
pub fn calculate_transfer_size<Source, Target>(
    sources: &[Source],
    targets: &[Target],
) -> (u64, usize)
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    let mut total_bytes = 0u64;
    let total_blocks = sources.len();

    for (src, dst) in sources.iter().zip(targets.iter()) {
        let src_data = src.block_data();
        let _dst_data = dst.block_data();

        // Calculate size based on whether blocks are fully contiguous
        let block_size = if src_data.is_fully_contiguous() {
            // For fully contiguous blocks, get the full block size
            match src_data.block_view() {
                Ok(view) => view.size() as u64,
                Err(_) => {
                    // Fallback: calculate from layout parameters
                    let layers = src_data.num_layers() as u64;
                    let outer_dims = src_data.num_outer_dims() as u64;
                    let page_size = src_data.page_size() as u64;
                    let inner_dim = src_data.num_inner_dims() as u64;
                    // Assuming 2 bytes per element (adjust as needed)
                    layers * outer_dims * page_size * inner_dim * 2
                }
            }
        } else {
            // For non-contiguous blocks, sum up layer sizes
            let mut layer_total = 0u64;
            for layer_idx in 0..src_data.num_layers() {
                for outer_idx in 0..src_data.num_outer_dims() {
                    match src_data.layer_view(layer_idx, outer_idx) {
                        Ok(view) => layer_total += view.size() as u64,
                        Err(_) => {
                            // Fallback calculation
                            let page_size = src_data.page_size() as u64;
                            let inner_dim = src_data.num_inner_dims() as u64;
                            layer_total += page_size * inner_dim * 2; // Assuming 2 bytes per element
                        }
                    }
                }
            }
            layer_total
        };

        total_bytes += block_size;
    }

    (total_bytes, total_blocks)
}

/// Calculate transfer size for a single block transfer
fn calculate_single_transfer_size<Source, Target>(
    source: &Source,
    _target: &mut Target,
) -> (u64, usize)
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    let src_data = source.block_data();

    // Calculate size based on whether blocks are fully contiguous
    let block_size = if src_data.is_fully_contiguous() {
        // For fully contiguous blocks, get the full block size
        match src_data.block_view() {
            Ok(view) => view.size() as u64,
            Err(_) => {
                // Fallback: calculate from layout parameters
                let layers = src_data.num_layers() as u64;
                let outer_dims = src_data.num_outer_dims() as u64;
                let page_size = src_data.page_size() as u64;
                let inner_dim = src_data.num_inner_dims() as u64;
                // Assuming 2 bytes per element (adjust as needed)
                layers * outer_dims * page_size * inner_dim * 2
            }
        }
    } else {
        // For non-contiguous blocks, sum up layer sizes
        let mut layer_total = 0u64;
        for layer_idx in 0..src_data.num_layers() {
            for outer_idx in 0..src_data.num_outer_dims() {
                match src_data.layer_view(layer_idx, outer_idx) {
                    Ok(view) => layer_total += view.size() as u64,
                    Err(_) => {
                        // Fallback calculation
                        let page_size = src_data.page_size() as u64;
                        let inner_dim = src_data.num_inner_dims() as u64;
                        layer_total += page_size * inner_dim * 2; // Assuming 2 bytes per element
                    }
                }
            }
        }
        layer_total
    };

    (block_size, 1)
}

/// Hook for recording a single layer transfer
pub fn hook_layer_transfer<Source, Target>(
    source: &Source,
    target: &Target,
    layer_idx: usize,
    outer_idx: usize,
) where
    Source: BlockDataProvider,
    Target: BlockDataProvider,
{
    if global_benchmark().is_none() {
        return;
    }

    let src_data = source.block_data();
    let src_storage = src_data.storage_type();
    let dst_storage = target.block_data().storage_type();
    let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

    // Get the layer view size
    let layer_size = match src_data.layer_view(layer_idx, outer_idx) {
        Ok(view) => view.size() as u64,
        Err(_) => {
            // Fallback calculation
            let page_size = src_data.page_size() as u64;
            let inner_dim = src_data.num_inner_dims() as u64;
            page_size * inner_dim * 2 // Assuming 2 bytes per element
        }
    };

    tracing::info!(
        "Layer transfer: {} -> {}, layer={}, outer={}, {} bytes",
        storage_type_to_str(src_storage),
        storage_type_to_str(dst_storage),
        layer_idx,
        outer_idx,
        layer_size
    );

    // Record the transfer as LayerSeparate layout
    let layout = LayoutType::LayerSeparate { outer_contiguous: true };
    record_transfer(transfer_path, layout, layer_size, 1);
}

/// Hook for recording a single layer transfer with explicit size
pub fn hook_layer_transfer_with_size(
    src_storage: &StorageType,
    dst_storage: &StorageType,
    _layer_idx: usize,
    _outer_idx: usize,
    layer_size: usize,
) {
    if global_benchmark().is_none() {
        return;
    }

    let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

    // Record the transfer as LayerSeparate layout
    // Note: We pass blocks=0 because this is just a layer transfer, not a full block
    // The transfer count will still increment, but block count won't
    let layout = LayoutType::LayerSeparate { outer_contiguous: true };
    record_transfer(transfer_path, layout, layer_size as u64, 0);
}

/// Hook for recording a contiguous block transfer with explicit size
pub fn hook_contiguous_block_transfer_with_size(
    src_storage: &StorageType,
    dst_storage: &StorageType,
    block_size: usize,
) {
    if global_benchmark().is_none() {
        return;
    }

    let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

    // Record the transfer as FullyContiguous layout
    // We pass blocks=1 because this is a full block transfer
    let layout = LayoutType::FullyContiguous;
    record_transfer(transfer_path, layout, block_size as u64, 1);
}

/// Hook for NIXL transfers - call this before initiating NIXL transfer
pub fn hook_nixl_transfer<Source, Target>(sources: &[Source], targets: &mut [Target])
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    tracing::info!("hook_nixl_transfer called with {} sources", sources.len());

    if sources.is_empty() || targets.is_empty() {
        tracing::info!("hook_nixl_transfer: Empty sources or targets, returning");
        return;
    }

    // Get storage types to determine transfer path
    let src_storage = sources[0].block_data().storage_type();
    let dst_storage = targets[0].block_data().storage_type();
    let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

    // Calculate transfer size
    let (total_bytes, total_blocks) = calculate_transfer_size(sources, targets);

    tracing::info!(
        "NIXL transfer: {} -> {}, {} blocks, {} bytes",
        storage_type_to_str(src_storage),
        storage_type_to_str(dst_storage),
        total_blocks,
        total_bytes
    );

    // Record the transfer (using a fixed layout type since we don't track layout variations)
    let layout = LayoutType::LayerSeparate { outer_contiguous: true };
    record_transfer(transfer_path, layout, total_bytes, total_blocks);
}

/// Hook for single NIXL transfer - call this before initiating NIXL transfer for one block
pub fn hook_nixl_single_transfer<Source, Target>(source: &Source, target: &Target)
where
    Source: BlockDataProvider,
    Target: BlockDataProvider,
{
    tracing::info!("hook_nixl_single_transfer called");

    // Calculate transfer size - we need to cast target to get its data
    let src_data = source.block_data();
    let dst_data = target.block_data();

    let block_size = if src_data.is_fully_contiguous() {
        match src_data.block_view() {
            Ok(view) => view.size() as u64,
            Err(_) => {
                let layers = src_data.num_layers() as u64;
                let outer_dims = src_data.num_outer_dims() as u64;
                let page_size = src_data.page_size() as u64;
                let inner_dim = src_data.num_inner_dims() as u64;
                layers * outer_dims * page_size * inner_dim * 2
            }
        }
    } else {
        let mut layer_total = 0u64;
        for layer_idx in 0..src_data.num_layers() {
            for outer_idx in 0..src_data.num_outer_dims() {
                match src_data.layer_view(layer_idx, outer_idx) {
                    Ok(view) => layer_total += view.size() as u64,
                    Err(_) => {
                        let page_size = src_data.page_size() as u64;
                        let inner_dim = src_data.num_inner_dims() as u64;
                        layer_total += page_size * inner_dim * 2;
                    }
                }
            }
        }
        layer_total
    };

    let src_storage = src_data.storage_type();
    let dst_storage = dst_data.storage_type();
    let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

    tracing::info!(
        "NIXL single transfer: {} -> {}, 1 blocks, {} bytes",
        storage_type_to_str(src_storage),
        storage_type_to_str(dst_storage),
        block_size
    );

    let layout = LayoutType::FullyContiguous;
    record_transfer(transfer_path, layout, block_size, 1);
}

/// Hook for CUDA transfers - call this before initiating CUDA transfer
pub fn hook_cuda_transfer<Source, Target>(sources: &[Source], targets: &mut [Target])
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    eprintln!("[BENCH] hook_cuda_transfer called with {} sources", sources.len());

    if sources.is_empty() || targets.is_empty() {
        eprintln!("[BENCH] hook_cuda_transfer: Empty sources or targets, returning");
        return;
    }

    let src_layout = DEFAULT_LAYOUT;
    let dst_layout = DEFAULT_LAYOUT;

    let src_storage = sources[0].block_data().storage_type();
    let dst_storage = targets[0].block_data().storage_type();
    let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

    let (total_bytes, total_blocks) = calculate_transfer_size(sources, targets);

    eprintln!(
        "[BENCH] CUDA transfer: {} -> {}, {} blocks, {} bytes, src_layout={:?}, dst_layout={:?}",
        storage_type_to_str(src_storage),
        storage_type_to_str(dst_storage),
        total_blocks,
        total_bytes,
        src_layout,
        dst_layout
    );

    record_transfer(transfer_path, src_layout, total_bytes, total_blocks);

    if src_layout != dst_layout {
        record_transfer(transfer_path, dst_layout, total_bytes, total_blocks);
    }

    tracing::info!(
        "Benchmarking CUDA transfer: {} -> {}, {} blocks, {} bytes, src_layout={:?}, dst_layout={:?}",
        storage_type_to_str(src_storage),
        storage_type_to_str(dst_storage),
        total_blocks,
        total_bytes,
        src_layout,
        dst_layout
    );
}

/// Hook for single CUDA transfer - call this before initiating CUDA transfer
pub fn hook_cuda_single_transfer<Source, Target>(source: &Source, target: &mut Target)
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    tracing::info!("hook_cuda_single_transfer called");

    // Calculate transfer size first before borrowing
    let (total_bytes, total_blocks) = calculate_single_transfer_size(source, target);

    let src_data = source.block_data();
    let dst_data = target.block_data_mut();
    {
        let src_layout = DEFAULT_LAYOUT;
        let dst_layout = DEFAULT_LAYOUT;

        let src_storage = src_data.storage_type();
        let dst_storage = dst_data.storage_type();
        let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

        tracing::info!(
            "CUDA single transfer: {} -> {}, {} blocks, {} bytes, src_layout={:?}, dst_layout={:?}",
            storage_type_to_str(src_storage),
            storage_type_to_str(dst_storage),
            total_blocks,
            total_bytes,
            src_layout,
            dst_layout
        );

        record_transfer(transfer_path, src_layout, total_bytes, total_blocks);

        if src_layout != dst_layout {
            record_transfer(transfer_path, dst_layout, total_bytes, total_blocks);
        }
    }
}

/// Hook for memcpy transfers - call this before initiating memcpy transfer
pub fn hook_memcpy_transfer<Source, Target>(sources: &[Source], targets: &mut [Target])
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    tracing::info!("hook_memcpy_transfer called with {} sources", sources.len());

    if sources.is_empty() || targets.is_empty() {
        tracing::info!("hook_memcpy_transfer: Empty sources or targets, returning");
        return;
    }

    let src_layout = DEFAULT_LAYOUT;
    let dst_layout = DEFAULT_LAYOUT;

    let src_storage = sources[0].block_data().storage_type();
    let dst_storage = targets[0].block_data().storage_type();
    let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

    let (total_bytes, total_blocks) = calculate_transfer_size(sources, targets);

    tracing::info!(
        "memcpy transfer: {} -> {}, {} blocks, {} bytes, src_layout={:?}, dst_layout={:?}",
        storage_type_to_str(src_storage),
        storage_type_to_str(dst_storage),
        total_blocks,
        total_bytes,
        src_layout,
        dst_layout
    );

    record_transfer(transfer_path, src_layout, total_bytes, total_blocks);

    if src_layout != dst_layout {
        record_transfer(transfer_path, dst_layout, total_bytes, total_blocks);
    }
}

/// Hook for single memcpy transfer - call this before initiating memcpy transfer
pub fn hook_memcpy_single_transfer<Source, Target>(source: &Source, target: &mut Target)
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    tracing::info!("hook_memcpy_single_transfer called");

    // Calculate transfer size first before borrowing
    let (total_bytes, total_blocks) = calculate_single_transfer_size(source, target);

    let src_data = source.block_data();
    let dst_data = target.block_data_mut();
    {
        let src_layout = DEFAULT_LAYOUT;
        let dst_layout = DEFAULT_LAYOUT;

        let src_storage = src_data.storage_type();
        let dst_storage = dst_data.storage_type();
        let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

        tracing::info!(
            "memcpy single transfer: {} -> {}, {} blocks, {} bytes, src_layout={:?}, dst_layout={:?}",
            storage_type_to_str(src_storage),
            storage_type_to_str(dst_storage),
            total_blocks,
            total_bytes,
            src_layout,
            dst_layout
        );

        record_transfer(transfer_path, src_layout, total_bytes, total_blocks);

        if src_layout != dst_layout {
            record_transfer(transfer_path, dst_layout, total_bytes, total_blocks);
        }
    }
}

/// Generic hook for any transfer - automatically detects transfer type
pub fn hook_generic_transfer<Source, Target>(
    sources: &[Source],
    targets: &mut [Target],
    strategy_hint: Option<&str>,
) where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    if sources.is_empty() || targets.is_empty() {
        return;
    }

    let src_layout = DEFAULT_LAYOUT;
    let dst_layout = DEFAULT_LAYOUT;

    let src_storage = sources[0].block_data().storage_type();
    let dst_storage = targets[0].block_data().storage_type();
    let transfer_path = TransferPath::from_storage_types(src_storage, dst_storage);

    let (total_bytes, total_blocks) = calculate_transfer_size(sources, targets);

    record_transfer(transfer_path, src_layout, total_bytes, total_blocks);

    if src_layout != dst_layout {
        record_transfer(transfer_path, dst_layout, total_bytes, total_blocks);
    }

    tracing::debug!(
        "Benchmarking generic transfer ({}): {} -> {}, {} blocks, {} bytes, src_layout={:?}, dst_layout={:?}",
        strategy_hint.unwrap_or("unknown"),
        storage_type_to_str(src_storage),
        storage_type_to_str(dst_storage),
        total_blocks,
        total_bytes,
        src_layout,
        dst_layout
    );
}



/// Convert StorageType to string representation for logging
fn storage_type_to_str(storage_type: &StorageType) -> &'static str {
    match storage_type {
        StorageType::System => "System",
        StorageType::Device(_) => "Device",
        StorageType::Pinned => "Pinned",
        StorageType::Disk(_) => "Disk",
        StorageType::Nixl => "Nixl",
        StorageType::Null => "Null",
    }
}

/// Utility to get a summary of current benchmarking state
pub fn get_benchmark_summary() -> Option<String> {
    global_benchmark().map(|benchmark| {
        let metrics = benchmark.get_metrics();
        let total_transfers: usize = metrics
            .values()
            .map(|m| m.transfer_count.load(std::sync::atomic::Ordering::Relaxed))
            .sum();
        let total_bytes: u64 = metrics
            .values()
            .map(|m| m.total_bytes.load(std::sync::atomic::Ordering::Relaxed))
            .sum();

        format!(
            "Benchmark '{}': {} transfers, {} total bytes across {} paths",
            benchmark.session_name,
            total_transfers,
            total_bytes,
            metrics.len()
        )
    })
}

/// Enable/disable benchmarking at runtime
pub fn set_benchmarking_enabled(enabled: bool) {
    if let Some(benchmark) = global_benchmark() {
        benchmark.set_enabled(enabled);
    }
}

/// Check if benchmarking is currently enabled
pub fn is_benchmarking_enabled() -> bool {
    global_benchmark().map(|b| b.is_enabled()).unwrap_or(false)
}

/// Reset all benchmark data
pub fn reset_benchmark_data() {
    if let Some(benchmark) = global_benchmark() {
        benchmark.reset();
    }
}

/// Macro for easy integration with existing transfer code
#[macro_export]
macro_rules! benchmark_transfer {
    ($sources:expr, $targets:expr) => {
        $crate::block_manager::bench::hooks::hook_generic_transfer($sources, $targets, None)
    };
    ($sources:expr, $targets:expr, $strategy:expr) => {
        $crate::block_manager::bench::hooks::hook_generic_transfer(
            $sources,
            $targets,
            Some($strategy),
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::{
        block::data::local::LocalBlockData,
        layout::{FullyContiguous, LayoutConfig},
        storage::tests::NullDeviceStorage,
    };
    use std::sync::Arc;

    #[test]
    fn test_layout_type_detection() {
        // This would require more complex setup with actual block data
        // For now, just test the basic logic
        assert_eq!(
            get_layout_type_from_block(&MockBlockData {
                fully_contiguous: true
            }),
            LayoutType::FullyContiguous
        );

        assert_eq!(
            get_layout_type_from_block(&MockBlockData {
                fully_contiguous: false
            }),
            LayoutType::LayerSeparate {
                outer_contiguous: false
            }
        );
    }

    // Mock block data for testing
    struct MockBlockData {
        fully_contiguous: bool,
    }

    impl MockBlockData {
        fn is_fully_contiguous(&self) -> bool {
            self.fully_contiguous
        }
    }
}
