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

//! Block layout management.
//!
//! This module provides traits and implementations for managing how blocks
//! are arranged in storage, including both contiguous and non-contiguous layouts.

use thiserror::Error;

use crate::block_manager::storage::Storage;
use crate::common::dtype::DType;
use tracing::instrument;

/// Errors that can occur during layout operations
#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Invalid block index: {0}")]
    InvalidBlockIndex(usize),

    #[error("Invalid layer index: {0}")]
    InvalidLayerIndex(usize),

    #[error("Operation failed: {0}")]
    OperationFailed(String),
}

/// Result type for layout operations
pub type Result<T> = std::result::Result<T, LayoutError>;

/// Storage pattern for layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerConfiguration {
    /// All layers are contiguous in memory [n_layers, ...]
    FullyContiguous,

    /// Each layer is stored separately with a common stride between blocks
    /// in different layers
    LayerContiguousWithCommonStride,

    /// Each layer is stored separately with no guaranteed stride
    LayerContiguousWithSeparateStride,

    /// Each page is stored separately with no guaranteed stride
    PageContiguousWithSeparateStride,

    /// NullLayout
    /// Used for testing and debugging
    Null,
}

/// Core trait for block layouts
pub trait BlockLayout: Send + Sync + std::fmt::Debug {
    /// Returns the total number of blocks this layout manages
    fn num_blocks(&self) -> usize;

    /// Returns the number of layers per block
    fn num_layers(&self) -> usize;

    /// Returns the size of each block in bytes
    fn page_size(&self) -> usize;

    /// Returns the inner dimension size
    fn inner_dim(&self) -> usize;

    /// Get the memory region for a specific page [page_size, inner_dim]
    fn get_memory_region(&self, block_idx: usize, layer_idx: usize) -> Result<u64>;

    /// Get the memory region for a specific page [page_size, inner_dim]
    fn memory_region_size(&self) -> usize;
}

/// Configuration for block layouts
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    pub num_blocks: usize,
    pub num_layers: usize,
    pub page_size: usize,
    pub inner_dim: usize,
    pub dtype: DType,
}

/// Contiguous memory layout where all blocks and layers are sequential
#[derive(Debug)]
pub struct FullyContiguous<S: Storage> {
    config: LayoutConfig,
    storage: S,
    layer_stride_in_bytes: usize,
    block_stride_in_bytes: usize,
    memory_region_size: usize,
}

impl<S: Storage> FullyContiguous<S> {
    /// Create a new contiguous layout
    #[instrument(level = "debug", skip(storage), fields(config = ?config))]
    pub fn new(config: LayoutConfig, storage: S) -> Result<Self> {
        // Validate storage size fits [n_blocks, n_layers, page_size, inner_dim]
        let required_size_in_elements =
            config.num_blocks * config.num_layers * config.page_size * config.inner_dim;
        let dtype_size = config.dtype.size_in_bytes();
        let required_size_in_bytes = required_size_in_elements * dtype_size;
        let provided_size = storage.size();

        tracing::debug!(
            provided_size,
            required_size_in_bytes,
            dtype_size,
            "Validating storage size"
        );

        if provided_size < required_size_in_bytes {
            tracing::warn!(
                provided_size,
                required_size_in_bytes,
                "Storage size too small"
            );
            return Err(LayoutError::InvalidConfig(format!(
                "Storage size {} is less than required size {}",
                provided_size, required_size_in_bytes
            )));
        }

        // Contiguous memory region on which the application uses for its own purposes
        // [page_size, inner_dim] elements
        let memory_region = config.page_size * config.inner_dim;
        let memory_region_size = memory_region * config.dtype.size_in_bytes();

        let layer_stride_in_bytes = memory_region * config.dtype.size_in_bytes();
        let block_stride_in_bytes = config.num_layers * layer_stride_in_bytes;

        tracing::debug!(
            memory_region_size,
            layer_stride_in_bytes,
            block_stride_in_bytes,
            "Calculated layout strides"
        );

        Ok(Self {
            config,
            storage,
            layer_stride_in_bytes,
            block_stride_in_bytes,
            memory_region_size,
        })
    }
}

impl<S: Storage> BlockLayout for FullyContiguous<S> {
    fn num_blocks(&self) -> usize {
        self.config.num_blocks
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn page_size(&self) -> usize {
        self.config.page_size
    }

    fn inner_dim(&self) -> usize {
        self.config.inner_dim
    }

    fn get_memory_region(&self, block_idx: usize, layer_idx: usize) -> Result<u64> {
        if block_idx >= self.num_blocks() {
            return Err(LayoutError::InvalidBlockIndex(block_idx));
        }

        if layer_idx >= self.num_layers() {
            return Err(LayoutError::InvalidLayerIndex(layer_idx));
        }

        let offset =
            block_idx * self.block_stride_in_bytes + layer_idx * self.layer_stride_in_bytes;

        Ok(self.storage.addr() + offset as u64)
    }

    fn memory_region_size(&self) -> usize {
        self.memory_region_size
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::block_manager::storage::tests::NullDeviceStorage;
    use crate::common::dtype::DType;
    use dynamo_runtime::logging::init as init_logging;
    use std::sync::Arc;

    const NUM_BLOCKS: usize = 7;
    const NUM_LAYERS: usize = 5;
    const PAGE_SIZE: usize = 4;
    const INNER_DIM: usize = 13;
    const DTYPE: DType = DType::FP32; // Example dtype

    fn setup_layout(storage_size: usize) -> Result<FullyContiguous<NullDeviceStorage>> {
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            dtype: DTYPE,
        };
        let storage = NullDeviceStorage::new(storage_size as u64);
        FullyContiguous::new(config, storage)
    }

    fn calculate_required_size() -> usize {
        NUM_BLOCKS * NUM_LAYERS * PAGE_SIZE * INNER_DIM * DTYPE.size_in_bytes()
    }

    #[test]
    fn test_fc_creation_success() {
        let required_size = calculate_required_size();
        let layout_result = setup_layout(required_size);
        assert!(layout_result.is_ok());
    }

    #[test]
    fn test_fc_creation_insufficient_storage() {
        init_logging();
        let required_size = calculate_required_size();
        let layout_result = setup_layout(required_size - 1); // One byte less
        assert!(layout_result.is_err());
        match layout_result.err().unwrap() {
            LayoutError::InvalidConfig(_) => {} // Expected error
            _ => panic!("Expected InvalidConfig error"),
        }
    }

    #[test]
    fn test_fc_accessor_methods() {
        let required_size = calculate_required_size();
        let layout = setup_layout(required_size).unwrap();

        assert_eq!(layout.num_blocks(), NUM_BLOCKS);
        assert_eq!(layout.num_layers(), NUM_LAYERS);
        assert_eq!(layout.page_size(), PAGE_SIZE);
        assert_eq!(layout.inner_dim(), INNER_DIM);
    }

    #[test]
    fn test_fc_memory_region_size() {
        let required_size = calculate_required_size();
        let layout = setup_layout(required_size).unwrap();
        let expected_region_size = PAGE_SIZE * INNER_DIM * DTYPE.size_in_bytes();
        assert_eq!(layout.memory_region_size(), expected_region_size);
    }

    #[test]
    fn test_fc_offset_calculation() {
        let required_size = calculate_required_size();
        let layout = setup_layout(required_size).unwrap();

        let dtype_size = DTYPE.size_in_bytes();
        let region_elements = PAGE_SIZE * INNER_DIM;
        let layer_stride = region_elements * dtype_size;
        let block_stride = NUM_LAYERS * layer_stride;
        let base_addr = layout.storage.addr(); // Should be 0 for NullDeviceStorage

        // Test first block, first layer
        let expected_offset_0_0 = base_addr + (0 * block_stride + 0 * layer_stride) as u64;
        assert_eq!(layout.get_memory_region(0, 0).unwrap(), expected_offset_0_0);

        // Test first block, last layer
        let last_layer_idx = NUM_LAYERS - 1;
        let expected_offset_0_last =
            base_addr + (0 * block_stride + last_layer_idx * layer_stride) as u64;
        assert_eq!(
            layout.get_memory_region(0, last_layer_idx).unwrap(),
            expected_offset_0_last
        );

        // Test last block, first layer
        let last_block_idx = NUM_BLOCKS - 1;
        let expected_offset_last_0 =
            base_addr + (last_block_idx * block_stride + 0 * layer_stride) as u64;
        assert_eq!(
            layout.get_memory_region(last_block_idx, 0).unwrap(),
            expected_offset_last_0
        );

        // Test last block, last layer
        let expected_offset_last_last =
            base_addr + (last_block_idx * block_stride + last_layer_idx * layer_stride) as u64;
        assert_eq!(
            layout
                .get_memory_region(last_block_idx, last_layer_idx)
                .unwrap(),
            expected_offset_last_last
        );

        // Test intermediate block/layer
        let mid_block_idx = NUM_BLOCKS / 2;
        let mid_layer_idx = NUM_LAYERS / 2;
        let expected_offset_mid_mid =
            base_addr + (mid_block_idx * block_stride + mid_layer_idx * layer_stride) as u64;
        assert_eq!(
            layout
                .get_memory_region(mid_block_idx, mid_layer_idx)
                .unwrap(),
            expected_offset_mid_mid
        );
    }

    #[test]
    fn test_fc_invalid_block_index() {
        let required_size = calculate_required_size();
        let layout = setup_layout(required_size).unwrap();
        let result = layout.get_memory_region(NUM_BLOCKS, 0); // Index == num_blocks (out of bounds)
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidBlockIndex(NUM_BLOCKS)
        ));
    }

    #[test]
    fn test_fc_invalid_layer_index() {
        let required_size = calculate_required_size();
        let layout = setup_layout(required_size).unwrap();
        let result = layout.get_memory_region(0, NUM_LAYERS); // Index == num_layers (out of bounds)
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidLayerIndex(NUM_LAYERS)
        ));
    }
}
