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

use crate::block_manager::storage::{Storage, StorageAllocator};
use crate::common::dtype::DType;
use derive_builder::Builder;
use tracing::instrument;
use validator::Validate;

/// Errors that can occur during layout operations
#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Validation failed: {0}")]
    ValidationError(#[from] validator::ValidationErrors),

    #[error("Invalid block index: {0}")]
    InvalidBlockIndex(usize),

    #[error("Invalid layer index: {0}")]
    InvalidLayerIndex(usize),

    #[error("Operation failed: {0}")]
    OperationFailed(String),
}

/// Storage pattern for layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutType {
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
pub trait BlockLayout:
    BlockLayoutConfig + BlockLayoutLookup + Send + Sync + std::fmt::Debug + 'static
{
    /// The type of storage this layout uses
    type StorageType: Storage;
}

pub trait BlockLayoutConfig {
    fn layout_type(&self) -> LayoutType;

    /// Returns the total number of blocks this layout manages
    fn num_blocks(&self) -> usize;

    /// Returns the number of layers per block
    fn num_layers(&self) -> usize;

    /// Returns the size of each block in bytes
    fn page_size(&self) -> usize;

    /// Returns the inner dimension size
    fn inner_dim(&self) -> usize;
}

pub trait BlockLayoutLookup {
    /// Get the memory region for a specific page [page_size, inner_dim]
    fn get_memory_region(&self, block_idx: usize, layer_idx: usize) -> Result<u64, LayoutError>;

    /// Get the memory region for a specific page [page_size, inner_dim]
    fn memory_region_size(&self) -> usize;
}

/// Configuration for block layouts
#[derive(Debug, Clone, Builder, Validate)]
pub struct LayoutConfig {
    #[validate(range(min = 1))]
    pub num_blocks: usize,

    #[validate(range(min = 1))]
    pub num_layers: usize,

    #[validate(range(min = 1))]
    pub page_size: usize,

    #[validate(range(min = 1))]
    pub inner_dim: usize,

    #[validate(custom(function = "validate_power_of_2"))]
    #[builder(default = "1")]
    pub alignment: usize,

    #[builder(default = "DType::FP16")]
    pub dtype: DType,
}

impl LayoutConfig {
    pub fn builder() -> LayoutConfigBuilder {
        LayoutConfigBuilder::default()
    }
}

/// Validation function for Option<usize> to check if it's Some(power_of_2).
fn validate_power_of_2(alignment: usize) -> Result<(), validator::ValidationError> {
    if !alignment.is_power_of_two() {
        // Return validation error if alignment is not a power of 2
        return Err(validator::ValidationError::new(
            "alignment_must_be_power_of_2",
        ));
    }
    // Passes validation if alignment is a power of 2
    Ok(())
}

/// Helper to align a value up to the nearest multiple of alignment.
/// Alignment must be a power of 2.
fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Internal struct to hold calculated layout dimensions specific to FullyContiguous.
// Module-level, but only used internally by FullyContiguous
#[derive(Debug, Clone)]
struct FullyContiguousConfig {
    inner: LayoutConfig,
    memory_region_size: usize,
    layer_stride_in_bytes: usize,
    natural_block_stride: usize,
    block_stride_in_bytes: usize, // Aligned if necessary
    layout_data_bytes: usize,     // Size of the layout data itself (post base offset)
}

impl FullyContiguousConfig {
    /// Calculates the core dimensions based on the configuration.
    /// Returns an error if the configuration is invalid.
    fn new(config: LayoutConfig) -> Result<Self, LayoutError> {
        // Validate first, propagating errors via `?`
        config.validate()?;

        let alignment = config.alignment;
        let memory_region_size = config.page_size * config.inner_dim * config.dtype.size_in_bytes();
        let layer_stride_in_bytes = memory_region_size;
        let natural_block_stride = config.num_layers * layer_stride_in_bytes;

        let block_stride_in_bytes = if alignment > 1 {
            align_up(natural_block_stride, alignment)
        } else {
            natural_block_stride
        };

        let layout_data_bytes =
            (config.num_blocks - 1) * block_stride_in_bytes + natural_block_stride;

        Ok(Self {
            inner: config,
            memory_region_size,
            layer_stride_in_bytes,
            natural_block_stride,
            block_stride_in_bytes,
            layout_data_bytes,
        })
    }

    /// Calculate the total number of bytes required for allocation, including initial alignment padding.
    /// Panics if the provided configuration is invalid.
    pub fn required_allocation_size(&self) -> usize {
        let initial_padding = if self.inner.alignment > 1 {
            self.inner.alignment - 1
        } else {
            0
        };
        self.layout_data_bytes + initial_padding
    }
}

/// Contiguous memory layout where all blocks and layers are sequential
#[derive(Debug)]
pub struct FullyContiguous<S: Storage> {
    config: FullyContiguousConfig,
    storage: S,

    // Offset from storage.addr() to the aligned start of block 0
    base_offset: usize,
}

impl<S: Storage> FullyContiguous<S> {
    /// Create a new contiguous layout using the provided configuration and pre-allocated storage.
    #[instrument(level = "debug", skip(storage), fields(config = ?config))]
    pub fn new(config: LayoutConfig, storage: S) -> Result<Self, LayoutError> {
        // Calculate dimensions, which includes validation.
        // Propagate validation error if it occurs.
        let config = FullyContiguousConfig::new(config)?;

        let provided_size = storage.size();
        let storage_addr = storage.addr();
        let alignment = config.inner.alignment;

        // Calculate base offset needed to align the start of block 0
        let base_offset = if alignment > 1 {
            align_up(storage_addr as usize, alignment) - storage_addr as usize
        } else {
            0
        };

        let total_required_size_with_offset = base_offset + config.layout_data_bytes;

        tracing::debug!(
            provided_size,
            total_required_size_with_offset,
            base_offset,
            required_layout_data_bytes = config.layout_data_bytes,
            alignment,
            "Validating storage size with base offset and alignment"
        );

        // Validate storage size fits the configuration *with base offset and alignment*
        if provided_size < total_required_size_with_offset {
            tracing::warn!(
                provided_size,
                total_required_size_with_offset,
                "Storage size too small for aligned layout including base offset"
            );
            return Err(LayoutError::InvalidConfig(format!(
                "Storage size {} is less than required size {} (including base offset for alignment)",
                provided_size,
                total_required_size_with_offset
            )));
        }

        tracing::debug!(
            config.memory_region_size,
            config.layer_stride_in_bytes,
            config.block_stride_in_bytes,
            config.natural_block_stride,
            alignment = config.inner.alignment,
            base_offset,
            "Calculated layout strides (aligned)"
        );

        Ok(Self {
            config,
            storage,
            base_offset,
        })
    }

    /// Allocate storage using the provided allocator and create a new FullyContiguous layout.
    ///
    /// Calculates the required size based on the configuration, allocates the storage
    /// (including potential padding for initial alignment), and then constructs the
    /// `FullyContiguous` layout instance.
    ///
    /// # Type Parameters
    ///
    /// * `A`: The type of the storage allocator, implementing `StorageAllocator<S>`.
    ///
    /// # Arguments
    ///
    /// * `config` - The layout configuration.
    /// * `allocator` - A reference to the storage allocator.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `FullyContiguous<S>` instance or an error if allocation
    /// or layout creation fails.
    #[instrument(level = "debug", skip(allocator), fields(config = ?config))]
    pub fn allocate<A: StorageAllocator<S>>(
        config: LayoutConfig,
        allocator: &A,
    ) -> Result<Self, LayoutError> {
        // Calculate total bytes needed. Propagate error if config is invalid.
        let config = FullyContiguousConfig::new(config)?;
        let bytes_to_allocate = config.required_allocation_size();

        tracing::debug!(
            bytes_to_allocate,
            alignment = config.inner.alignment,
            "Calculated storage size for allocation (with alignment padding)"
        );

        let storage = allocator.allocate(bytes_to_allocate).map_err(|e| {
            LayoutError::OperationFailed(format!("Storage allocation failed: {}", e))
        })?;
        tracing::debug!(
            allocated_size = storage.size(),
            allocated_addr = storage.addr(),
            "Storage allocated successfully"
        );

        // Pass the config by value as Self::new takes ownership
        Self::new(config.inner, storage)
    }
}

impl<S: Storage> BlockLayout for FullyContiguous<S> {
    type StorageType = S;
}

impl<S: Storage> BlockLayoutConfig for FullyContiguous<S> {
    fn layout_type(&self) -> LayoutType {
        LayoutType::FullyContiguous
    }

    fn num_blocks(&self) -> usize {
        self.config.inner.num_blocks
    }

    fn num_layers(&self) -> usize {
        self.config.inner.num_layers
    }

    fn page_size(&self) -> usize {
        self.config.inner.page_size
    }

    fn inner_dim(&self) -> usize {
        self.config.inner.inner_dim
    }
}

impl<S: Storage> BlockLayoutLookup for FullyContiguous<S> {
    fn get_memory_region(&self, block_idx: usize, layer_idx: usize) -> Result<u64, LayoutError> {
        if block_idx >= self.num_blocks() {
            return Err(LayoutError::InvalidBlockIndex(block_idx));
        }

        if layer_idx >= self.num_layers() {
            return Err(LayoutError::InvalidLayerIndex(layer_idx));
        }

        // Start from the aligned base address
        let aligned_start_addr = self.storage.addr() + self.base_offset as u64;

        // Calculate offset relative to the aligned start using stored config
        let block_offset = block_idx * self.config.block_stride_in_bytes;
        let layer_offset = layer_idx * self.config.layer_stride_in_bytes;
        let final_addr = aligned_start_addr + block_offset as u64 + layer_offset as u64;

        Ok(final_addr)
    }

    fn memory_region_size(&self) -> usize {
        // Access via stored dims
        self.config.memory_region_size
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::block_manager::storage::tests::{NullDeviceAllocator, NullDeviceStorage};
    use crate::block_manager::storage::{StorageType, SystemAllocator};
    use crate::common::dtype::DType;
    use dynamo_runtime::logging::init as init_logging;

    const NUM_BLOCKS: usize = 7;
    const NUM_LAYERS: usize = 5;
    const PAGE_SIZE: usize = 4;
    const INNER_DIM: usize = 13;
    const DTYPE: DType = DType::FP32; // Example dtype

    // Updated setup_layout: Calculates size internally, uses default alignment for simplicity in non-alignment tests.
    pub fn setup_layout(
        alignment: Option<usize>, // Option to override default alignment
    ) -> Result<FullyContiguous<NullDeviceStorage>, LayoutError> {
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: alignment.unwrap_or(1),
            dtype: DTYPE,
        };

        FullyContiguous::allocate(config, &NullDeviceAllocator)
    }

    #[test]
    fn test_fc_creation_invalid_alignment() {
        let config = LayoutConfig::builder()
            .num_blocks(NUM_BLOCKS)
            .num_layers(NUM_LAYERS)
            .page_size(PAGE_SIZE)
            .inner_dim(INNER_DIM)
            .alignment(3)
            .build()
            .unwrap();

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_fc_creation_success() {
        // Setup with default (None) alignment
        let layout_result = setup_layout(None);
        assert!(
            layout_result.is_ok(),
            "Layout creation failed: {:?}",
            layout_result.err()
        );
    }

    #[test]
    fn test_fc_creation_insufficient_storage() {
        init_logging();
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: 1,
            dtype: DTYPE,
        };
        // Calculate correct size needed
        let fc_config = FullyContiguousConfig::new(config.clone()).unwrap();
        let required_size = fc_config.required_allocation_size();
        let storage = NullDeviceStorage::new((required_size - 1) as u64);
        let layout_result = FullyContiguous::new(config, storage);

        assert!(layout_result.is_err());
        match layout_result.err().unwrap() {
            LayoutError::InvalidConfig(_) => {} // Expected error
            e => panic!("Expected InvalidConfig error, got {:?}", e),
        }
    }

    #[test]
    fn test_fc_accessor_methods() {
        let layout = setup_layout(None).expect("Layout setup failed");

        assert_eq!(layout.num_blocks(), NUM_BLOCKS);
        assert_eq!(layout.num_layers(), NUM_LAYERS);
        assert_eq!(layout.page_size(), PAGE_SIZE);
        assert_eq!(layout.inner_dim(), INNER_DIM);
    }

    #[test]
    fn test_fc_memory_region_size() {
        let layout = setup_layout(None).expect("Layout setup failed");
        let expected_region_size = PAGE_SIZE * INNER_DIM * DTYPE.size_in_bytes();
        assert_eq!(layout.memory_region_size(), expected_region_size);
    }

    #[test]
    fn test_fc_offset_calculation() {
        let layout = setup_layout(None).expect("Layout setup failed");

        let dims = layout.config.clone();
        let block_stride = dims.block_stride_in_bytes;
        let layer_stride = dims.layer_stride_in_bytes;
        let base_addr = layout.storage.addr() + layout.base_offset as u64;

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
        let layout = setup_layout(None).expect("Layout setup failed");
        let result = layout.get_memory_region(NUM_BLOCKS, 0); // Index == num_blocks (out of bounds)
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidBlockIndex(NUM_BLOCKS)
        ));
    }

    #[test]
    fn test_fc_invalid_layer_index() {
        let layout = setup_layout(None).expect("Layout setup failed");
        let result = layout.get_memory_region(0, NUM_LAYERS); // Index == num_layers (out of bounds)
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            LayoutError::InvalidLayerIndex(NUM_LAYERS)
        ));
    }

    #[test]
    fn test_fc_allocation_system() {
        init_logging();
        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: 1,
            dtype: DTYPE,
        };

        let allocator = SystemAllocator::default();
        let layout_result = FullyContiguous::allocate(config, &allocator);

        assert!(layout_result.is_ok());
        let layout = layout_result.unwrap();

        // Basic checks on the allocated layout
        assert_eq!(layout.num_blocks(), NUM_BLOCKS);
        assert_eq!(layout.num_layers(), NUM_LAYERS);
        assert_eq!(layout.page_size(), PAGE_SIZE);
        assert_eq!(layout.inner_dim(), INNER_DIM);
        assert_eq!(layout.storage.storage_type(), StorageType::System);
        assert_eq!(
            layout.storage.size(),
            layout.config.required_allocation_size()
        );

        assert_eq!(
            layout.storage.size(),
            NUM_BLOCKS * NUM_LAYERS * PAGE_SIZE * INNER_DIM * DTYPE.size_in_bytes()
        );
    }

    #[test]
    fn test_fc_alignment() {
        init_logging();
        const ALIGNMENT: usize = 256; // Must be power of 2

        let config = LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: ALIGNMENT,
            dtype: DTYPE,
        };

        // Calculate expected size needed *for the data layout itself*
        let memory_region_size = PAGE_SIZE * INNER_DIM * DTYPE.size_in_bytes();
        assert_eq!(memory_region_size, 208);

        let natural_block_stride = NUM_LAYERS * memory_region_size;
        assert_eq!(natural_block_stride, 1040);

        let aligned_block_stride = align_up(natural_block_stride, ALIGNMENT);
        assert_eq!(aligned_block_stride, 1280);

        // Calculate the expected *allocated* size (data + initial padding)
        let fc_config = FullyContiguousConfig::new(config.clone()).unwrap();
        let expected_allocated_size = fc_config.required_allocation_size();

        // Use allocate method
        let allocator = SystemAllocator::default();
        let layout_result = FullyContiguous::allocate(config.clone(), &allocator);

        assert!(
            layout_result.is_ok(),
            "Allocation failed: {:?}",
            layout_result.err()
        );
        let layout = layout_result.unwrap();

        // Verify total *allocated* size matches expectation
        assert_eq!(
            layout.storage.size(),
            expected_allocated_size,
            "Allocated storage size mismatch"
        );
        assert_eq!(
            layout.config.block_stride_in_bytes, aligned_block_stride,
            "Stored block stride mismatch"
        );

        // Check alignment of block starts
        let addr_block_0 = layout
            .get_memory_region(0, 0)
            .expect("Failed to get addr block 0");
        let addr_block_1 = layout
            .get_memory_region(1, 0)
            .expect("Failed to get addr block 1");
        let addr_block_2 = layout
            .get_memory_region(2, 0)
            .expect("Failed to get addr block 2");

        // All blocks should now be aligned due to base_offset adjustment
        assert_eq!(
            addr_block_0 % ALIGNMENT as u64,
            0,
            "Block 0 start address is not aligned"
        );
        assert_eq!(
            addr_block_1 % ALIGNMENT as u64,
            0,
            "Block 1 start address is not aligned"
        );
        assert_eq!(
            addr_block_2 % ALIGNMENT as u64,
            0,
            "Block 2 start address is not aligned"
        );

        // Verify the difference matches the aligned stride
        assert_eq!(
            addr_block_1 - addr_block_0,
            aligned_block_stride as u64,
            "Stride between block 0 and 1 mismatch"
        );
        assert_eq!(
            addr_block_2 - addr_block_1,
            aligned_block_stride as u64,
            "Stride between block 1 and 2 mismatch"
        );
    }
}
