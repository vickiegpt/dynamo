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
    pub fn new(config: LayoutConfig, storage: S) -> Result<Self> {
        // Validate storage size fits [n_blocks, n_layers, page_size, inner_dim]
        if storage.size()
            < (config.num_blocks * config.num_layers * config.page_size * config.inner_dim)
        {
            return Err(LayoutError::InvalidConfig(format!(
                "Storage size {} is less than required size {}",
                storage.size(),
                config.page_size
                    * config.inner_dim
                    * config.num_blocks
                    * config.num_layers
                    * config.dtype.size_in_bytes()
            )));
        }

        // Contiguous memory region on which the application uses for its own purposes
        // [page_size, inner_dim] elements
        let memory_region = config.page_size * config.inner_dim;
        let memory_region_size = memory_region * config.dtype.size_in_bytes();

        let layer_stride_in_bytes = memory_region * config.dtype.size_in_bytes();
        let block_stride_in_bytes = config.num_layers * layer_stride_in_bytes;

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

#[derive(Debug)]
pub struct NullLayout {
    config: LayoutConfig,
}

impl NullLayout {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            config: LayoutConfig {
                num_blocks,
                num_layers: 0,
                page_size: 0,
                inner_dim: 0,
                dtype: DType::U8,
            },
        }
    }
}

impl BlockLayout for NullLayout {
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

    fn get_memory_region(&self, _block_idx: usize, _layer_idx: usize) -> Result<u64> {
        Ok(0)
    }

    fn memory_region_size(&self) -> usize {
        0
    }
}
