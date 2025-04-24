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

pub mod contiguous;

use crate::dtype::DType;
use crate::storage::StorageType;

use derive_builder::Builder;
use thiserror::Error;
use tracing::instrument;
use validator::Validate;

/// Errors that can occur during layout operations
#[derive(Debug, Error)]
#[allow(missing_docs)]
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
    fn get_memory_region(&self, block_idx: usize, layer_idx: usize) -> Result<u64, LayoutError>;

    /// Get the memory region for a specific page [page_size, inner_dim]
    fn memory_region_size(&self) -> usize;

    /// Returns the data type of the layout
    fn dtype(&self) -> DType;

    /// Returns the storage type of the layout
    fn storage_type(&self) -> StorageType;
}

/// Configuration for block layouts
#[derive(Debug, Clone, Builder, Validate)]
pub struct LayoutConfig {
    /// Number of blocks in the layout
    #[validate(range(min = 1))]
    pub num_blocks: usize,

    /// Number of layers per block
    #[validate(range(min = 1))]
    pub num_layers: usize,

    /// Number of pages per block
    #[validate(range(min = 1))]
    pub page_size: usize,

    /// Inner dimension size
    #[validate(range(min = 1))]
    pub inner_dim: usize,

    /// Alignment for the layout
    #[validate(custom(function = "validate_power_of_2"))]
    #[builder(default = "1")]
    pub alignment: usize,

    /// Data type for the layout
    #[builder(default = "DType::FP16")]
    pub dtype: DType,
}

fn is_power_of_2(n: usize) -> bool {
    // Check if n is not zero and is a power of two.
    n != 0 && (n & (n - 1)) == 0
}

/// Validation function for Option<usize> to check if it's Some(power_of_2).
fn validate_power_of_2(alignment: usize) -> Result<(), validator::ValidationError> {
    if !is_power_of_2(alignment) {
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
