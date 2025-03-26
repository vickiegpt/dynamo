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

//! MemoryLayout extend Storage to define the memory layout of a block across one
//! or more memory regions.
//!
//! A layout defines the scope of contiguous memory for a block and the well-defined
//! or not-well-defined stride of the starting addresses of each layer within the block.
//!
//! The layout does not define the type of block being described, meaning that the data
//! within the [block_size, inner_dim] is not yet defined. This is the responsibility of
//! a higher-level structure.

mod lwcb;
mod lwscbl;

use super::{
    storage::{OwnedStorage, StorageType},
    Dimensions,
};
use crate::kv::storage::Storage;
use dynamo_runtime::{raise, Result};
use humanize_bytes::humanize_bytes_binary;
use validator::Validate;

pub trait MemoryLayout {
    /// The [configuration object][Dimensions] that defines the layout and storage requirements
    fn dimensions(&self) -> &Dimensions;

    /// Whether the blocks and layers are contiguous.

    /// If true, then the an entire block and its internal layers are contiguous in memory
    /// and can be copied with a single memory copy or nixl rdma transfer.
    fn is_contiguous(&self) -> bool;
}

impl Dimensions {
    /// Total number of bytes required to store all blocks.
    ///
    /// The blocks may or may not be contiguous in memory.
    ///
    /// If not contiguous, the value returned represents the minimum storage size
    /// required to store all the blocks.
    pub fn expected_storage_size(&self) -> usize {
        self.n_blocks * self.expected_storage_per_block()
    }

    /// Total number of bytes required to store a single block.
    ///
    /// The block may or may not be contiguous in memory.
    ///
    /// If not contiguous, the value returned represents the minimum storage size
    /// required to store the block.
    pub fn expected_storage_per_block(&self) -> usize {
        self.n_layers * self.expected_storage_per_block_per_layer()
    }

    /// Total number of bytes for the smallest guaranteed contiguous block
    /// of memory for a single layer.
    pub fn expected_storage_per_block_per_layer(&self) -> usize {
        self.block_size * self.inner_dim * self.dtype.size_in_bytes()
    }
}

/// Blocks are layer-wise contiguous.
///
/// A block can be fully described by a single slice on the outer dimension
/// Requires a single nixl memory registration
#[derive(Debug, Clone, Validate)]
pub struct LayerWiseContiguousBlockLayout {
    dimensions: Dimensions,
    storage: OwnedStorage,
}

/// Blocks are non-contiguous
/// Each layer is defined by a contiguous slab of memory
/// The block can be described by a single offset per layer; thus requires a per-layer
/// starting address + common offset to to describe the layer_ptr starting address
/// Requires `n_layer` nixl memory registrations
#[derive(Debug, Clone, Validate)]
pub struct LayerWiseSlabContiguousBlockLayout {
    dimensions: Dimensions,
    layers: Vec<OwnedStorage>,
}
