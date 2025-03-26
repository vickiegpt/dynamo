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

// pub mod agent;
pub mod block;
pub mod layer;
pub mod layouts;
pub mod manager;
pub mod reserved;
pub mod reuse;
pub mod sequence;
pub mod storage;

use layer::KvBlockStorage;
use reserved::*;
use storage::OwnedStorage;

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{atomic::AtomicU64, Arc, RwLock},
};

use async_trait::async_trait;
use derive_builder::Builder;
use derive_getters::{Dissolve, Getters};
use dynamo_runtime::{
    raise,
    utils::pool::{PoolExt, PoolItem, PoolValue, Returnable, SharedPoolItem},
    Error, Result,
};
use validator::Validate;

use crate::tokens::{PartialTokenBlock, SequenceHash, TokenBlock, Tokens};

use tracing as log;

pub type UniqueBlock<T> = PoolItem<KvBlock<T>>;
pub type SharedBlock<T> = SharedPoolItem<KvBlock<T>>;

/// Memory layout type for inner dimensions, i.e. the KV/latent dimension
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// [2, block_size, inner_dim/2] - KV dimension first
    KvFirst,

    /// [block_size, 2, inner_dim/2] - Block dimension first
    BlockFirst,

    /// MLA =>[block_size, inner_dim] where inner_dim = latent_size
    MLA,

    /// Custom layout that requires special handling
    Custom,
}

/// Storage pattern for layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerConfiguration {
    /// All layers are contiguous in memory [n_layers, ...]
    Contiguous,

    /// Each layer is stored separately with no guaranteed stride
    NonContiguous,
}

/// Blocks can have different storage strategies
pub trait BlockStorage {
    // /// Common dimensions for the block storage
    // fn dimensions(&self) -> &Dimensions;

    /// Whether the blocks and layers are contiguous.
    ///
    /// If true, then the an entire block and its internal layers are contiguous in memory
    /// and can be copied with a single memory copy or nixl rdma transfer.
    fn is_contiguous(&self) -> bool;

    // /// Returns the memory layout pattern of this storage
    // fn memory_layout(&self) -> MemoryLayout;

    // /// Number of tokens in the block
    // fn block_size(&self) -> usize;

    // /// Hidden dimension size
    // fn inner_dim(&self) -> usize;

    // /// Number of layers in the block
    // fn n_layers(&self) -> usize;

    // /// Pointer to the starting memory address for the entire block
    // /// Only meaningful for LayerStorage::Contiguous
    // fn block_ptr(&self) -> Result<Option<u64>, Error>;

    // /// Pointer to layer's inner memory region
    // /// For KvFirst layout: pointer to [2, block_size, inner_dim/2] for this layer
    // /// For BlockFirst layout: pointer to [block_size, 2, inner_dim/2] for this layer
    // /// For MLA layout: pointer to [block_size, inner_dim] for this layer
    // fn layer_ptr(&self, layer_id: usize) -> Result<u64, Error>;

    // /// Pointer to the key tensor for a specific layer
    // fn k_ptr(&self, block_id: usize, layer_id: usize) -> Result<u64, Error>;

    // /// Pointer to the value tensor for a specific layer
    // fn v_ptr(&self, layer_id: usize) -> Result<u64, Error>;

    // /// Size in bytes of one layer's data
    // fn bytes_per_layer(&self) -> usize;

    // /// Size in bytes of key or value data for one layer
    // fn bytes_per_layer_per_k_or_v(&self) -> usize;

    // /// Stride between layers in bytes (only relevant for contiguous storage)
    // fn layer_stride(&self) -> Option<usize>;

    /// Check if this storage is compatible for direct transfer with another storage
    fn is_compatible_with<T: BlockStorage + ?Sized>(&self, other: &T) -> bool;

    // /// Get a description of the memory layout for debugging
    // fn layout_description(&self) -> String {
    //     format!(
    //         "{}:{},  n_layers={}, block_size={}, inner_dim={}",
    //         match self.is_contiguous() {
    //             true => "Contiguous",
    //             false => "NonContiguous",
    //         },
    //         match self.memory_layout() {
    //             MemoryLayout::KvFirst => "KvFirst",
    //             MemoryLayout::BlockFirst => "BlockFirst",
    //             MemoryLayout::MLA => "MLA",
    //             MemoryLayout::Custom => "Custom",
    //         },
    //         self.n_layers(),
    //         self.block_size(),
    //         self.inner_dim(),
    //     )
    // }
}

#[derive(Debug, Clone, Default)]
pub struct NullStorage {
    dimensions: Dimensions,
}

impl BlockStorage for NullStorage {
    // fn dimensions(&self) -> &Dimensions {
    //     &self.dimensions
    // }

    // fn is_contiguous(&self) -> bool {
    //     false
    // }

    // fn memory_layout(&self) -> MemoryLayout {
    //     MemoryLayout::Custom
    // }

    // fn block_size(&self) -> usize {
    //     0
    // }

    // fn inner_dim(&self) -> usize {
    //     0
    // }

    // fn n_layers(&self) -> usize {
    //     0
    // }

    // fn block_ptr(&self) -> Result<Option<u64>, Error> {
    //     Ok(None)
    // }

    // fn inner_ptr(&self, _block_id: usize, _layer_id: usize) -> Result<u64, Error> {
    //     Ok(0)
    // }

    // fn bytes_per_layer(&self) -> usize {
    //     0
    // }

    // fn bytes_per_layer_per_k_or_v(&self) -> usize {
    //     0
    // }

    // fn layer_stride(&self) -> Option<usize> {
    //     None
    // }

    fn is_compatible_with<T: BlockStorage + ?Sized>(&self, _other: &T) -> bool {
        false
    }
}

#[derive(Debug, Clone, Default)]
pub struct KvBlock<T: BlockStorage + Send + Sync> {
    token_block: TokenBlock,
    priority: u32,
    return_tick: u64,
    storage: T,
}

impl<T: BlockStorage + Send + Sync> KvBlock<T> {
    /// Creates a new KvBlock with the given token block
    pub fn new(token_block: TokenBlock) -> KvBlock<NullStorage> {
        let storage = NullStorage {};
        KvBlock {
            token_block,
            priority: 0,
            return_tick: 0,
            storage,
        }
    }

    /// Updates the token block
    pub fn update_token_block(&mut self, token_block: TokenBlock) {
        self.token_block = token_block;
    }

    /// Resets the block to its initial state
    pub(crate) fn reset(&mut self) {
        self.token_block = TokenBlock::default();
        self.priority = 0;
        self.return_tick = 0;
    }
}

impl<T: BlockStorage + Send + Sync + 'static> Returnable for KvBlock<T> {
    fn on_return(&mut self) {}
}

#[derive(Debug, Clone, Copy, Builder, Validate, Getters)]
pub struct Dimensions {
    #[getter(copy)]
    #[validate(range(min = 1))]
    n_blocks: usize,

    #[getter(copy)]
    #[validate(range(min = 1))]
    n_layers: usize,

    #[getter(copy)]
    #[validate(range(min = 1))]
    block_size: usize,

    #[getter(copy)]
    #[validate(range(min = 1))]
    inner_dim: usize,

    #[getter(copy)]
    dtype: DType,
}

pub struct NixlContext {}
pub struct NixlDescriptor {}
pub struct NixlRegisteredStorage {
    storage: OwnedStorage,
    descriptor: NixlDescriptor,
    context: Arc<NixlContext>,
}
