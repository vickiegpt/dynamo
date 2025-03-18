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

pub mod layer;
pub mod manager;
pub mod reserved;
pub mod reuse;
pub mod sequence;
pub mod storage;

use layer::KvBlockStorage;
use reserved::*;

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{atomic::AtomicU64, Arc, RwLock},
};

use async_trait::async_trait;
use derive_getters::Dissolve;
use dynamo_runtime::{
    raise,
    utils::pool::{PoolExt, PoolItem, PoolValue, Returnable, SharedPoolItem},
    Result,
};

use crate::tokens::{PartialTokenBlock, SequenceHash, TokenBlock, Tokens};

use tracing as log;

pub type UniqueBlock<T> = PoolItem<KvBlock<T>>;
pub type SharedBlock<T> = SharedPoolItem<KvBlock<T>>;

#[derive(Debug, Clone, Default)]
pub struct KvBlock<T: BlockStorage + Send + Sync> {
    token_block: TokenBlock,
    priority: u32,
    return_tick: u64,
    storage: T,
}

#[derive(Debug, Clone, Default)]
pub struct NullStorage {}

pub trait BlockStorage {
    /// Pointer to the leading element of the key tensor for layer `layer_id`
    /// The `u64` can be converted to a void* for use in C code
    fn k_ptr(&self, layer_id: usize) -> Result<u64, anyhow::Error>;

    /// Pointer to the leading element of the value tensor for layer `layer_id`
    /// The `u64` can be converted to a void* for use in C code
    fn v_ptr(&self, layer_id: usize) -> Result<u64, anyhow::Error>;

    // /// Number of key-value pairs per block. This is the number of tokens in
    // /// represented by a single block.
    // fn tokens_count_per_block(&self) -> usize;

    /// Size of the key and value tensors for layer `layer_id` in bytes
    /// This is the size of the key or value tensor in bytes. The key and value
    /// tensors may or may not be contiguous in memory. However, each key or value
    /// tensor corresponds to a single block is contiguous in memory.
    fn bytes_per_block_per_k_or_v(&self) -> usize;

    /// Returns true if the key and value tensors for layer `layer_id` are contiguous
    /// in memory. If true, the starting address of the concatenated key and value
    /// tensors can be computed as the memory region starting at:
    /// `k_ptr(layer_id) + 2*bytes_per_block_per_k_or_v(layer_id)`
    fn k_and_v_are_contiguous(&self) -> bool;
}

impl BlockStorage for NullStorage {
    fn k_ptr(&self, _layer_id: usize) -> Result<u64, anyhow::Error> {
        Ok(0 as u64)
    }

    fn v_ptr(&self, _layer_id: usize) -> Result<u64, anyhow::Error> {
        Ok(0 as u64)
    }

    // fn tokens_count_per_block(&self) -> usize {
    //     0 as usize
    // }

    fn bytes_per_block_per_k_or_v(&self) -> usize {
        0 as usize
    }

    fn k_and_v_are_contiguous(&self) -> bool {
        false
    }
}

// pub struct KvStorage {
//     data: u64,
//     size: usize,

//     layer_idx: usize,
//     block_idx: usize,

//     /// The layout of the tensor
//     layout: layer::KvLayer,
// }

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
        // self.storage = None;
        // self.storage_state = StorageState::Absent;
    }
}

impl<T: BlockStorage + Send + Sync + 'static> Returnable for KvBlock<T> {
    fn on_return(&mut self) {}
}

pub struct KvBlockConfig {}

// Host memory but special class of host memory
pub struct PinnedBlockStorage {
    block_id: usize,
    block_storage: Arc<KvBlockStorage>,
}
pub struct DeviceBlockStorage {
    block_id: usize,
    block_storage: Arc<KvBlockStorage>,
}

pub type KvBlockPinned = KvBlock<PinnedBlockStorage>;
pub type KvBlockDevice = KvBlock<DeviceBlockStorage>;

impl BlockStorage for PinnedBlockStorage {
    fn k_ptr(&self, layer_id: usize) -> Result<u64> {
        self.block_storage.k_ptr(self.block_id, layer_id)
    }

    fn v_ptr(&self, layer_id: usize) -> Result<u64> {
        self.block_storage.v_ptr(self.block_id, layer_id)
    }

    fn bytes_per_block_per_k_or_v(&self) -> usize {
        self.block_storage.bytes_per_block_per_k_or_v()
    }

    fn k_and_v_are_contiguous(&self) -> bool {
        self.block_storage.k_and_v_are_contiguous()
    }
}

impl BlockStorage for DeviceBlockStorage {
    fn k_ptr(&self, layer_id: usize) -> Result<u64> {
        self.block_storage.k_ptr(self.block_id, layer_id)
    }

    fn v_ptr(&self, layer_id: usize) -> Result<u64> {
        self.block_storage.v_ptr(self.block_id, layer_id)
    }

    fn bytes_per_block_per_k_or_v(&self) -> usize {
        self.block_storage.bytes_per_block_per_k_or_v()
    }

    fn k_and_v_are_contiguous(&self) -> bool {
        self.block_storage.k_and_v_are_contiguous()
    }
}
