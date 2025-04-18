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

//! # KV Block Pools
//!
//! The InactiveBlockPool manages KV blocks that are not actively in use but retain their previous state.
//!
//! ## Key Features:
//!
//! - **State Preservation**: Blocks in the pool maintain their previous state and can be reused.
//!
//! - **Priority-Based FIFO**: Blocks are returned in first-in, first-out order within their priority levels.
//!   Lower priority values are processed first, allowing important blocks to be retained longer.
//!
//! - **State Matching**: Blocks can be matched against their previous state instead of being taken randomly,
//!   enabling efficient reuse of blocks with specific sequence hashes.
//!
//! - **Priority Management**: Priorities can be applied to blocks based on their sequence hash,
//!   requiring some external knowledge of the block's characteristics.
//!
//! - **State Management**: Blocks can have their states wiped clean/reset individually or in groups.
//!   The entire pool can also be reset as needed.
//!
//! - **Synchronization**: Fence operations ensure all higher priority operations have completed
//!   before proceeding. Note that this is not a true fence - higher priority operations issued
//!   after the fence will still be processed before the fence completes.
//!
//! The [ActiveBlockPool] manages KV blocks that are actively in use.

pub mod active;
mod inactive;

use derive_getters::Dissolve;
use dynamo_runtime::{
    raise,
    utils::pool::{PoolExt, PoolItem, PoolValue, ReturnHandle, SharedPoolItem},
    Result,
};
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    ops::Deref,
    sync::Arc,
};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use crate::{
    block_manager::block::BlockState,
    tokens::{SequenceHash, TokenBlock},
};

use super::block::{Block, BlockMetadata};
use super::storage::Storage;

pub type BlockType<S, M> = Block<S, M>;
pub type UniqueBlock<S, M> = PoolItem<Block<S, M>>;
pub type SharedBlock<S, M> = SharedPoolItem<Block<S, M>>;

#[derive(Debug, thiserror::Error)]
pub enum BlockPoolError {
    #[error("No blocks available: requested {0}, available {1}")]
    InsufficientBlocksAvailable(usize, usize),
}

#[derive(Default)]
pub struct InactiveBlockPool<S: Storage, M: BlockMetadata> {
    // Direct lookup by sequence_hash
    lookup_map: HashMap<SequenceHash, BlockType<S, M>>,

    // Ordered by timestamp (oldest first)
    priority_set: BTreeSet<PriorityKey<M>>,

    // Fully Uninitialized
    uninitialized_set: VecDeque<BlockType<S, M>>,

    // Return Tick
    return_tick: u64,

    // Total blocks
    total_blocks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriorityKey<M: BlockMetadata> {
    metadata: M,
    sequence_hash: SequenceHash,
}

impl<M: BlockMetadata> PriorityKey<M> {
    fn new(metadata: M, sequence_hash: SequenceHash) -> Self {
        Self {
            metadata,
            sequence_hash,
        }
    }

    pub fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }

    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    pub fn update_metadata(&mut self, metadata: M) {
        self.metadata = metadata;
    }
}

// customize ord and partial ord for to store first by priority (lowest to highest), then by return_tick (lowest to highest)
impl<M: BlockMetadata> PartialOrd for PriorityKey<M> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<M: BlockMetadata> Ord for PriorityKey<M> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.metadata
            .cmp(&other.metadata)
            .then(self.sequence_hash.cmp(&other.sequence_hash))
    }
}
