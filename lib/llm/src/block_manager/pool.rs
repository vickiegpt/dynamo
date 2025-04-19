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

mod active;
mod inactive;
mod priority_key;
mod state;

use inactive::InactiveBlockPool;
use priority_key::PriorityKey;

use dynamo_runtime::{
    utils::pool::{PoolItem, SharedPoolItem},
    Result,
};
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, Weak},
};

use super::block::*;
use super::events::*;
use super::storage::Storage;

use crate::tokens::{SequenceHash, TokenBlock};

pub type BlockType<S, M> = Block<S, M>;
pub type UniqueBlock<S, M> = PoolItem<Block<S, M>>;
pub type SharedBlock<S, M> = SharedPoolItem<Block<S, M>>;

#[derive(Debug, thiserror::Error)]
pub enum BlockPoolError {
    #[error("Block is not complete")]
    BlockNotComplete,

    #[error("Not enough blocks available, requested: {0}, available: {1}")]
    NotEnoughBlocksAvailable(usize, usize),

    #[error("Invalid MutableBlock: {0}")]
    InvalidMutableBlock(String),

    #[error("Failed to register block: {0}")]
    FailedToRegisterBlock(String),
}

/// Manages the blocks in a specific storage backend
pub struct BlockPool<S: Storage, M: BlockMetadata> {
    inner: Arc<Mutex<State<S, M>>>,
}

pub struct MutableBlock<S: Storage, M: BlockMetadata> {
    block: Option<Block<S, M>>,
    state: Arc<Mutex<State<S, M>>>,
}

pub struct ImmutableBlock<S: Storage, M: BlockMetadata> {
    block: Arc<MutableBlock<S, M>>,
}

struct ActiveBlockPool<S: Storage, M: BlockMetadata> {
    map: HashMap<SequenceHash, Weak<MutableBlock<S, M>>>,
}

struct State<S: Storage, M: BlockMetadata> {
    active: ActiveBlockPool<S, M>,
    inactive: InactiveBlockPool<S, M>,
    events: Arc<dyn EventManager>,
}

impl<S: Storage, M: BlockMetadata> BlockPool<S, M> {
    /// Creates a new [BlockPool] with the given [EventManager]
    /// On creation, the manager will not have any blocks allocated
    pub fn new(events: Arc<dyn EventManager>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(State::new(events))),
        }
    }

    /// Adds a list of blocks to the inactive pool
    pub fn add_blocks(&self, blocks: Vec<Block<S, M>>) {
        let mut state = self.inner.lock().unwrap();
        state.inactive.add_blocks(blocks);
    }

    /// Attempts to allocate a number of blocks from the inactive pool
    /// If there are not enough blocks available, it will return an error
    pub fn allocate_blocks(
        &mut self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, M>>, BlockPoolError> {
        let mut state = self.inner.lock().unwrap();
        state.allocate_blocks(count, self.inner.clone())
    }

    /// Registers [MutableBlock] with the [BlockPool]
    ///
    /// The result will be an [ImmutableBlock] which may or may not be the same storage block
    /// that was passed in as a parameter.
    ///
    /// This accounts for the inflight cases where two identical blocks are created near in time.
    /// The first block with a common [SequenceHash] will be registered. Any subsequent blocks with
    /// the same [SequenceHash] will have its [MutableBlock] returned to the pool
    /// as an [ImmutableBlock]
    pub fn register_block(
        &mut self,
        block: MutableBlock<S, M>,
    ) -> Result<ImmutableBlock<S, M>, BlockPoolError> {
        self.inner.lock().unwrap().register_block(block)
    }

    /// Attempts to match the given [SequenceHash] to an existing block
    ///
    /// Matches will be attempted in the following order:
    /// - Active pool
    /// - Inactive pool
    pub fn match_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Option<ImmutableBlock<S, M>> {
        let mut state = self.inner.lock().unwrap();

        if let Some(immutable) = state.active.match_sequence_hash(sequence_hash) {
            Some(immutable)
        } else if let Some(block) = state.inactive.match_sequence_hash(sequence_hash) {
            assert!(block.is_registered(), "block is not registered");
            let shared = Arc::new(MutableBlock {
                block: Some(block),
                state: self.inner.clone(),
            });

            state
                .active
                .map
                .insert(sequence_hash, Arc::downgrade(&shared));

            Some(ImmutableBlock { block: shared })
        } else {
            None
        }
    }
}

impl<S: Storage, M: BlockMetadata> MutableBlock<S, M> {
    fn into_parts(mut self) -> (Option<Block<S, M>>, Arc<Mutex<State<S, M>>>) {
        let block = self.block.take();
        let state = Arc::clone(&self.state);
        (block, state)
    }
}

impl<S: Storage, M: BlockMetadata> Drop for MutableBlock<S, M> {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            // TODO: move return_block to the state
            self.state.lock().unwrap().inactive.return_block(block);
        }
    }
}

impl<S: Storage, M: BlockMetadata> Deref for MutableBlock<S, M> {
    type Target = Block<S, M>;

    fn deref(&self) -> &Self::Target {
        self.block.as_ref().expect("block was dropped")
    }
}

impl<S: Storage, M: BlockMetadata> DerefMut for MutableBlock<S, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.block.as_mut().expect("block was dropped")
    }
}

impl<S: Storage, M: BlockMetadata> Deref for ImmutableBlock<S, M> {
    type Target = Block<S, M>;
    fn deref(&self) -> &Self::Target {
        self.block
            .as_ref()
            .block
            .as_ref()
            .expect("block was dropped")
    }
}
