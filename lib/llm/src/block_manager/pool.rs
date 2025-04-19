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

//! # KV Cache Block Pool Management
//!
//! This module provides the primary [`BlockPool`] structure for managing KV cache blocks.
//! It orchestrates the allocation, registration, and reuse of blocks by coordinating
//! between an [`ActiveBlockPool`] and an [`InactiveBlockPool`].
//!
//! ## Core Components:
//!
//! - **[`BlockPool`]**: The main entry point for interacting with the block management system.
//!   It holds the shared state containing both active and inactive pools.
//! - **[`ActiveBlockPool`]**: Manages blocks that are currently associated with active sequences.
//!   It primarily uses weak references to track these blocks, allowing them to be potentially
//!   reclaimed by the inactive pool if no strong references remain.
//! - **[`InactiveBlockPool`]**: Manages blocks that are not currently in active use. It supports
//!   block reuse by matching sequence hashes and employs a priority-based eviction strategy
//!   for acquiring free blocks.
//! - **[`MutableBlock`]**: Represents a uniquely owned block, typically obtained from allocation.
//!   It allows modification and is returned to the inactive pool upon being dropped.
//! - **[`ImmutableBlock`]**: Represents a shared, immutable reference to a block, usually after
//!   it has been registered or matched. Ensures that multiple sequences can reference the
//!   same underlying block data.
//!
//! ## Workflow:
//!
//! 1.  Blocks are initially added to the [`BlockPool`] via [`BlockPool::add_blocks`], populating the
//!     [`InactiveBlockPool`].
//! 2.  Sequences request blocks via [`BlockPool::allocate_blocks`], which attempts to acquire them
//!     from the [`InactiveBlockPool`]. This returns [`MutableBlock`]s.
//! 3.  Once a [`MutableBlock`] is filled and ready, it's registered using [`BlockPool::register_block`].
//!     This process checks the both the [`ActiveBlockPool`] and the [`InactiveBlockPool`] for existing blocks
//!     with the same content hash. It returns an [`ImmutableBlock`] representing the canonical block
//!     (either the one provided or an existing one).
//! 4.  Sequences can also try to reuse blocks directly using [`BlockPool::match_sequence_hash`], which
//!     checks both the active and inactive pools.
//! 5.  When an [`ImmutableBlock`] is no longer needed by any sequence (its `Arc` count drops to zero),
//!     the underlying [`MutableBlock`] (if it still exists via the weak reference in the active pool)
//!     can eventually be returned to the [`InactiveBlockPool`] when its final strong reference (the `Arc`
//!     within `ImmutableBlock`) is dropped.
//! 6.  Dropped [`MutableBlock`]s are automatically returned to the [`InactiveBlockPool`].

mod active;
mod inactive;
mod priority_key;
mod state;

use active::ActiveBlockPool;
use inactive::InactiveBlockPool;
use priority_key::PriorityKey;

use super::block::{Block, BlockMetadata};
use super::events::EventManager;
use super::storage::Storage;

use crate::tokens::{SequenceHash, TokenBlock};

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, Weak},
};

use dynamo_runtime::{
    utils::pool::{PoolItem, SharedPoolItem},
    Result,
};

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

struct State<S: Storage, M: BlockMetadata> {
    active: ActiveBlockPool<S, M>,
    inactive: InactiveBlockPool<S, M>,
    events: Arc<dyn EventManager>,
}

impl<S: Storage, M: BlockMetadata> BlockPool<S, M> {
    /// Creates a new [`BlockPool`] with the given [`EventManager`].
    ///
    /// The pool starts empty and requires blocks to be added via [`add_blocks`].
    ///
    /// # Arguments
    ///
    /// * `events` - An [`Arc<dyn EventManager>`] used for publishing block registration/removal events.
    ///
    /// # Returns
    ///
    /// A new [`BlockPool`] instance.
    pub fn new(events: Arc<dyn EventManager>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(State::new(events))),
        }
    }

    /// Adds a vector of [`Block`]s to the [`InactiveBlockPool`].
    ///
    /// These blocks are typically created from a [`super::block::BlockStorageCollection`]
    /// and represent the initial set of available cache blocks.
    /// Blocks added this way are initially reset.
    ///
    /// # Arguments
    ///
    /// * `blocks` - A [`Vec<Block<S, M>>`] to add to the inactive pool.
    pub fn add_blocks(&self, blocks: Vec<Block<S, M>>) {
        let mut state = self.inner.lock().unwrap();
        state.inactive.add_blocks(blocks);
    }

    /// Attempts to allocate a specified number of free blocks from the [`InactiveBlockPool`].
    ///
    /// Blocks acquired this way are returned as [`MutableBlock`]s, granting unique ownership
    /// and allowing modification. Dropping a [`MutableBlock`] automatically returns it
    /// to the [`InactiveBlockPool`].
    ///
    /// # Arguments
    ///
    /// * `count` - The number of blocks to allocate.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing:
    /// - `Ok(Vec<MutableBlock<S, M>>)`: If successful, a vector of allocated mutable blocks.
    /// - `Err(BlockPoolError)`: If not enough blocks are available in the inactive pool.
    pub fn allocate_blocks(
        &mut self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, M>>, BlockPoolError> {
        let mut state = self.inner.lock().unwrap();
        state.allocate_blocks(count, self.inner.clone())
    }

    /// Registers a [`MutableBlock`] (presumably after filling it) with the pool,
    /// making it potentially available for sharing via the [`ActiveBlockPool`].
    ///
    /// This function checks if a block with the same sequence hash already exists
    /// in the active pool. If so, it returns an [`ImmutableBlock`] pointing to the
    /// existing block, and the provided `block` is implicitly dropped (returned to
    /// the inactive pool). If no matching block exists, the provided `block` is
    /// added to the active pool (via a weak reference) and an [`ImmutableBlock`]
    /// pointing to it is returned.
    ///
    /// # Arguments
    ///
    /// * `block` - The [`MutableBlock<S, M>`] to register.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing:
    /// - `Ok(ImmutableBlock<S, M>)`: An immutable, shareable reference to the registered block.
    /// - `Err(BlockPoolError)`: If the provided block is in an invalid state (e.g., has no sequence hash).
    pub fn register_block(
        &mut self,
        block: MutableBlock<S, M>,
    ) -> Result<ImmutableBlock<S, M>, BlockPoolError> {
        self.inner.lock().unwrap().register_block(block)
    }

    /// Attempts to match the given [`SequenceHash`] to an existing block, checking
    /// both the active and inactive pools.
    ///
    /// Checks the [`ActiveBlockPool`] first. If a valid strong reference exists, it returns
    /// an [`ImmutableBlock`] cloned from it. If the weak reference exists but is stale,
    /// it's removed.
    ///
    /// If not found in the active pool, it checks the [`InactiveBlockPool`]. If found there,
    /// the block is moved to the active pool (tracked by a weak reference) and returned
    /// as a new [`ImmutableBlock`].
    ///
    /// # Arguments
    ///
    /// * `sequence_hash` - The [`SequenceHash`] to look for.
    ///
    /// # Returns
    ///
    /// An [`Option<ImmutableBlock<S, M>>`] containing the shared block if found, otherwise `None`.
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
                .insert_weak_block_ref(sequence_hash, Arc::downgrade(&shared));

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
