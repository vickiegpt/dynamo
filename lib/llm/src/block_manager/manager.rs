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

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock, Weak};

use crate::tokens::SequenceHash;

pub use super::block::*;
pub use super::events::*;
pub use super::pool::active::*;
pub use super::pool::*;
pub use super::storage::Storage;

#[derive(Debug, thiserror::Error)]
pub enum BlockStorageManagerError {
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
pub struct BlockStorageManager<S: Storage, M: BlockMetadata> {
    inner: Arc<RwLock<State<S, M>>>,
}

impl<S: Storage, M: BlockMetadata> BlockStorageManager<S, M> {
    /// Creates a new [BlockStorageManager] with the given [EventManager]
    /// On creation, the manager will not have any blocks allocated
    pub fn new(events: Arc<dyn EventManager>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(State::new(events))),
        }
    }

    /// Adds a list of blocks to the inactive pool
    pub fn add_blocks(&self, blocks: Vec<Block<S, M>>) {
        let mut state = self.inner.write().unwrap();
        state.inactive.add_blocks(blocks);
    }

    /// Attempts to allocate a number of blocks from the inactive pool
    /// If there are not enough blocks available, it will return an error
    pub fn allocate_blocks(
        &mut self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, M>>, BlockStorageManagerError> {
        let mut state = self.inner.write().unwrap();
        state.allocate_blocks(count, self.inner.clone())
    }

    /// Registers [MutableBlock] with the [BlockStorageManager]
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
    ) -> Result<ImmutableBlock<S, M>, BlockStorageManagerError> {
        self.inner.write().unwrap().register_block(block)
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
        let mut state = self.inner.write().unwrap();

        if let Some(immutable) = state.active.match_sequence_hash(sequence_hash) {
            return Some(immutable);
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

struct State<S: Storage, M: BlockMetadata> {
    active: ActiveBlockMap<S, M>,
    inactive: InactiveBlockPool<S, M>,
    events: Arc<dyn EventManager>,
}

impl<S: Storage, M: BlockMetadata> State<S, M> {
    fn new(events: Arc<dyn EventManager>) -> Self {
        Self {
            active: ActiveBlockMap::new(),
            inactive: InactiveBlockPool::new(),
            events,
        }
    }

    pub fn allocate_blocks(
        &mut self,
        count: usize,
        state: Arc<RwLock<State<S, M>>>,
    ) -> Result<Vec<MutableBlock<S, M>>, BlockStorageManagerError> {
        let available_blocks = self.inactive.available_blocks() as usize;

        if available_blocks < count {
            tracing::debug!(
                "not enough blocks available, requested: {}, available: {}",
                count,
                available_blocks
            );
            return Err(BlockStorageManagerError::NotEnoughBlocksAvailable(
                count,
                available_blocks,
            ));
        }

        let mut blocks = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(block) = self.inactive.acquire_free_block() {
                blocks.push(MutableBlock {
                    block: Some(block),
                    state: state.clone(),
                });
            }
        }

        Ok(blocks)
    }

    pub fn register_block(
        &mut self,
        block: MutableBlock<S, M>,
    ) -> Result<ImmutableBlock<S, M>, BlockStorageManagerError> {
        let (mut block, state) = block.into_parts();

        let mut block = block
            .take()
            .ok_or(BlockStorageManagerError::InvalidMutableBlock(
                "inner block was dropped".to_string(),
            ))?;

        // need to validate that the block is in a complete with a valid sequence hash
        // next, we need to ensure there were no mid-air collisions with the sequence hash, meaning:
        // - the sequence hash is not already in the active map
        // - the sequence hash is not already in the inactive pool

        let sequence_hash = block.sequence_hash().map_err(|e| {
            BlockStorageManagerError::InvalidMutableBlock(format!(
                "block has no sequence hash: {}",
                e.to_string()
            ))
        })?;

        if let Some(immutable) = self.active.match_sequence_hash(sequence_hash) {
            self.return_block(block);
            return Ok(immutable);
        }

        if let Some(mutable) = self.inactive.match_sequence_hash(sequence_hash) {
            self.return_block(block);
            let block = MutableBlock {
                block: Some(mutable),
                state,
            };
            return self.active.register(block);
        }

        // the block is not in the active or inactive pool; now we can register it with the event manager
        // and add it to the active pool

        block
            .register(self.events.as_ref())
            .map_err(|e| BlockStorageManagerError::FailedToRegisterBlock(e.to_string()))?;

        assert!(block.is_registered(), "block is not registered");

        let mutable = MutableBlock {
            block: Some(block),
            state,
        };

        self.active.register(mutable)
    }

    /// Returns a block to the inactive pool
    pub fn return_block(&mut self, mut block: Block<S, M>) {
        self.active.remove(&mut block);
        self.inactive.return_block(block);
    }
}

pub struct MutableBlock<S: Storage, M: BlockMetadata> {
    block: Option<Block<S, M>>,
    state: Arc<RwLock<State<S, M>>>,
}

impl<S: Storage, M: BlockMetadata> MutableBlock<S, M> {
    fn into_parts(mut self) -> (Option<Block<S, M>>, Arc<RwLock<State<S, M>>>) {
        let block = self.block.take();
        let state = Arc::clone(&self.state);
        (block, state)
    }
}

impl<S: Storage, M: BlockMetadata> Drop for MutableBlock<S, M> {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            // TODO: move return_block to the state
            self.state.write().unwrap().inactive.return_block(block);
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

pub struct ImmutableBlock<S: Storage, M: BlockMetadata> {
    block: Arc<MutableBlock<S, M>>,
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

struct ActiveBlockMap<S: Storage, M: BlockMetadata> {
    map: HashMap<SequenceHash, Weak<MutableBlock<S, M>>>,
}

impl<S: Storage, M: BlockMetadata> ActiveBlockMap<S, M> {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        block: MutableBlock<S, M>,
    ) -> Result<ImmutableBlock<S, M>, BlockStorageManagerError> {
        let sequence_hash = block.sequence_hash().map_err(|_| {
            BlockStorageManagerError::InvalidMutableBlock("block has no sequence hash".to_string())
        })?;

        let shared = Arc::new(block);

        match self.map.entry(sequence_hash) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let weak = entry.get();
                if let Some(arc) = weak.upgrade() {
                    Ok(ImmutableBlock { block: arc })
                } else {
                    // Weak reference is no longer alive, update it in the map
                    entry.insert(Arc::downgrade(&shared));
                    Ok(ImmutableBlock { block: shared })
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Arc::downgrade(&shared));
                Ok(ImmutableBlock { block: shared })
            }
        }
    }

    pub fn remove(&mut self, block: &mut Block<S, M>) {
        if let Ok(sequence_hash) = block.sequence_hash() {
            if let Some(weak) = self.map.get(&sequence_hash) {
                if let Some(_arc) = weak.upgrade() {
                    block.reset();
                    return;
                }
            }
            self.map.remove(&sequence_hash);
        }
    }

    pub fn match_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Option<ImmutableBlock<S, M>> {
        if let Some(weak) = self.map.get(&sequence_hash) {
            if let Some(arc) = weak.upgrade() {
                Some(ImmutableBlock { block: arc })
            } else {
                // Weak reference is no longer alive, remove it from the map
                self.map.remove(&sequence_hash);
                None
            }
        } else {
            None
        }
    }
}
