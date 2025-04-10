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

pub mod state;
pub mod view;

pub use state::BlockState;

use crate::tokens::SequenceHash;

use super::layout::BlockLayout;
use super::storage::{Storage, StorageError};

use std::sync::Arc;
use thiserror::Error;

use dynamo_runtime::utils::pool::Returnable;

/// Result type for Block operations
pub type BlockResult<T> = std::result::Result<T, BlockError>;

/// Errors specific to block storage operations
#[derive(Debug, Error)]
pub enum BlockError {
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Invalid block index: {0}")]
    InvalidBlockIndex(usize),

    #[error("Invalid layer index: {0}")]
    InvalidLayerIndex(usize),

    #[error("Operation failed: {0}")]
    OperationFailed(String),

    #[error("Lock error: {0}")]
    LockError(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Unregistered")]
    Unregistered,
}

pub trait BlockMetadata: Default + std::fmt::Debug + Clone + Ord + Send + Sync + 'static {
    /// Called when the block is acquired from the pool
    fn on_acquired(&mut self);

    /// Called when the block is returned to the pool
    fn on_returned(&mut self, tick: u64);

    /// Resets the metadata to the default value
    /// If called, the [BlockMetadata::is_reset()] should return true
    fn reset_metadata(&mut self);

    // NOTE: we might not need this
    /// Returns true if the metadata is reset
    fn is_reset(&self) -> bool;
}

/// A block with storage and associated metadata/state
#[derive(Debug)]
pub struct Block<S: Storage, M: BlockMetadata> {
    storage: BlockData<S>,
    metadata: M,
    state: BlockState,
}

impl<S: Storage, M: BlockMetadata> Block<S, M> {
    /// Create a new block with default metadata/state
    pub fn new(storage: BlockData<S>, metadata: M) -> BlockResult<Self> {
        Ok(Self {
            storage,
            metadata,
            state: BlockState::Reset,
        })
    }

    // /// Create a new block with custom metadata/state
    // pub fn with_metadata_state<M, St>(
    //     storage: BlockData<S>,
    //     _metadata: M,
    //     _state: St,
    // ) -> BlockResult<Self> {
    //     // Would store metadata and state
    //     Ok(Self {
    //         storage,
    //         sequence_hash: 0,
    //         block_hash: 0,
    //     })
    // }

    pub fn sequence_hash(&self) -> Result<SequenceHash, BlockError> {
        match self.state() {
            BlockState::Complete(state) => Ok(state.token_block.sequence_hash()),
            BlockState::Registered(state) => Ok(state.sequence_hash),
            _ => Err(BlockError::InvalidState(
                "Block is not complete".to_string(),
            )),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.state = BlockState::Reset;
        self.metadata.reset_metadata();
    }

    /// Get the metadata of the block
    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    /// Get a mutable reference to the metadata of the block
    pub fn metadata_mut(&mut self) -> &mut M {
        &mut self.metadata
    }

    /// Get a reference to the state of the block
    pub fn state(&self) -> &BlockState {
        &self.state
    }

    /// Get a mutable reference to the state of the block
    pub(crate) fn state_mut(&mut self) -> &mut BlockState {
        &mut self.state
    }

    /// Returns true if the block is empty
    pub fn is_empty(&self) -> bool {
        matches!(self.state, BlockState::Reset)
    }

    /// Returns true if the block is in the complete or registered state
    pub fn is_complete(&self) -> bool {
        matches!(
            self.state,
            BlockState::Complete(_) | BlockState::Registered(_)
        )
    }

    /// Returns true if the block is in the registered state
    pub fn is_registered(&self) -> bool {
        matches!(self.state, BlockState::Registered(_))
    }

    /// Get a read-only view of a layer
    pub fn layer_view(&self, layer_idx: usize) -> BlockResult<view::BlockView<S>> {
        self.storage.layer_view(layer_idx)
    }

    /// Get a mutable view of a layer
    pub fn layer_view_mut(&mut self, layer_idx: usize) -> BlockResult<view::BlockViewMut<S>> {
        self.storage.layer_view_mut(layer_idx)
    }

    /// Get the number of blocks in the block
    pub fn num_blocks(&self) -> usize {
        self.storage.layout.num_blocks()
    }

    /// Get the number of layers in the block
    pub fn num_layers(&self) -> usize {
        self.storage.layout.num_layers()
    }

    /// Get the size of each block in the block
    pub fn page_size(&self) -> usize {
        self.storage.layout.page_size()
    }

    /// Get the inner dimension of the block
    pub fn inner_dim(&self) -> usize {
        self.storage.layout.inner_dim()
    }
}

impl<S: Storage, M: BlockMetadata> Returnable for Block<S, M> {
    fn on_return(&mut self) {}
}

/// Individual block storage - cannot be cloned to ensure uniqueness
#[derive(Debug)]
pub struct BlockData<S: Storage> {
    layout: Arc<dyn BlockLayout>,
    block_idx: usize,
    storage: std::marker::PhantomData<S>,
}

impl<S: Storage> BlockData<S> {
    /// Create a new block storage
    fn new(layout: Arc<dyn BlockLayout>, block_idx: usize) -> Self {
        Self {
            layout,
            block_idx,
            storage: std::marker::PhantomData,
        }
    }

    /// Get a read-only view of this block's storage for a layer
    pub fn layer_view(&self, layer_idx: usize) -> BlockResult<view::BlockView<S>> {
        let offset = self
            .layout
            .get_memory_region(self.block_idx, layer_idx)
            .map_err(|e| {
                BlockError::OperationFailed(format!("Failed to get layer region: {}", e))
            })?;

        unsafe { view::BlockView::new(self, offset as usize, self.layout.memory_region_size()) }
    }

    /// Get a mutable view of this block's storage for a layer
    ///
    /// # Safety
    /// The caller must ensure:
    /// - No other views of this block are concurrently accessed
    /// - This is enforced in Rust by BlockView requiring unique access
    /// - Cannot be enforced when using with Python bindings or CUDA kernels
    pub fn layer_view_mut(&mut self, layer_idx: usize) -> BlockResult<view::BlockViewMut<S>> {
        let offset = self
            .layout
            .get_memory_region(self.block_idx, layer_idx)
            .map_err(|e| {
                BlockError::OperationFailed(format!("Failed to get layer region: {}", e))
            })?;

        unsafe { view::BlockViewMut::new(self, offset as usize, self.layout.memory_region_size()) }
    }
}

/// Collection that holds shared storage and layout
#[derive(Debug)]
pub struct BlockStorageCollection<S: Storage, M: BlockMetadata> {
    layout: Arc<dyn BlockLayout>,
    storage: std::marker::PhantomData<S>,
    metadata: std::marker::PhantomData<M>,
}

impl<S: Storage, M: BlockMetadata> BlockStorageCollection<S, M> {
    /// Create a new block storage collection
    pub fn new(layout: impl BlockLayout + 'static) -> BlockResult<Self> {
        let layout = Arc::new(layout);

        Ok(Self {
            layout,
            storage: std::marker::PhantomData,
            metadata: std::marker::PhantomData,
        })
    }

    /// Convert collection into Vec<Block> with default metadata/state
    pub fn into_blocks(self) -> BlockResult<Vec<Block<S, M>>> {
        (0..self.layout.num_blocks())
            .map(|idx| {
                let storage = BlockData::new(self.layout.clone(), idx);
                Block::new(storage, M::default())
            })
            .collect()
    }

    // /// Convert collection into Vec<Block> with custom metadata/state
    // pub fn into_blocks_with<F, M, St>(self, f: F) -> BlockResult<Vec<Block<S>>>
    // where
    //     F: Fn(usize) -> (M, St),
    // {
    //     (0..self.layout.num_blocks())
    //         .map(|idx| {
    //             let storage = BlockData::new(self.storage.clone(), self.layout.clone(), idx);
    //             let (metadata, state) = f(idx);
    //             Block::with_metadata_state(storage, metadata, state)
    //         })
    //         .collect()
    // }
}
