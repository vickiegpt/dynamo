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

pub mod registry;
pub mod state;
pub mod view;

pub use anyhow::Result;
pub use state::BlockState;

use crate::tokens::{SaltHash, SequenceHash, Token, TokenBlock};

use super::events::PublishHandle;
use super::layout::{BlockLayout, LayoutError};

use std::sync::Arc;
use thiserror::Error;

/// Result type for Block operations
pub type BlockResult<T> = std::result::Result<T, BlockError>;

/// Errors specific to block storage operations
#[derive(Debug, Error)]
pub enum BlockError {
    #[error(transparent)]
    Layout(#[from] LayoutError),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

pub trait BlockMetadata: Default + std::fmt::Debug + Clone + Ord + Send + Sync + 'static {
    /// Called when the block is acquired from the pool
    fn on_acquired(&mut self, tick: u64);

    /// Called when the block is returned to the pool
    fn on_returned(&mut self, tick: u64);

    /// Resets the metadata to the default value
    /// If called, the [BlockMetadata::is_reset()] should return true
    fn reset_metadata(&mut self);
}

/// A block with storage and associated metadata/state
#[derive(Debug)]
pub struct Block<L: BlockLayout, M: BlockMetadata> {
    data: BlockData<L>,
    metadata: M,
    state: BlockState,
}

impl<L: BlockLayout, M: BlockMetadata> Block<L, M> {
    /// Create a new block with default metadata/state
    pub fn new(data: BlockData<L>, metadata: M) -> BlockResult<Self> {
        Ok(Self {
            data,
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
            BlockState::Complete(state) => Ok(state.token_block().sequence_hash()),
            BlockState::Registered(state) => Ok(state.sequence_hash()),
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

    /// Update the metadata of the block
    pub fn update_metadata(&mut self, metadata: M) {
        self.metadata = metadata;
    }

    /// Update the state of the block
    #[allow(dead_code)]
    pub(crate) fn update_state(&mut self, state: BlockState) {
        self.state = state;
    }

    /// Get a reference to the state of the block
    pub fn state(&self) -> &BlockState {
        &self.state
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
        matches!(&self.state, BlockState::Registered(_state))
    }

    /// Get a read-only view of a layer
    pub fn layer_view(&self, layer_idx: usize) -> BlockResult<view::BlockView<L>> {
        self.data.layer_view(layer_idx)
    }

    /// Get a mutable view of a layer
    pub fn layer_view_mut(&mut self, layer_idx: usize) -> BlockResult<view::BlockViewMut<L>> {
        self.data.layer_view_mut(layer_idx)
    }

    /// Get the number of blocks in the block
    pub fn num_blocks(&self) -> usize {
        self.data.layout.num_blocks()
    }

    /// Get the number of layers in the block
    pub fn num_layers(&self) -> usize {
        self.data.layout.num_layers()
    }

    /// Get the size of each block in the block
    pub fn page_size(&self) -> usize {
        self.data.layout.page_size()
    }

    /// Get the inner dimension of the block
    pub fn inner_dim(&self) -> usize {
        self.data.layout.inner_dim()
    }

    pub(crate) fn metadata_on_acquired(&mut self, tick: u64) {
        self.metadata.on_acquired(tick);
    }

    pub(crate) fn metadata_on_returned(&mut self, tick: u64) {
        self.metadata.on_returned(tick);
    }
}

pub(crate) trait PrivateBlockExt {
    fn register(
        &mut self,
        registry: &mut registry::BlockRegistry,
    ) -> Result<PublishHandle, registry::BlockRegistationError>;
}

impl<L: BlockLayout, M: BlockMetadata> PrivateBlockExt for Block<L, M> {
    fn register(
        &mut self,
        registry: &mut registry::BlockRegistry,
    ) -> Result<PublishHandle, registry::BlockRegistationError> {
        registry.register_block(&mut self.state)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BlockTokenError {}

pub trait BlockExt {
    /// Reset the state of the block
    fn reset(&mut self);

    /// Initialize a sequence on the block
    ///
    /// The block must be in the [BlockState::Reset] state.
    ///
    /// After initialization, the block will be in the [BlockState::Partial] state.
    fn initialize_sequence(&mut self, salt_hash: SaltHash) -> Result<()>;

    /// Add a token to the block
    /// If Ok, returns the number of remaining tokens in the block
    fn add_token(&mut self, token: Token) -> Result<()>;

    /// Apply a [TokenBlock] to the block
    /// The block must be [BlockState::Reset].
    ///
    /// Additionally, the [TokenBlock] must match the [BlockLayout::page_size()]
    fn apply_token_block(&mut self, token_block: &TokenBlock) -> Result<()>;
}

impl<L: BlockLayout, M: BlockMetadata> BlockExt for Block<L, M> {
    fn reset(&mut self) {
        self.reset();
    }

    fn initialize_sequence(&mut self, salt_hash: SaltHash) -> Result<()> {
        Ok(self
            .state
            .initialize_sequence(self.page_size(), salt_hash)?)
    }

    fn add_token(&mut self, token: Token) -> Result<()> {
        Ok(self.state.add_token(&token)?)
    }

    fn apply_token_block(&mut self, token_block: &TokenBlock) -> Result<()> {
        unimplemented!()
    }
}

/// Individual block storage - cannot be cloned to ensure uniqueness
#[derive(Debug)]
pub struct BlockData<L: BlockLayout> {
    layout: Arc<L>,
    block_idx: usize,
}

impl<L: BlockLayout> BlockData<L> {
    /// Create a new block storage
    fn new(layout: Arc<L>, block_idx: usize) -> Self {
        Self { layout, block_idx }
    }

    /// Get a read-only view of this block's storage for a layer
    pub fn layer_view(&self, layer_idx: usize) -> BlockResult<view::BlockView<L>> {
        let offset = self.layout.get_memory_region(self.block_idx, layer_idx)?;

        unsafe { view::BlockView::new(self, offset as usize, self.layout.memory_region_size()) }
    }

    /// Get a mutable view of this block's storage for a layer
    ///
    /// # Safety
    /// The caller must ensure:
    /// - No other views of this block are concurrently accessed
    /// - This is enforced in Rust by BlockView requiring unique access
    /// - Cannot be enforced when using with Python bindings or CUDA kernels
    pub fn layer_view_mut(&mut self, layer_idx: usize) -> BlockResult<view::BlockViewMut<L>> {
        let offset = self.layout.get_memory_region(self.block_idx, layer_idx)?;

        unsafe { view::BlockViewMut::new(self, offset as usize, self.layout.memory_region_size()) }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct BasicMetadata {
    priority: u32,
    returned_tick: u64,
    acquired_tick: u64,
}

impl BlockMetadata for BasicMetadata {
    fn on_acquired(&mut self, tick: u64) {
        self.acquired_tick = tick;
    }

    fn on_returned(&mut self, tick: u64) {
        self.returned_tick = tick;
    }

    fn reset_metadata(&mut self) {
        self.priority = 0;
    }
}
/// Collection that holds shared storage and layout
#[derive(Debug)]
pub struct Blocks<L: BlockLayout, M: BlockMetadata> {
    layout: Arc<L>,
    metadata: std::marker::PhantomData<M>,
}

impl<L: BlockLayout, M: BlockMetadata> Blocks<L, M> {
    /// Create a new block storage collection
    pub fn new(layout: L) -> BlockResult<Self> {
        let layout = Arc::new(layout);

        Ok(Self {
            layout,
            metadata: std::marker::PhantomData,
        })
    }

    /// Convert collection into Vec<Block> with default metadata/state
    pub fn into_blocks(self) -> BlockResult<Vec<Block<L, M>>> {
        (0..self.layout.num_blocks())
            .map(|idx| {
                let data = BlockData::new(self.layout.clone(), idx);
                Block::new(data, M::default())
            })
            .collect()
    }
}
