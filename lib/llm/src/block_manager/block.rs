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

pub use crate::tokens::TokenBlockError;
pub use anyhow::Result;
pub use state::{BlockState, BlockStateInvalid};

use crate::tokens::{SaltHash, SequenceHash, Token, TokenBlock, Tokens};

use super::events::PublishHandle;
use super::layout::{BlockLayout, LayoutError};

use std::fmt::Debug;
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

pub trait BlockExt {
    /// Reset the state of the block
    fn reset(&mut self);

    /// Initialize a sequence on the block using a [SaltHash]
    ///
    /// The block must be in the [BlockState::Reset] state.
    ///
    /// After initialization, the block will be in the [BlockState::Partial] state.
    fn init_sequence(&mut self, salt_hash: SaltHash) -> Result<()>;

    /// Appends a single token to the block if it is in the Partial state and not full.
    /// Returns `Err` if the block is not Partial or already full.
    fn add_token(&mut self, token: Token) -> Result<()>;

    /// Appends multiple tokens to the block if it is in the Partial state
    /// and has enough remaining capacity for *all* provided tokens.
    /// The block must be in the [BlockState::Partial] state.
    /// Returns `Err` if the block is not Partial or if there isn't enough space.
    fn add_tokens(&mut self, tokens: Tokens) -> Result<Tokens>;

    /// Removes the last token from the block.
    /// Requires the block to be in the Partial state and not empty.
    /// Returns `Err` otherwise.
    fn pop_token(&mut self) -> Result<()>;

    /// Removes the last `count` tokens from the block.
    /// Requires the block to be in the Partial state and have at least `count` tokens.
    /// Returns `Err` otherwise.
    fn pop_tokens(&mut self, count: usize) -> Result<()>;

    /// Commit the block
    /// Requires the block to be in the [BlockState::Partial] state and completely full.
    /// Transitions the state to [BlockState::Complete]. Returns `Err` otherwise.
    fn commit(&mut self) -> Result<()>;

    /// Apply a [TokenBlock] to the block
    /// Requires the block to be in the [BlockState::Reset] state.
    ///
    /// Additionally, the [TokenBlock] must match the [BlockLayout::page_size()]
    /// Transitions the state to [BlockState::Complete]. Returns `Err` otherwise.
    fn apply_token_block(&mut self, token_block: TokenBlock) -> Result<()>;

    /// Returns the number of tokens currently in the block.
    fn len(&self) -> usize;

    /// Returns the number of additional tokens that can be added (only valid for Partial state).
    fn remaining(&self) -> usize;

    /// Returns true if the block contains no tokens (only true for Reset or empty Partial state).
    fn is_empty(&self) -> bool;

    /// Returns true if the block is full.
    fn is_full(&self) -> bool;

    /// Returns a list of tokens in the block.
    fn tokens(&self) -> Option<&Tokens>;
}

impl<L: BlockLayout, M: BlockMetadata> BlockExt for Block<L, M> {
    fn reset(&mut self) {
        Block::reset(self);
    }

    fn init_sequence(&mut self, salt_hash: SaltHash) -> Result<()> {
        Ok(self
            .state
            .initialize_sequence(self.page_size(), salt_hash)?)
    }

    fn add_token(&mut self, token: Token) -> Result<()> {
        self.state.add_token(token)
    }

    fn add_tokens(&mut self, tokens: Tokens) -> Result<Tokens> {
        self.state.add_tokens(tokens)
    }

    fn pop_token(&mut self) -> Result<()> {
        self.state.pop_token()
    }

    fn pop_tokens(&mut self, count: usize) -> Result<()> {
        self.state.pop_tokens(count)
    }

    fn commit(&mut self) -> Result<()> {
        self.state.commit()
    }

    fn apply_token_block(&mut self, token_block: TokenBlock) -> Result<()> {
        if self.page_size() != token_block.tokens().len() {
            return Err(BlockStateInvalid(format!(
                "TokenBlock size ({}) does not match Block page size ({})",
                token_block.tokens().len(),
                self.page_size()
            ))
            .into());
        }
        self.state.apply_token_block(token_block)
    }

    fn len(&self) -> usize {
        match self.state.len() {
            Some(len) => len,
            None => self.page_size(),
        }
    }

    fn remaining(&self) -> usize {
        self.state.remaining()
    }

    fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    fn is_full(&self) -> bool {
        self.len() == self.page_size()
    }

    fn tokens(&self) -> Option<&Tokens> {
        self.state.tokens()
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
        let offset = self.layout.memory_region_addr(self.block_idx, layer_idx)?;

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
        let offset = self.layout.memory_region_addr(self.block_idx, layer_idx)?;

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
    layout: Box<L>,
    metadata: std::marker::PhantomData<M>,
}

impl<L: BlockLayout, M: BlockMetadata> Blocks<L, M> {
    /// Create a new block storage collection
    pub fn new(layout: L) -> BlockResult<Self> {
        let layout = Box::new(layout);

        Ok(Self {
            layout,
            metadata: std::marker::PhantomData,
        })
    }

    /// Convert collection into Vec<Block> with default metadata/state
    pub fn into_blocks(self) -> BlockResult<Vec<Block<L, M>>> {
        // convert box to arc
        let layout = Arc::new(*self.layout);

        (0..layout.num_blocks())
            .map(|idx| {
                let data = BlockData::new(layout.clone(), idx);
                Block::new(data, M::default())
            })
            .collect()
    }
}

mod nixl {
    use super::*;

    use super::super::{
        layout::nixl::{NixlLayout, ToSerializedNixlBlockLayout},
        storage::nixl::{NixlEnabledStorage, NixlStorage},
    };
    use nixl_sys::{Agent as NixlAgent, OptArgs};

    impl<L: NixlLayout, M: BlockMetadata> Blocks<L, M>
    where
        L::StorageType: NixlEnabledStorage,
    {
        /// Register the blocks with an NIXL agent
        pub fn nixl_register(
            &mut self,
            agent: &NixlAgent,
            opt_args: Option<&OptArgs>,
        ) -> anyhow::Result<()> {
            self.layout.nixl_register(agent, opt_args)
        }
    }
    pub struct RemoteBlocks {
        layout: Arc<dyn BlockLayout<StorageType = NixlStorage>>,
    }

    impl RemoteBlocks {
        pub fn new(layout: Arc<dyn BlockLayout<StorageType = NixlStorage>>) -> Self {
            Self { layout }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tokens::TokenBlockSequence;

    use super::super::layout::tests::setup_layout;
    use super::*;

    const BLOCK_SIZE: usize = 4;
    const SALT_HASH: SaltHash = 12345;

    // Helper to create a block with a specific state for testing
    fn _create_test_block<L: BlockLayout, M: BlockMetadata>(
        layout: Arc<L>,
        block_idx: usize,
        state: BlockState,
        metadata: M,
    ) -> Block<L, M> {
        let data = BlockData::new(layout, block_idx);
        let mut block = Block::new(data, metadata).expect("Failed to create block");
        block.update_state(state); // Set the desired initial state
        block
    }

    // Helper to create a default reset block
    fn create_reset_block() -> Block<impl BlockLayout, BasicMetadata> {
        let layout = setup_layout(None).unwrap();
        let data = BlockData::new(Arc::new(layout), 0);
        Block::new(data, BasicMetadata::default()).unwrap()
    }

    // Helper to create a complete TokenBlock for testing apply_token_block
    fn create_full_token_block() -> TokenBlock {
        let tokens = Tokens::from(vec![1, 2, 3, 4]);
        let salt_hash = SALT_HASH;
        let block_size = BLOCK_SIZE;
        let (mut blocks, _) = TokenBlockSequence::split_tokens(tokens, block_size, salt_hash);
        blocks.pop().unwrap()
    }

    #[test]
    fn test_block_state_transitions_and_ops() {
        let mut block = create_reset_block();
        assert!(matches!(block.state(), BlockState::Reset));

        // --- Reset State --- //
        assert!(block.add_token(1).is_err(), "Append on Reset should fail");
        assert!(
            block.add_tokens(Tokens::from(vec![1])).is_err(),
            "Extend on Reset should fail"
        );
        assert!(block.commit().is_err(), "Commit on Reset should fail");
        assert!(block.pop_token().is_err(), "Pop on Reset should fail");
        assert!(
            block.pop_tokens(1).is_err(),
            "Pop tokens on Reset should fail"
        );

        // --- Reset -> Partial (via init_sequence) --- //
        assert!(block.init_sequence(SALT_HASH).is_ok());
        assert!(matches!(block.state(), BlockState::Partial(_)));

        // --- Partial State --- //
        let invalid_block = create_full_token_block();
        assert!(
            block.apply_token_block(invalid_block).is_err(),
            "Apply block on Partial should fail"
        );

        // Append tokens
        assert!(block.add_token(1).is_ok()); // 1
        assert!(block.add_token(2).is_ok()); // 1, 2
        assert!(block.add_tokens(Tokens::from(vec![3])).is_ok()); // 1, 2, 3
        assert_eq!(block.len(), 3);

        // Extend beyond capacity (should fail)
        let new_tokens = Tokens::from(vec![4, 5]);
        assert_eq!(block.add_tokens(new_tokens.clone()).unwrap().as_ref(), &[5]);

        // Extend to fill capacity
        assert!(block.add_tokens(Tokens::from(vec![4])).is_ok()); // 1, 2, 3, 4
        assert_eq!(block.len(), BLOCK_SIZE);

        // Append when full (should fail)
        assert!(block.add_token(5).is_err(), "Append on full Partial block");

        // Pop tokens
        assert!(block.pop_token().is_ok()); // After pop: 1, 2, 3
        assert_eq!(block.len(), 3);

        // Pop multiple tokens
        assert!(block.pop_tokens(2).is_ok()); // After pop: [1]
        assert_eq!(block.len(), 1);

        // Pop too many tokens (should fail)
        assert!(block.pop_tokens(2).is_err(), "Pop too many tokens");
        assert_eq!(block.len(), 1);

        // Pop last token
        assert!(block.pop_token().is_ok()); // empty
        assert_eq!(block.len(), 0);
        assert!(block.is_empty());

        // Fill block again for commit
        assert!(block.add_tokens(Tokens::from(vec![1, 2, 3, 4])).is_ok());
        assert_eq!(block.len(), BLOCK_SIZE);

        // --- Partial -> Complete (via commit) --- //
        assert!(block.commit().is_ok());
        assert!(matches!(block.state(), BlockState::Complete(_)));
        assert_eq!(block.tokens().unwrap().as_ref(), &[1, 2, 3, 4]);

        // --- Complete State --- //
        assert!(
            block.init_sequence(SALT_HASH).is_err(),
            "Init sequence on Complete should fail"
        );
        assert!(
            block.add_token(5).is_err(),
            "Append on Complete should fail"
        );
        assert!(
            block.add_tokens(Tokens::from(vec![5])).is_err(),
            "Extend on Complete should fail"
        );
        assert!(block.commit().is_err(), "Commit on Complete should fail");
        assert!(block.pop_token().is_err(), "Pop on Complete should fail");
        assert!(
            block.pop_tokens(1).is_err(),
            "Pop tokens on Complete should fail"
        );
        let invalid_block = create_full_token_block();
        assert!(
            block.apply_token_block(invalid_block).is_err(),
            "Apply block on Complete should fail"
        );

        // --- Complete -> Reset (via reset) --- //
        block.reset();
        assert!(matches!(block.state(), BlockState::Reset));

        // --- Reset -> Complete (via apply_token_block) --- //
        let full_block = create_full_token_block();
        assert!(block.apply_token_block(full_block.clone()).is_ok());
        assert!(matches!(block.state(), BlockState::Complete(_)));
        let applied_tokens = block.tokens().unwrap();
        assert_eq!(applied_tokens, full_block.tokens());

        // Testing applying to a non-reset state:
        let mut non_reset_block = create_reset_block();
        non_reset_block.init_sequence(SALT_HASH).unwrap(); // Put in Partial state
        assert!(
            non_reset_block.apply_token_block(full_block).is_err(),
            "Apply block to non-reset state"
        );
    }

    #[test]
    fn test_block_state_incomplete_commit() {
        // Commit incomplete block (should fail)
        let mut partial_block = create_reset_block();
        partial_block.init_sequence(SALT_HASH).unwrap();
        partial_block.add_token(1).unwrap();
        partial_block.add_tokens(Tokens::from(vec![2, 3])).unwrap();
        assert_eq!(partial_block.len(), 3);
        assert!(
            partial_block.commit().is_err(),
            "Commit on incomplete Partial block"
        );
    }

    #[test]
    fn test_error_types() {
        let mut block = create_reset_block();
        block.init_sequence(SALT_HASH).unwrap();

        // Fill the block
        block.add_tokens(Tokens::from(vec![1, 2, 3, 4])).unwrap();

        // Append when full
        let append_err = block.add_token(5).unwrap_err();
        assert!(append_err.is::<TokenBlockError>());
        assert_eq!(
            *append_err.downcast_ref::<TokenBlockError>().unwrap(),
            TokenBlockError::Full
        );

        // .add_tokens will try to fill the block and return the remaining tokens in the Tokens passed in
        let new_tokens = Tokens::from(vec![5]);
        let ret_tokens = block.add_tokens(new_tokens.clone()).unwrap();
        assert_eq!(new_tokens, ret_tokens);

        // Commit when full (should succeed)
        block.commit().unwrap();

        // Commit when Complete
        let commit_err = block.commit().unwrap_err();
        assert!(commit_err.is::<BlockStateInvalid>());

        // Reset and test pop empty
        block.reset();
        block.init_sequence(SALT_HASH).unwrap();
        let pop_err = block.pop_token().unwrap_err();
        assert!(pop_err.is::<TokenBlockError>());
        assert_eq!(
            *pop_err.downcast_ref::<TokenBlockError>().unwrap(),
            TokenBlockError::Empty
        );

        let pop_tokens_err = block.pop_tokens(1).unwrap_err();
        assert!(pop_tokens_err.is::<TokenBlockError>());
        assert_eq!(
            *pop_tokens_err.downcast_ref::<TokenBlockError>().unwrap(),
            TokenBlockError::InsufficientTokens
        );

        // Test commit incomplete
        block.add_token(1).unwrap();
        let commit_incomplete_err = block.commit().unwrap_err();
        assert!(commit_incomplete_err.is::<TokenBlockError>());
        assert_eq!(
            *commit_incomplete_err
                .downcast_ref::<TokenBlockError>()
                .unwrap(),
            TokenBlockError::Incomplete
        );
    }
}
