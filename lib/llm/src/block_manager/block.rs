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

use crate::block_manager::storage::Storage;
use crate::tokens::{SaltHash, SequenceHash, Token, TokenBlock, Tokens};

use super::events::PublishHandle;
use super::layout::{BlockLayout, LayoutError, LayoutType};

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
pub struct Block<S: Storage, M: BlockMetadata> {
    data: BlockData<S>,
    metadata: M,
    state: BlockState,
}

impl<S: Storage, M: BlockMetadata> Block<S, M> {
    /// Create a new block with default metadata/state
    pub fn new(data: BlockData<S>, metadata: M) -> BlockResult<Self> {
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

impl<S: Storage, M: BlockMetadata> PrivateBlockExt for Block<S, M> {
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

impl<S: Storage, M: BlockMetadata> BlockExt for Block<S, M> {
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

pub trait BlockDataExt<S: Storage> {
    /// Returns true if the block data is fully contiguous
    fn is_fully_contiguous(&self) -> bool;

    /// Get a read-only view of this block's storage for a layer
    fn layer_view(&self, layer_idx: usize) -> BlockResult<view::LayerView<S>>;

    /// Get a mutable view of this block's storage for a layer
    fn layer_view_mut(&mut self, layer_idx: usize) -> BlockResult<view::LayerViewMut<S>>;

    /// Get a read-only view of this block's storage
    fn block_view(&self) -> BlockResult<view::BlockView<S>>;

    /// Get a mutable view of this block's storage
    fn block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<S>>;
}

/// Individual block storage - cannot be cloned to ensure uniqueness
#[derive(Debug)]
pub struct BlockData<S: Storage> {
    layout: Arc<dyn BlockLayout<StorageType = S>>,
    block_idx: usize,
}

impl<S> BlockData<S>
where
    S: Storage,
{
    /// Create a new block storage
    fn new(layout: Arc<dyn BlockLayout<StorageType = S>>, block_idx: usize) -> Self {
        Self { layout, block_idx }
    }
}

impl<S: Storage> BlockDataExt<S> for BlockData<S> {
    fn is_fully_contiguous(&self) -> bool {
        self.layout.layout_type() == LayoutType::FullyContiguous
    }

    fn layer_view(&self, layer_idx: usize) -> BlockResult<view::LayerView<S>> {
        let offset = self.layout.memory_region_addr(self.block_idx, layer_idx)?;
        unsafe { view::LayerView::new(self, offset as usize, self.layout.memory_region_size()) }
    }

    fn layer_view_mut(&mut self, layer_idx: usize) -> BlockResult<view::LayerViewMut<S>> {
        let offset = self.layout.memory_region_addr(self.block_idx, layer_idx)?;
        unsafe { view::LayerViewMut::new(self, offset as usize, self.layout.memory_region_size()) }
    }

    fn block_view(&self) -> BlockResult<view::BlockView<S>> {
        if self.is_fully_contiguous() {
            let offset = self.layout.memory_region_addr(self.block_idx, 0)?;
            let size = self.layout.memory_region_size() * self.layout.num_layers();
            unsafe { view::BlockView::new(self, offset as usize, size) }
        } else {
            Err(BlockError::InvalidState(
                "Block is not fully contiguous".to_string(),
            ))
        }
    }

    fn block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<S>> {
        if self.is_fully_contiguous() {
            let offset = self.layout.memory_region_addr(self.block_idx, 0)?;
            let size = self.layout.memory_region_size() * self.layout.num_layers();
            unsafe { view::BlockViewMut::new(self, offset as usize, size) }
        } else {
            Err(BlockError::InvalidState(
                "Block is not fully contiguous".to_string(),
            ))
        }
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

impl<L: BlockLayout + 'static, M: BlockMetadata> Blocks<L, M> {
    /// Create a new block storage collection
    pub fn new(layout: L) -> BlockResult<Self> {
        let layout = Box::new(layout);

        Ok(Self {
            layout,
            metadata: std::marker::PhantomData,
        })
    }

    /// Convert collection into Vec<Block> with default metadata/state
    pub fn into_blocks(self) -> BlockResult<Vec<Block<L::StorageType, M>>> {
        // convert box to arc
        let layout: Arc<dyn BlockLayout<StorageType = L::StorageType>> = Arc::new(*self.layout);

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

    use super::view::{BlockKind, Kind, LayerKind};

    use super::super::{
        layout::nixl::NixlLayout,
        storage::nixl::{MemType, NixlEnabledStorage, NixlStorage},
    };
    use nixl_sys::{Agent as NixlAgent, MemoryRegion, NixlDescriptor, OptArgs};

    // --- Mutability Marker ---
    pub trait MutabilityKind: Debug + Clone + Copy + Send + Sync + 'static {}

    #[derive(Debug, Clone, Copy)]
    pub struct IsMutable;
    impl MutabilityKind for IsMutable {}

    #[derive(Debug, Clone, Copy)]
    pub struct IsImmutable;
    impl MutabilityKind for IsImmutable {}

    // pub struct NixlBlockData<S: Storage + NixlEnabledStorage> {
    //     layout: Arc<dyn NixlLayout<StorageType = S>>,
    //     block_idx: usize,
    //     mem_type: MemType,
    //     device_id: u64,
    // }

    // impl<S: Storage + NixlEnabledStorage> NixlBlockData<S> {
    //     pub fn new(layout: Arc<dyn NixlLayout<StorageType = S>>, block_idx: usize) -> Self {
    //         let mem_type = layout.mem_type();
    //         let device_id = layout.device_id();
    //         Self {
    //             layout,
    //             block_idx,
    //             mem_type,
    //             device_id,
    //         }
    //     }
    // }

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

    /// A unified, lifetime-bound descriptor containing information needed for NIXL operations.
    /// Typed by Kind (Block/Layer) and Mutability (IsMutable/IsImmutable).
    #[derive(Copy, Clone)] // Can be Copy/Clone as it holds basic data + markers
    pub struct NixlMemoryDescriptor<'a, K: Kind, M: MutabilityKind> {
        addr: u64,
        size: usize,
        mem_type: MemType,
        device_id: u64,
        _lifetime: std::marker::PhantomData<&'a ()>, // Binds the descriptor's lifetime to 'a
        _kind: std::marker::PhantomData<K>,          // Stores the Kind marker type
        _mutability: std::marker::PhantomData<M>,    // Stores the Mutability marker type
    }

    // Helper function to get the short type name
    fn short_type_name<T>() -> &'static str {
        let name = core::any::type_name::<T>();
        name.split("::").last().unwrap_or(name)
    }

    // Implement Debug manually to avoid bounds on K/M
    impl<'a, K: Kind, M: MutabilityKind> Debug for NixlMemoryDescriptor<'a, K, M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("NixlMemoryDescriptor")
                .field("addr", &self.addr)
                .field("size", &self.size)
                .field("mem_type", &self.mem_type)
                .field("device_id", &self.device_id)
                .field("kind", &short_type_name::<K>()) // Show marker types
                .field("mutability", &short_type_name::<M>())
                .finish()
        }
    }

    impl<'a, K: Kind, M: MutabilityKind> NixlMemoryDescriptor<'a, K, M> {
        /// Creates a new NixlMemoryDescriptor. Typically called via conversion methods.
        #[inline]
        pub(crate) fn new(addr: u64, size: usize, mem_type: MemType, device_id: u64) -> Self {
            Self {
                addr,
                size,
                mem_type,
                device_id,
                _lifetime: std::marker::PhantomData,
                _kind: std::marker::PhantomData,
                _mutability: std::marker::PhantomData,
            }
        }
    }

    impl<'a, K: Kind, M: MutabilityKind> MemoryRegion for NixlMemoryDescriptor<'a, K, M> {
        unsafe fn as_ptr(&self) -> *const u8 {
            self.addr as *const u8
        }

        fn size(&self) -> usize {
            self.size
        }
    }

    impl<'a, K: Kind, M: MutabilityKind> NixlDescriptor for NixlMemoryDescriptor<'a, K, M> {
        fn mem_type(&self) -> MemType {
            self.mem_type
        }

        fn device_id(&self) -> u64 {
            self.device_id
        }
    }

    pub trait NixlBlockDataExt<S: Storage>: BlockDataExt<S> {
        /// Get the NIXL memory descriptor for the entire block
        fn as_block_descriptor(
            &self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsImmutable>>;

        /// Get the NIXL memory descriptor for the entire block
        fn as_block_descriptor_mut(
            &mut self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsMutable>>;

        /// Get the NIXL memory descriptor for a specific layer
        fn as_layer_descriptor(
            &self,
            layer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsImmutable>>;

        /// Get the NIXL memory descriptor for a specific layer
        fn as_layer_descriptor_mut(
            &mut self,
            layer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsMutable>>;
    }

    impl<S: Storage + NixlDescriptor> NixlBlockDataExt<S> for BlockData<S> {
        fn as_block_descriptor(
            &self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsImmutable>> {
            Ok(self.block_view()?.as_nixl_descriptor())
        }

        fn as_block_descriptor_mut(
            &mut self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsMutable>> {
            Ok(self.block_view_mut()?.as_nixl_descriptor_mut())
        }

        fn as_layer_descriptor(
            &self,
            layer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsImmutable>> {
            Ok(self.layer_view(layer_idx)?.as_nixl_descriptor())
        }

        fn as_layer_descriptor_mut(
            &mut self,
            layer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsMutable>> {
            Ok(self.layer_view_mut(layer_idx)?.as_nixl_descriptor_mut())
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

    pub struct RemoteBlock {
        data: BlockData<NixlStorage>,
        block_idx: usize,
    }

    impl RemoteBlock {
        pub fn new(
            layout: Arc<dyn BlockLayout<StorageType = NixlStorage>>,
            block_idx: usize,
        ) -> Self {
            let data = BlockData::new(layout, block_idx);
            Self { data, block_idx }
        }
    }

    impl BlockDataExt<NixlStorage> for RemoteBlock {
        fn is_fully_contiguous(&self) -> bool {
            self.data.is_fully_contiguous()
        }

        fn layer_view(&self, layer_idx: usize) -> BlockResult<view::LayerView<NixlStorage>> {
            self.data.layer_view(layer_idx)
        }

        fn layer_view_mut(
            &mut self,
            layer_idx: usize,
        ) -> BlockResult<view::LayerViewMut<NixlStorage>> {
            self.data.layer_view_mut(layer_idx)
        }

        fn block_view(&self) -> BlockResult<view::BlockView<NixlStorage>> {
            self.data.block_view()
        }

        fn block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<NixlStorage>> {
            self.data.block_view_mut()
        }
    }

    impl NixlBlockDataExt<NixlStorage> for RemoteBlock {
        fn as_block_descriptor(
            &self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsImmutable>> {
            self.data.as_block_descriptor()
        }

        fn as_block_descriptor_mut(
            &mut self,
        ) -> BlockResult<NixlMemoryDescriptor<'_, BlockKind, IsMutable>> {
            self.data.as_block_descriptor_mut()
        }

        fn as_layer_descriptor(
            &self,
            layer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsImmutable>> {
            self.data.as_layer_descriptor(layer_idx)
        }

        fn as_layer_descriptor_mut(
            &mut self,
            layer_idx: usize,
        ) -> BlockResult<NixlMemoryDescriptor<'_, LayerKind, IsMutable>> {
            self.data.as_layer_descriptor_mut(layer_idx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::nixl::*;

    use super::super::layout::{
        nixl::{NixlLayout, SerializedNixlBlockLayout, ToSerializedNixlBlockLayout},
        tests::setup_layout,
        FullyContiguous, LayoutConfig,
    };
    use crate::block_manager::storage::SystemAllocator;
    use crate::tokens::TokenBlockSequence;

    use dynamo_runtime::logging::init as init_logging;
    use nixl_sys::Agent as NixlAgent;

    const BLOCK_SIZE: usize = 4;
    const SALT_HASH: SaltHash = 12345;

    // Helper to create a default reset block
    fn create_reset_block() -> Block<impl Storage, BasicMetadata> {
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

    #[test]
    fn test_nixl_block_data_ext() {
        init_logging();

        let config = LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(2)
            .page_size(4)
            .inner_dim(13)
            .build()
            .unwrap();

        let mut layout = FullyContiguous::allocate(config, &SystemAllocator::default()).unwrap();
        let agent = NixlAgent::new("test").unwrap();

        tracing::info!("Registering layout");
        layout.nixl_register(&agent, None).unwrap();
        tracing::info!("Layout registered");

        let serialized = layout.serialize().unwrap();
        let layout = Arc::new(layout);

        let data = BlockData::new(layout.clone(), 0);
        let block_desc = data.as_block_descriptor().unwrap();
        println!("Block descriptor: {:?}", block_desc);

        let data = BlockData::new(layout.clone(), 1);
        let block_desc = data.as_block_descriptor().unwrap();
        println!("Block descriptor: {:?}", block_desc);

        let remote_layout = SerializedNixlBlockLayout::deserialize(&serialized).unwrap();
        println!("Nixl layout: {:?}", remote_layout);

        let remote_block = RemoteBlock::new(remote_layout.clone(), 0);
        let remote_desc = remote_block.as_block_descriptor().unwrap();
        println!("Remote Descriptor: {:?}", remote_desc);

        // drop(layout);
        tracing::info!("Layout dropped");
    }
}
