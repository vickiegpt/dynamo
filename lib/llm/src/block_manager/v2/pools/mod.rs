// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block pool implementations for managing active and inactive blocks.
//!
//! This module contains the core pool structures that track blocks in different states:
//! - `ActiveBlockPool`: Tracks registered blocks via weak references
//! - `InactiveBlockPool`: Manages available blocks via a free list

pub mod block;
pub mod registry;

// pub mod inactive;
pub mod registered;
pub mod reset;
pub mod reuse_policy;

//#h[cfg(test)]
//mod test_raii;

use std::{ops::Deref, sync::Arc};

pub use crate::tokens::{SequenceHash, TokenBlock};

use block::{Block, BlockId, Complete, Registered, Reset};
use registry::BlockRegistrationHandle;

//pub use inactive::InactiveBlockPool;
pub use registered::RegisteredPool;

pub trait BlockMetadata: Clone + Send + Sync + 'static {}
impl<T: Clone + Send + Sync + 'static> BlockMetadata for T {}

pub trait BlockAllocator<T: BlockMetadata> {
    // fn new(blocks: Vec<Block<T, Reset>>) -> Arc<Self>
    // where
    //     Self: Sized;

    /// Insert a block into the pool
    fn insert(&mut self, block: Block<T, Reset>);

    /// Acquire the first block to be reused
    fn pop(&mut self) -> Option<Block<T, Reset>>;

    /// Get the number of available blocks
    fn len(&self) -> usize;
}

pub trait BlockMatcher<T: BlockMetadata> {
    fn find_match(&self, seq_hash: SequenceHash) -> Option<ImmutableBlock<T>>;
}

/// Policy for handling duplicate blocks with the same sequence hash
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDuplicationPolicy {
    /// Allow duplicate blocks - each gets its own DuplicateBlock wrapper
    Allow,
    /// Reject duplicates - return the existing primary block instead
    Reject,
}

// Re-export the new RAII guard types - no need to re-export here since they're defined in this module

/// A block that is free and available for allocation
/// This block must be in a Registered state and have a valid sequence hash
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InactiveBlock {
    pub block_id: BlockId,
    pub seq_hash: SequenceHash,
}

/// RAII guard for [`Block<T, Reset>`] that automatically returns to ResetPool on drop
pub struct MutableBlock<T: BlockMetadata> {
    block: Option<Block<T, Reset>>,
    return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
}

/// RAII guard for [`Block<T, Complete>`] that automatically returns to ResetPool on drop
pub struct CompleteBlock<T: BlockMetadata> {
    block: Option<Block<T, Complete>>,
    return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
}

pub trait RegisteredBlock<T>: Send + Sync {
    /// Get the block ID
    fn block_id(&self) -> BlockId;

    /// Get the sequence hash
    fn sequence_hash(&self) -> SequenceHash;

    /// Get the registration handle
    fn registration_handle(&self) -> &BlockRegistrationHandle;
}

/// RAII guard for [`Block<T, Registered>`] that automatically returns to RegisteredPool on drop
pub(crate) struct PrimaryBlock<T: BlockMetadata> {
    block: Option<Arc<Block<T, Registered>>>,
    return_fn: Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>,
}

struct DuplicateBlock<T: BlockMetadata> {
    block: Option<Block<T, Registered>>,
    return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
    _primary: Arc<PrimaryBlock<T>>,
}

pub struct ImmutableBlock<T: BlockMetadata> {
    block: Arc<dyn RegisteredBlock<T>>,
}

// RegisteredPool implementation moved to registered.rs

impl<T: BlockMetadata> MutableBlock<T> {
    /// Create a new MutableBlock in Reset state
    fn new(block: Block<T, Reset>, return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>) -> Self {
        Self {
            block: Some(block),
            return_fn,
        }
    }

    /// Get the block ID
    pub fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    /// Transition from Reset to Complete state
    pub fn complete(mut self, token_block: TokenBlock) -> CompleteBlock<T> {
        let block = self.block.take().unwrap().complete(token_block);

        CompleteBlock {
            block: Some(block),
            return_fn: self.return_fn.clone(),
        }
    }
}

impl<T: BlockMetadata> CompleteBlock<T> {
    /// Get the block ID
    pub fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    /// Access token block if in Complete state
    pub fn token_block(&self) -> &TokenBlock {
        self.block.as_ref().unwrap().token_block()
    }

    /// Get sequence hash if in Complete state
    pub fn sequence_hash(&self) -> SequenceHash {
        self.block.as_ref().unwrap().sequence_hash()
    }

    pub fn reset(mut self) -> MutableBlock<T> {
        let block = self.block.take().unwrap().reset();

        MutableBlock {
            block: Some(block),
            return_fn: self.return_fn.clone(),
        }
    }
}

impl<T: BlockMetadata> PrimaryBlock<T> {
    /// Create a new RegisteredBlock
    fn new(
        block: Arc<Block<T, Registered>>,
        return_fn: Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>,
    ) -> Self {
        Self {
            block: Some(block),
            return_fn,
        }
    }

    fn register(self) -> ImmutableBlock<T> {
        let block = self.block.clone().unwrap();
        block.registration_handle().attach_block(self)
    }
}

impl<T: BlockMetadata> RegisteredBlock<T> for PrimaryBlock<T> {
    fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    fn sequence_hash(&self) -> SequenceHash {
        self.block.as_ref().unwrap().sequence_hash()
    }

    fn registration_handle(&self) -> &BlockRegistrationHandle {
        self.block.as_ref().unwrap().registration_handle()
    }
}

impl<T: BlockMetadata> DuplicateBlock<T> {
    /// Create a new DuplicateBlock
    fn new(
        block: Block<T, Registered>,
        primary: Arc<PrimaryBlock<T>>,
        return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
    ) -> Self {
        Self {
            block: Some(block),
            return_fn,
            _primary: primary,
        }
    }
}

impl<T: BlockMetadata> RegisteredBlock<T> for DuplicateBlock<T> {
    fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    fn sequence_hash(&self) -> SequenceHash {
        self.block.as_ref().unwrap().sequence_hash()
    }

    fn registration_handle(&self) -> &BlockRegistrationHandle {
        self.block.as_ref().unwrap().registration_handle()
    }
}

impl<T: BlockMetadata> Deref for ImmutableBlock<T> {
    type Target = dyn RegisteredBlock<T>;

    fn deref(&self) -> &Self::Target {
        self.block.as_ref()
    }
}

impl<T: BlockMetadata> Drop for MutableBlock<T> {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block);
        }
    }
}

impl<T: BlockMetadata> Drop for CompleteBlock<T> {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block.reset());
        }
    }
}

impl<T: BlockMetadata> Drop for PrimaryBlock<T> {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block);
        }
    }
}

impl<T: BlockMetadata> Drop for DuplicateBlock<T> {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block.reset());
        }
    }
}
