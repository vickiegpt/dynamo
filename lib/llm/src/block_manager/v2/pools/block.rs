// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Type-state pattern for block lifecycle with compile-time state enforcement.
//!
//! Blocks transition through states: Reset → Complete → Registered → Reset.
//! The type system prevents invalid state transitions at compile time.

use super::registry::BlockRegistrationHandle;
use crate::tokens::{SequenceHash, TokenBlock};
use std::marker::PhantomData;

/// Block identifier type
pub type BlockId = u64;

// Generic Block with marker and state markers
#[derive(Debug)]
pub struct Block<T, State> {
    block_id: BlockId,
    state: State,
    marker: PhantomData<T>,
}

// State marker types
#[derive(Debug)]
pub struct Reset;

// State-specific data holders
#[derive(Debug)]
pub struct Complete {
    token_block: TokenBlock,
}

#[derive(Debug)]
pub struct Registered {
    sequence_hash: SequenceHash,
    registration_handle: BlockRegistrationHandle,
}

// Implementation for Reset state
impl<T> Block<T, Reset> {
    pub fn new(block_id: BlockId) -> Self {
        Self {
            block_id,
            state: Reset,
            marker: PhantomData,
        }
    }

    pub fn complete(self, token_block: TokenBlock) -> Block<T, Complete> {
        Block {
            block_id: self.block_id,
            state: Complete { token_block },
            marker: PhantomData,
        }
    }

    pub fn reset(self) -> Block<T, Reset> {
        self // Already in reset state
    }
}

// Implementation for Complete state
impl<T> Block<T, Complete> {
    pub fn register(self, registration_handle: BlockRegistrationHandle) -> Block<T, Registered> {
        Block {
            block_id: self.block_id,
            state: Registered {
                sequence_hash: self.state.token_block.sequence_hash(),
                registration_handle,
            },
            marker: PhantomData,
        }
    }

    pub fn token_block(&self) -> &TokenBlock {
        &self.state.token_block
    }

    pub fn sequence_hash(&self) -> SequenceHash {
        self.state.token_block.sequence_hash()
    }

    pub fn reset(self) -> Block<T, Reset> {
        Block {
            block_id: self.block_id,
            state: Reset,
            marker: PhantomData,
        }
    }
}

// Implementation for Registered state
impl<T> Block<T, Registered> {
    pub fn sequence_hash(&self) -> SequenceHash {
        self.state.sequence_hash
    }

    pub(crate) fn registration_handle(&self) -> &BlockRegistrationHandle {
        &self.state.registration_handle
    }

    pub fn reset(self) -> Block<T, Reset> {
        // Drop the registration handle
        Block {
            block_id: self.block_id,
            state: Reset,
            marker: PhantomData,
        }
    }
}

// Common methods for all states
impl<T, State> Block<T, State> {
    #[inline]
    pub fn block_id(&self) -> BlockId {
        self.block_id
    }
}
