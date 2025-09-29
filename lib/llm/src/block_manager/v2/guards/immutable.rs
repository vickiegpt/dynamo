// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guards for immutable and weak block references

use std::{
    ops::Deref,
    sync::{Arc, Weak},
};

use super::{super::pools::BlockMetadata, RegisteredBlock};
use crate::tokens::SequenceHash;

/// RAII guard for registered blocks with upgrade capability
pub struct ImmutableBlock<T: BlockMetadata> {
    block: Arc<dyn RegisteredBlock<T>>,
    upgrade_fn: Arc<dyn Fn(SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> + Send + Sync>,
}

/// Weak reference to a registered block with upgrade capability
pub struct WeakBlock<T: BlockMetadata> {
    sequence_hash: SequenceHash,
    block: Weak<dyn RegisteredBlock<T>>,
    upgrade_fn: Arc<dyn Fn(SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> + Send + Sync>,
}

impl<T: BlockMetadata> ImmutableBlock<T> {
    /// Create a new ImmutableBlock with an upgrade function
    pub fn new(
        block: Arc<dyn RegisteredBlock<T>>,
        upgrade_fn: Arc<dyn Fn(SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> + Send + Sync>,
    ) -> Self {
        Self { block, upgrade_fn }
    }

    /// Downgrade to a WeakBlock
    pub fn downgrade(&self) -> WeakBlock<T> {
        WeakBlock {
            sequence_hash: self.sequence_hash(),
            block: Arc::downgrade(&self.block),
            upgrade_fn: self.upgrade_fn.clone(),
        }
    }
}

impl<T: BlockMetadata> WeakBlock<T> {
    /// Try to upgrade this WeakBlock back to an ImmutableBlock
    pub fn upgrade(&self) -> Option<ImmutableBlock<T>> {
        // First try to upgrade the weak reference directly
        if let Some(block) = self.block.upgrade() {
            return Some(ImmutableBlock::new(block, self.upgrade_fn.clone()));
        }

        // If that fails, use the upgrade function to search for the block
        if let Some(block) = (self.upgrade_fn)(self.sequence_hash) {
            return Some(ImmutableBlock::new(block, self.upgrade_fn.clone()));
        }

        None
    }

    /// Get the sequence hash
    pub fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }
}

impl<T: BlockMetadata> Deref for ImmutableBlock<T> {
    type Target = dyn RegisteredBlock<T>;

    fn deref(&self) -> &Self::Target {
        self.block.as_ref()
    }
}
