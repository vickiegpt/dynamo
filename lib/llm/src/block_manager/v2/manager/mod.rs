// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block Manager v2

pub mod builder;

#[cfg(test)]
mod builder_tests;

pub use builder::{BlockManagerConfigBuilder, BlockManagerBuilderError, InactiveBackendConfig};

use super::pools::{
    ActivePool, BlockDuplicationPolicy, BlockMetadata, SequenceHash,
    inactive::InactivePool,
    registry::BlockRegistry,
    reset::ResetPool,
    *,
};

/// BlockManager v2 with pluggable inactive pool backends
pub struct BlockManager<T: BlockMetadata> {
    reset_pool: ResetPool<T>,
    active_pool: ActivePool<T>,
    inactive_pool: InactivePool<T>,
    block_registry: BlockRegistry,
    duplication_policy: BlockDuplicationPolicy,
}

impl<T: BlockMetadata> BlockManager<T> {
    /// Create a new builder for BlockManager
    ///
    /// # Example
    /// ```ignore
    /// let manager = BlockManager::builder()
    ///     .block_count(1000)
    ///     .with_multi_lru_backend()
    ///     .build()?;
    /// ```
    pub fn builder() -> BlockManagerConfigBuilder<T> {
        BlockManagerConfigBuilder::default()
    }

    pub fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        let mut blocks = self.reset_pool.try_allocate_blocks(count);
        match self.inactive_pool.allocate_blocks(count - blocks.len()) {
            Some(remaining) => {
                blocks.extend(remaining);
                Some(blocks)
            }
            None => return None,
        }
    }

    pub fn register_blocks(&self, blocks: Vec<CompleteBlock<T>>) -> Vec<ImmutableBlock<T>> {
        let pool_return_fn = self.inactive_pool.return_fn();
        blocks
            .into_iter()
            .map(|block| {
                let handle = self
                    .block_registry
                    .register_sequence_hash(block.sequence_hash());
                handle.register_block(block, self.duplication_policy, pool_return_fn.clone())
            })
            .collect()
    }

    pub fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>> {
        // First try to match against active blocks
        let matched = self.active_pool.find_matches(seq_hash);

        // If we didn't match all hashes, try inactive blocks for the remaining ones
        let remaining_hashes = &seq_hash[matched.len()..];
        if !remaining_hashes.is_empty() {
            let mut all_matched = matched;
            all_matched.extend(self.inactive_pool.find_blocks(remaining_hashes));
            all_matched
        } else {
            matched
        }
    }
}
