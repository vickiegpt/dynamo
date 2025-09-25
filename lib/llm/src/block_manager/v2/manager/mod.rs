// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block Manager v2

use super::pools::{
    BlockDuplicationPolicy, BlockMetadata, SequenceHash,
    block::{Block, Reset},
    inactive::InactivePool,
    registry::BlockRegistry,
    reset::ResetPool,
    reuse_policy::fifo::FifoReusePolicy,
    *,
};

use derive_builder::Builder;

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct BlockManager<T: BlockMetadata> {
    reset_pool: ResetPool<T>,
    registered_pool: InactivePool<T>,
    block_registry: BlockRegistry,
    #[builder(default = "BlockDuplicationPolicy::Allow")]
    duplication_policy: BlockDuplicationPolicy,
}

impl<T: BlockMetadata> BlockManager<T> {
    pub fn builder() -> BlockManagerBuilder<T> {
        BlockManagerBuilder::default()
    }

    pub fn new(count: usize, block_registry: BlockRegistry) -> Self {
        let blocks: Vec<Block<T, Reset>> = (0..count as u64)
            .map(|id| Block::new(id))
            .collect::<Vec<_>>();

        let reset_pool = ResetPool::new(blocks);
        let reuse_policy = Box::new(FifoReusePolicy::new());
        let registered_pool = InactivePool::new(reuse_policy, &reset_pool);

        Self {
            reset_pool,
            registered_pool,
            block_registry,
            duplication_policy: BlockDuplicationPolicy::Allow,
        }
    }

    pub fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        let mut blocks = self.reset_pool.try_allocate_blocks(count);
        match self.registered_pool.allocate_blocks(count - blocks.len()) {
            Some(remaining) => {
                blocks.extend(remaining);
                Some(blocks)
            }
            None => return None,
        }
    }

    pub fn register_blocks(&self, blocks: Vec<CompleteBlock<T>>) -> Vec<ImmutableBlock<T>> {
        let pool_return_fn = self.registered_pool.return_fn();
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
        let mut matched = Vec::with_capacity(seq_hash.len());
        let pool_return_fn = self.registered_pool.return_fn();

        // match against active blocks
        for hash in seq_hash {
            match self.block_registry.match_sequence_hash(*hash) {
                Some(handle) => {
                    if let Some(block) = handle.try_get_block::<T>(pool_return_fn.clone()) {
                        matched.push(block);
                    }
                }
                None => {
                    break;
                }
            }
        }

        let remaining_hashes = &seq_hash[matched.len()..];

        // match against inactive blocks
        matched.extend(self.registered_pool.find_blocks(remaining_hashes));

        matched
    }
}
