// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;

use lru::LruCache;

use crate::block_manager::v2::pools::{
    BlockMetadata, SequenceHash,
    block::{Block, Registered},
};

use super::super::InactivePoolBackend;

pub struct LruBackend<T: BlockMetadata> {
    cache: LruCache<SequenceHash, Block<T, Registered>>,
}

impl<T: BlockMetadata> LruBackend<T> {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            cache: LruCache::new(capacity),
        }
    }
}

impl<T: BlockMetadata> InactivePoolBackend<T> for LruBackend<T> {
    fn find_matches(&mut self, hashes: &[SequenceHash]) -> Vec<Block<T, Registered>> {
        let mut matches = Vec::with_capacity(hashes.len());

        for hash in hashes {
            if let Some(block) = self.cache.pop(hash) {
                matches.push(block);
            } else {
                break;
            }
        }

        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>> {
        let mut allocated = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some((_seq_hash, block)) = self.cache.pop_lru() {
                allocated.push(block);
            } else {
                break;
            }
        }

        allocated
    }

    fn insert(&mut self, block: Block<T, Registered>) {
        let seq_hash = block.sequence_hash();

        // Assert we're not causing an eviction
        debug_assert!(
            self.cache.len() < self.cache.cap().get(),
            "LRU backend insert would cause eviction! len={}, cap={}. \
             This indicates insufficient capacity for all blocks.",
            self.cache.len(),
            self.cache.cap().get()
        );

        self.cache.put(seq_hash, block);
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn has_block(&self, seq_hash: SequenceHash) -> bool {
        self.cache.peek(&seq_hash).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::v2::pools::{block::Block, registry::BlockRegistry};
    use crate::tokens::TokenBlockSequence;

    #[derive(Debug, Clone, PartialEq)]
    struct TestData {
        value: u64,
    }

    fn create_registered_block(id: u64, token_value: u32) -> (Block<TestData, Registered>, u64) {
        let tokens = vec![
            token_value,
            token_value + 1,
            token_value + 2,
            token_value + 3,
        ];
        let token_block_seq = TokenBlockSequence::from_slice(&tokens, 4, Some(42));
        let token_block = if let Some(block) = token_block_seq.blocks().first() {
            block.clone()
        } else {
            let mut partial = token_block_seq.into_parts().1;
            partial.commit().expect("Should be able to commit")
        };

        let actual_seq_hash = token_block.sequence_hash();
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(actual_seq_hash);

        let final_block = Block::new(id).complete(token_block).register(handle);
        (final_block, actual_seq_hash)
    }

    #[test]
    fn test_lru_eviction_order() {
        let mut backend = LruBackend::new(NonZeroUsize::new(3).unwrap());

        let (block1, hash1) = create_registered_block(1, 100);
        let (block2, hash2) = create_registered_block(2, 200);
        let (block3, hash3) = create_registered_block(3, 300);

        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);

        assert_eq!(backend.len(), 3);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].block_id(), 1);

        assert!(!backend.has_block(hash1));
        assert!(backend.has_block(hash2));
        assert!(backend.has_block(hash3));
    }

    #[test]
    fn test_lru_capacity_limit() {
        let mut backend = LruBackend::new(NonZeroUsize::new(2).unwrap());

        let (block1, hash1) = create_registered_block(1, 100);
        let (block2, hash2) = create_registered_block(2, 200);
        let (block3, hash3) = create_registered_block(3, 300);

        backend.insert(block1);
        backend.insert(block2);
        assert_eq!(backend.len(), 2);

        backend.insert(block3);
        assert_eq!(backend.len(), 2);

        assert!(!backend.has_block(hash1));
        assert!(backend.has_block(hash2));
        assert!(backend.has_block(hash3));
    }

    #[test]
    fn test_lru_peek_doesnt_affect_order() {
        let mut backend = LruBackend::new(NonZeroUsize::new(2).unwrap());

        let (block1, hash1) = create_registered_block(1, 100);
        let (block2, hash2) = create_registered_block(2, 200);

        backend.insert(block1);
        backend.insert(block2);

        assert!(backend.has_block(hash1));

        let (block3, hash3) = create_registered_block(3, 300);
        backend.insert(block3);

        assert!(!backend.has_block(hash1));
        assert!(backend.has_block(hash2));
        assert!(backend.has_block(hash3));
    }

    #[test]
    fn test_lru_allocate_more_than_available() {
        let mut backend = LruBackend::new(NonZeroUsize::new(10).unwrap());

        let (block1, _) = create_registered_block(1, 100);
        let (block2, _) = create_registered_block(2, 200);
        backend.insert(block1);
        backend.insert(block2);

        let allocated = backend.allocate(5);
        assert_eq!(allocated.len(), 2);
        assert_eq!(backend.len(), 0);
    }
}
