// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;

use crate::block_manager::v2::pools::{
    BlockMetadata, SequenceHash,
    block::{Block, Registered},
    frequency_sketch::FrequencyTracker,
};

use super::super::InactivePoolBackend;

pub struct MultiLruBackend<T: BlockMetadata> {
    priority_pools: [LruCache<SequenceHash, Block<T, Registered>>; 4],
    frequency_tracker: Arc<dyn FrequencyTracker>,
    frequency_thresholds: [u8; 3],
}

impl<T: BlockMetadata> MultiLruBackend<T> {
    pub fn new(capacity: NonZeroUsize, frequency_tracker: Arc<dyn FrequencyTracker>) -> Self {
        let level_capacity = NonZeroUsize::new(
            std::cmp::max(1, capacity.get() / 4)
        ).unwrap();

        Self {
            priority_pools: [
                LruCache::new(level_capacity),
                LruCache::new(level_capacity),
                LruCache::new(level_capacity),
                LruCache::new(level_capacity),
            ],
            frequency_tracker,
            frequency_thresholds: [2, 6, 15], // Old default for backward compatibility
        }
    }

    /// Create with custom frequency thresholds
    /// The 4 levels are fixed, but thresholds can be customized
    ///
    /// # Arguments
    /// * `capacity_per_level` - Capacity for each of the 4 LRU pools
    /// * `thresholds` - Array of 3 thresholds: [cold->warm, warm->hot, hot->very_hot]
    /// * `frequency_tracker` - Shared frequency tracker
    pub fn new_with_thresholds(
        capacity_per_level: NonZeroUsize,
        thresholds: &[u8; 3],
        frequency_tracker: Arc<dyn FrequencyTracker>,
    ) -> Self {
        // Validate thresholds
        debug_assert!(
            thresholds[0] < thresholds[1] && thresholds[1] < thresholds[2],
            "Thresholds must be in ascending order: {:?}",
            thresholds
        );
        debug_assert!(
            thresholds[2] <= 15,
            "Maximum threshold cannot exceed 15 (4-bit counter limit), got: {}",
            thresholds[2]
        );

        Self {
            priority_pools: [
                LruCache::new(capacity_per_level),
                LruCache::new(capacity_per_level),
                LruCache::new(capacity_per_level),
                LruCache::new(capacity_per_level),
            ],
            frequency_tracker,
            frequency_thresholds: *thresholds,
        }
    }

    fn calculate_priority_level(&self, seq_hash: SequenceHash) -> usize {
        let frequency = self.frequency_tracker.count(seq_hash);
        let [t1, t2, t3] = self.frequency_thresholds;

        if frequency < t1 as u32 {
            0  // Cold: 0 to (t1 - 1)
        } else if frequency < t2 as u32 {
            1  // Warm: t1 to (t2 - 1)
        } else if frequency < t3 as u32 {
            2  // Hot: t2 to (t3 - 1)
        } else {
            3  // Very Hot: t3 to 15
        }
    }
}

impl<T: BlockMetadata> InactivePoolBackend<T> for MultiLruBackend<T> {
    fn find_matches(&mut self, hashes: &[SequenceHash]) -> Vec<Block<T, Registered>> {
        let mut matches = Vec::with_capacity(hashes.len());

        for hash in hashes {
            let mut found = false;

            for pool in &mut self.priority_pools {
                if let Some(block) = pool.pop(hash) {
                    matches.push(block);
                    found = true;
                    break;
                }
            }

            if !found {
                break;
            }
        }

        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>> {
        let mut allocated = Vec::with_capacity(count);

        for _ in 0..count {
            let mut found = false;

            for pool in &mut self.priority_pools {
                if let Some((_seq_hash, block)) = pool.pop_lru() {
                    allocated.push(block);
                    found = true;
                    break;
                }
            }

            if !found {
                break;
            }
        }

        allocated
    }

    fn insert(&mut self, block: Block<T, Registered>) {
        let seq_hash = block.sequence_hash();
        let level = self.calculate_priority_level(seq_hash);

        // Assert the target pool isn't full (would cause eviction)
        debug_assert!(
            self.priority_pools[level].len() < self.priority_pools[level].cap().get(),
            "MultiLRU level {} insert would cause eviction! len={}, cap={}. \
             This indicates insufficient capacity for all blocks.",
            level,
            self.priority_pools[level].len(),
            self.priority_pools[level].cap().get()
        );

        self.priority_pools[level].put(seq_hash, block);
    }

    fn len(&self) -> usize {
        self.priority_pools.iter().map(|pool| pool.len()).sum()
    }

    fn has_block(&self, seq_hash: SequenceHash) -> bool {
        self.priority_pools.iter().any(|pool| pool.peek(&seq_hash).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::v2::pools::{
        block::Block,
        frequency_sketch::TinyLFUTracker,
        registry::BlockRegistry,
    };
    use crate::tokens::TokenBlockSequence;

    #[derive(Debug, Clone, PartialEq)]
    struct TestData {
        value: u64,
    }

    fn create_registered_block(id: u64, token_value: u32) -> (Block<TestData, Registered>, u64) {
        let tokens = vec![token_value, token_value + 1, token_value + 2, token_value + 3];
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
    fn test_multi_lru_priority_levels() {
        let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
        let mut backend = MultiLruBackend::new(NonZeroUsize::new(12).unwrap(), frequency_tracker.clone());

        let (block1, hash1) = create_registered_block(1, 100);
        let (block2, hash2) = create_registered_block(2, 200);
        let (block3, hash3) = create_registered_block(3, 300);
        let (block4, hash4) = create_registered_block(4, 400);

        frequency_tracker.touch(hash2);
        frequency_tracker.touch(hash2);

        for _ in 0..6 {
            frequency_tracker.touch(hash3);
        }

        for _ in 0..16 {
            frequency_tracker.touch(hash4);
        }

        let freq1 = frequency_tracker.count(hash1);
        let freq2 = frequency_tracker.count(hash2);
        let freq3 = frequency_tracker.count(hash3);
        let freq4 = frequency_tracker.count(hash4);

        assert_eq!(backend.calculate_priority_level(hash1), 0); // Cold
        assert_eq!(backend.calculate_priority_level(hash2), 1); // Warm
        assert_eq!(backend.calculate_priority_level(hash3), 2); // Hot
        assert_eq!(backend.calculate_priority_level(hash4), 3); // Very hot (15)

        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);
        backend.insert(block4);

        assert_eq!(backend.len(), 4);
        assert!(backend.has_block(hash1));
        assert!(backend.has_block(hash2));
        assert!(backend.has_block(hash3));
        assert!(backend.has_block(hash4));
    }

    #[test]
    fn test_multi_lru_eviction_order() {
        let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
        let mut backend = MultiLruBackend::new(NonZeroUsize::new(8).unwrap(), frequency_tracker.clone());

        let (block1, hash1) = create_registered_block(1, 100);
        let (block2, hash2) = create_registered_block(2, 200);
        let (block3, hash3) = create_registered_block(3, 300);

        for _ in 0..6 {
            frequency_tracker.touch(hash3);
        }

        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);

        let allocated = backend.allocate(2);
        assert_eq!(allocated.len(), 2);
        assert_eq!(allocated[0].block_id(), 1);
        assert_eq!(allocated[1].block_id(), 2);

        assert!(!backend.has_block(hash1));
        assert!(!backend.has_block(hash2));
        assert!(backend.has_block(hash3));
    }

    #[test]
    fn test_multi_lru_find_matches() {
        let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
        let mut backend = MultiLruBackend::new(NonZeroUsize::new(8).unwrap(), frequency_tracker.clone());

        let (block1, hash1) = create_registered_block(1, 100);
        let (block2, hash2) = create_registered_block(2, 200);
        let (block3, hash3) = create_registered_block(3, 300);

        for _ in 0..3 {
            frequency_tracker.touch(hash2);
        }

        for _ in 0..10 {
            frequency_tracker.touch(hash3);
        }

        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);

        let matches = backend.find_matches(&[hash1, hash2, hash3]);
        assert_eq!(matches.len(), 3);
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_multi_lru_capacity_distribution() {
        let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
        let mut backend = MultiLruBackend::new(NonZeroUsize::new(8).unwrap(), frequency_tracker.clone());

        let (block1, hash1) = create_registered_block(1, 100);
        let (block2, hash2) = create_registered_block(2, 200);
        let (block3, hash3) = create_registered_block(3, 300);
        let (block4, hash4) = create_registered_block(4, 400);

        for _ in 0..3 {
            frequency_tracker.touch(hash2);
        }

        for _ in 0..7 {
            frequency_tracker.touch(hash3);
        }

        for _ in 0..15 {
            frequency_tracker.touch(hash4);
        }

        backend.insert(block1);
        backend.insert(block2);
        backend.insert(block3);
        backend.insert(block4);

        assert_eq!(backend.len(), 4);
        assert!(backend.has_block(hash1));
        assert!(backend.has_block(hash2));
        assert!(backend.has_block(hash3));
        assert!(backend.has_block(hash4));

        let (block5, hash5) = create_registered_block(5, 500);
        let (block6, hash6) = create_registered_block(6, 600);
        let (block7, hash7) = create_registered_block(7, 700);
        let (block8, hash8) = create_registered_block(8, 800);

        backend.insert(block5);
        backend.insert(block6);
        backend.insert(block7);
        backend.insert(block8);

        let current_len = backend.len();
        assert!(current_len >= 4 && current_len <= 8);

        let (block9, _hash9) = create_registered_block(9, 900);
        backend.insert(block9);

        let new_len = backend.len();
        assert!(new_len >= 4 && new_len <= 8);
    }
}