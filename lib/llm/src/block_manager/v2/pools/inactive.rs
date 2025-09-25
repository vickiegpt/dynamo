// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inactive block pool that manages blocks that are inactive which may or may
//! not have state carried over from the last time they were used.

// ReturnableBlock is no longer used - replaced with RAII guards

use super::*;

/// Inactive block pool that manages available blocks via a free list.
///
/// This pool maintains blocks that are available for allocation.
/// It uses a configurable free list implementation for block selection strategy.
#[derive(Debug)]
pub struct InactiveBlockPool<T> {
    // Blocks in the Reset state
    reset_blocks: VecDeque<Block<T, Reset>>,

    // Blocks in the Registered state
    registered_blocks: HashMap<SequenceHash, Block<T, Registered>>,

    // Resuse Policy
    reuse_policy: Box<dyn ReusePolicy>,
}

impl<T> InactiveBlockPool<T> {
    pub fn new(reuse_policy: Box<dyn ReusePolicy>) -> Self {
        Self {
            reuse_policy,
            reset_blocks: VecDeque::new(),
            registered_blocks: HashMap::new(),
        }
    }

    /// Allocate a specified number of blocks
    pub fn allocate_blocks(&mut self, count: usize) -> Option<Vec<Block<T, Reset>>> {
        // Early exit if not enough blocks available
        if self.available_blocks() < count {
            return None;
        }

        // Now we know we have enough blocks, allocation cannot fail
        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            let block = self
                .allocate_block()
                .expect("allocate_block failed despite availability check");
            blocks.push(block);
        }
        Some(blocks)
    }

    /// Check if a block exists in the inactive pool
    pub fn has_block(&self, hash: SequenceHash) -> bool {
        self.registered_blocks.contains_key(&hash)
    }

    /// Find blocks by sequence hashes in the inactive pool
    pub fn find_blocks(&mut self, hashes: &[SequenceHash]) -> Vec<Block<T, Registered>> {
        let mut blocks: Vec<Block<T, Registered>> = Vec::with_capacity(hashes.len());
        for hash in hashes {
            if let Some(block) = self.registered_blocks.remove(hash) {
                self.reuse_policy
                    .remove(block.block_id())
                    .expect("Block not found in reuse policy");
                blocks.push(block);
            } else {
                break;
            }
        }

        blocks
    }

    // ReturnableBlock methods removed - now using RAII guards
    // Blocks are automatically returned via Drop implementations

    fn allocate_block(&mut self) -> Option<Block<T, Reset>> {
        self.reset_blocks.pop_front().or_else(|| {
            self.reuse_policy.next_free().and_then(|inactive_block| {
                Some(
                    self.registered_blocks
                        .remove(&inactive_block.seq_hash)
                        .expect("Block not found in registered blocks")
                        .reset(),
                )
            })
        })
    }

    /// Return a reset block
    fn return_reset_block(&mut self, block: Block<T, Reset>) {
        self.reset_blocks.push_back(block);
    }

    /// Return a shared registered block
    fn return_shared_registered_block(&mut self, block: Arc<Block<T, Registered>>) {
        if let Ok(block) = Arc::try_unwrap(block) {
            self.return_registered_block(block);
        }
    }

    /// Return a registered block
    fn return_registered_block(&mut self, block: Block<T, Registered>) {
        // if we can take full ownership, we can insert it into the reuse policy
        // otherwise, there are outstanding strong references.
        let seq_hash = block.sequence_hash();
        let _ = self.reuse_policy.insert(InactiveBlock {
            block_id: block.block_id(),
            seq_hash: seq_hash,
        });
        self.registered_blocks.insert(seq_hash, block);
    }

    /// Get the number of available blocks
    pub fn available_blocks(&self) -> usize {
        self.reset_blocks.len() + self.registered_blocks.len()
    }

    /// Check if the pool has available blocks
    pub fn has_available_blocks(&self) -> bool {
        !self.reset_blocks.is_empty() || !self.registered_blocks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::v2::{
        ReturnableBlock,
        block::{Block, Complete, Registered, Reset},
        pools::reuse_policy::fifo::FifoReusePolicy,
        registry::BlockRegistry,
    };
    use std::sync::Arc;

    /// Test data structure for unit tests
    #[derive(Debug, Clone, PartialEq)]
    struct TestData {
        value: u64,
    }

    /// Helper function to create a reset block with predictable ID
    fn create_reset_block(id: u64) -> Block<TestData, Reset> {
        Block::new(id)
    }

    /// Helper function to create a complete block
    fn create_complete_block(id: u64, seq_hash: u64) -> Block<TestData, Complete> {
        Block::new(id).complete(seq_hash)
    }

    /// Helper function to create a registered block
    fn create_registered_block(id: u64, seq_hash: u64) -> Block<TestData, Registered> {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence(seq_hash);
        create_complete_block(id, seq_hash).register(handle)
    }

    /// Helper to create a test pool with FIFO reuse policy
    fn create_test_pool() -> InactiveBlockPool<TestData> {
        let reuse_policy = Box::new(FifoReusePolicy::new());
        InactiveBlockPool::new(reuse_policy)
    }

    /// Helper to assert pool state matches expected counts
    fn assert_pool_state(
        pool: &InactiveBlockPool<TestData>,
        expected_reset: usize,
        expected_registered: usize,
    ) {
        let total_expected = expected_reset + expected_registered;
        assert_eq!(
            pool.available_blocks(),
            total_expected,
            "Total available blocks should be {}",
            total_expected
        );
        assert_eq!(
            pool.has_available_blocks(),
            total_expected > 0,
            "has_available_blocks should be {}",
            total_expected > 0
        );
    }

    // =========================================================================
    // Pool Initialization & Basic State Tests
    // =========================================================================

    #[test]
    fn test_new_pool_starts_empty() {
        let pool = create_test_pool();

        assert_eq!(pool.available_blocks(), 0);
        assert!(!pool.has_available_blocks());
        assert!(!pool.has_block(100)); // Random sequence hash
    }

    #[test]
    fn test_pool_with_fifo_reuse_policy() {
        let reuse_policy = Box::new(FifoReusePolicy::new());
        let pool = InactiveBlockPool::<TestData>::new(reuse_policy);

        // Basic verification that pool was created successfully
        assert_eq!(pool.available_blocks(), 0);
        assert!(!pool.has_available_blocks());
    }

    // =========================================================================
    // Reset Block Operations Tests
    // =========================================================================

    #[test]
    fn test_return_and_allocate_single_reset_block() {
        let mut pool = create_test_pool();

        // Return a reset block to the pool
        let block = create_reset_block(1);
        pool.return_block(ReturnableBlock::Reset(block));

        assert_pool_state(&pool, 1, 0);

        // Allocate the block back
        let blocks = pool.allocate_blocks(1).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_id(), 1);

        assert_pool_state(&pool, 0, 0);
    }

    #[test]
    fn test_return_multiple_reset_blocks_fifo_order() {
        let mut pool = create_test_pool();

        // Return blocks in order: 1, 2, 3
        for id in 1..=3 {
            let block = create_reset_block(id);
            pool.return_block(ReturnableBlock::Reset(block));
        }

        assert_pool_state(&pool, 3, 0);

        // Allocate blocks - should come out in FIFO order: 1, 2, 3
        let blocks = pool.allocate_blocks(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].block_id(), 1);
        assert_eq!(blocks[1].block_id(), 2);
        assert_eq!(blocks[2].block_id(), 3);
    }

    #[test]
    fn test_allocate_blocks_returns_reset_blocks_first() {
        let mut pool = create_test_pool();

        // Add both reset and registered blocks
        let reset_block = create_reset_block(1);
        pool.return_block(ReturnableBlock::Reset(reset_block));

        let registered_block = create_registered_block(2, 200);
        pool.return_registered_block(registered_block);

        assert_pool_state(&pool, 1, 1);

        // Allocate one block - should get the reset block first
        let blocks = pool.allocate_blocks(1).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_id(), 1); // Reset block came first

        assert_pool_state(&pool, 0, 1);
    }

    // =========================================================================
    // Registered Block Operations Tests
    // =========================================================================

    #[test]
    fn test_return_registered_block_adds_to_pool() {
        let mut pool = create_test_pool();

        let registered_block = create_registered_block(1, 100);
        let seq_hash = registered_block.seq_hash();

        pool.return_registered_block(registered_block);

        assert_pool_state(&pool, 0, 1);
        assert!(pool.has_block(seq_hash));
        assert_eq!(pool.reuse_policy.len(), 1);
    }

    #[test]
    fn test_allocate_registered_block_transitions_to_reset() {
        let mut pool = create_test_pool();

        // Return a registered block
        let registered_block = create_registered_block(1, 100);
        pool.return_registered_block(registered_block);

        assert_pool_state(&pool, 0, 1);

        // Allocate it - should come back as reset block
        let blocks = pool.allocate_blocks(1).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_id(), 1);
        // Block should now be in Reset state (verified by successful allocation)

        assert_pool_state(&pool, 0, 0);
    }

    #[test]
    fn test_registered_blocks_follow_reuse_policy_order() {
        let mut pool = create_test_pool();

        // Return multiple registered blocks with delays to ensure FIFO order
        for id in 1..=3 {
            let block = create_registered_block(id, id * 100);
            pool.return_registered_block(block);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        assert_pool_state(&pool, 0, 3);

        // Allocate all blocks - should follow FIFO order from reuse policy
        let blocks = pool.allocate_blocks(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].block_id(), 1); // First returned
        assert_eq!(blocks[1].block_id(), 2); // Second returned
        assert_eq!(blocks[2].block_id(), 3); // Third returned
    }

    // =========================================================================
    // Mixed State Operations Tests
    // =========================================================================

    #[test]
    fn test_mixed_pool_accurate_counting() {
        let mut pool = create_test_pool();

        // Add 2 reset blocks and 3 registered blocks
        for id in 1..=2 {
            let block = create_reset_block(id);
            pool.return_block(ReturnableBlock::Reset(block));
        }

        for id in 3..=5 {
            let block = create_registered_block(id, id * 100);
            pool.return_registered_block(block);
        }

        assert_pool_state(&pool, 2, 3);
        assert_eq!(pool.available_blocks(), 5);
        assert!(pool.has_available_blocks());
    }

    #[test]
    fn test_allocation_prefers_reset_over_registered() {
        let mut pool = create_test_pool();

        // Add registered block first
        let registered_block = create_registered_block(1, 100);
        pool.return_registered_block(registered_block);

        // Then add reset block
        let reset_block = create_reset_block(2);
        pool.return_block(ReturnableBlock::Reset(reset_block));

        assert_pool_state(&pool, 1, 1);

        // Allocate one block - should get reset block despite registered being added first
        let blocks = pool.allocate_blocks(1).unwrap();
        assert_eq!(blocks[0].block_id(), 2); // Reset block

        assert_pool_state(&pool, 0, 1);
    }

    // =========================================================================
    // Batch Allocation Tests
    // =========================================================================

    #[test]
    fn test_allocate_exact_available_count() {
        let mut pool = create_test_pool();

        // Add exactly 3 blocks
        for id in 1..=3 {
            let block = create_reset_block(id);
            pool.return_block(ReturnableBlock::Reset(block));
        }

        assert_pool_state(&pool, 3, 0);

        // Allocate exactly all blocks
        let blocks = pool.allocate_blocks(3).unwrap();
        assert_eq!(blocks.len(), 3);

        assert_pool_state(&pool, 0, 0);
    }

    #[test]
    fn test_allocate_more_than_available_fails() {
        let mut pool = create_test_pool();

        // Add only 2 blocks
        for id in 1..=2 {
            let block = create_reset_block(id);
            pool.return_block(ReturnableBlock::Reset(block));
        }

        assert_pool_state(&pool, 2, 0);

        // Try to allocate 3 blocks - should fail and rollback
        let result = pool.allocate_blocks(3);
        assert!(result.is_none());

        // All blocks should still be in the pool (rollback succeeded)
        assert_pool_state(&pool, 2, 0);
    }

    #[test]
    fn test_partial_allocation_rollback() {
        let mut pool = create_test_pool();

        // Add 1 reset block and 1 registered block (total 2)
        let reset_block = create_reset_block(1);
        pool.return_block(ReturnableBlock::Reset(reset_block));

        let registered_block = create_registered_block(2, 200);
        pool.return_registered_block(registered_block);

        assert_pool_state(&pool, 1, 1);

        // Try to allocate 3 blocks - should fail due to insufficient blocks
        // (Early exit: no rollback needed since no allocation is attempted)
        let result = pool.allocate_blocks(3);
        assert!(result.is_none());

        // Verify no blocks were touched: reset and registered blocks unchanged
        assert_pool_state(&pool, 1, 1);
    }

    #[test]
    fn test_allocate_zero_blocks_succeeds() {
        let mut pool = create_test_pool();

        let blocks = pool.allocate_blocks(0).unwrap();
        assert_eq!(blocks.len(), 0);

        assert_pool_state(&pool, 0, 0);
    }

    // =========================================================================
    // Find Blocks Operations Tests
    // =========================================================================

    #[test]
    fn test_find_single_existing_block() {
        let mut pool = create_test_pool();

        let registered_block = create_registered_block(1, 100);
        pool.return_registered_block(registered_block);

        assert_pool_state(&pool, 0, 1);
        assert!(pool.has_block(100));

        // Find the block
        let found_blocks = pool.find_blocks(&[100]);
        assert_eq!(found_blocks.len(), 1);
        assert_eq!(found_blocks[0].block_id(), 1);
        assert_eq!(found_blocks[0].seq_hash(), 100);

        // Block should be removed from pool after finding
        assert_pool_state(&pool, 0, 0);
        assert!(!pool.has_block(100));
    }

    #[test]
    fn test_find_multiple_blocks_in_order() {
        let mut pool = create_test_pool();

        // Add blocks with sequence hashes 100, 200, 300
        for (id, seq_hash) in [(1, 100), (2, 200), (3, 300)] {
            let block = create_registered_block(id, seq_hash);
            pool.return_registered_block(block);
        }

        assert_pool_state(&pool, 0, 3);

        // Find blocks in a different order
        let found_blocks = pool.find_blocks(&[300, 100, 200]);
        assert_eq!(found_blocks.len(), 3);

        // Should get blocks in the order they were requested
        assert_eq!(found_blocks[0].seq_hash(), 300);
        assert_eq!(found_blocks[1].seq_hash(), 100);
        assert_eq!(found_blocks[2].seq_hash(), 200);

        assert_pool_state(&pool, 0, 0);
    }

    #[test]
    fn test_find_blocks_stops_on_first_miss() {
        let mut pool = create_test_pool();

        // Add blocks with sequence hashes 100 and 300 (missing 200)
        let block1 = create_registered_block(1, 100);
        pool.return_registered_block(block1);

        let block3 = create_registered_block(3, 300);
        pool.return_registered_block(block3);

        assert_pool_state(&pool, 0, 2);

        // Find blocks 100, 200, 300 - should stop at first miss (200)
        let found_blocks = pool.find_blocks(&[100, 200, 300]);
        assert_eq!(found_blocks.len(), 1); // Only found first block
        assert_eq!(found_blocks[0].seq_hash(), 100);

        // Block 300 should still be in pool since search stopped at 200
        assert_pool_state(&pool, 0, 1);
        assert!(pool.has_block(300));
    }

    #[test]
    fn test_find_nonexistent_blocks_returns_empty() {
        let mut pool = create_test_pool();

        // Find blocks that don't exist
        let found_blocks = pool.find_blocks(&[999, 888]);
        assert_eq!(found_blocks.len(), 0);

        assert_pool_state(&pool, 0, 0);
    }

    #[test]
    fn test_has_block_check() {
        let mut pool = create_test_pool();

        let registered_block = create_registered_block(1, 100);
        pool.return_registered_block(registered_block);

        assert!(pool.has_block(100));
        assert!(!pool.has_block(200));

        // Remove the block and check again
        let _found = pool.find_blocks(&[100]);
        assert!(!pool.has_block(100));
    }

    // =========================================================================
    // SharedRegistered Block Handling Tests
    // =========================================================================

    #[test]
    fn test_return_shared_registered_with_single_ref() {
        let mut pool = create_test_pool();

        let registered_block = create_registered_block(1, 100);
        let shared_block = Arc::new(registered_block);

        // Return the shared block (single reference should unwrap successfully)
        pool.return_block(ReturnableBlock::SharedRegistered(shared_block));

        assert_pool_state(&pool, 0, 1);
        assert!(pool.has_block(100));
    }

    #[test]
    fn test_return_shared_registered_with_multiple_refs() {
        let mut pool = create_test_pool();

        let registered_block = create_registered_block(1, 100);
        let shared_block = Arc::new(registered_block);
        let _extra_ref = Arc::clone(&shared_block); // Keep an extra reference

        // Return the shared block (should not unwrap due to extra reference)
        pool.return_block(ReturnableBlock::SharedRegistered(shared_block));

        // Block should not be added to pool since Arc::try_unwrap failed
        assert_pool_state(&pool, 0, 0);
    }

    #[test]
    fn test_shared_registered_lifecycle() {
        let mut pool = create_test_pool();

        // Create and return a shared registered block
        let registered_block = create_registered_block(1, 100);
        let shared_block = Arc::new(registered_block);

        pool.return_block(ReturnableBlock::SharedRegistered(shared_block));
        assert_pool_state(&pool, 0, 1);

        // Find and get the block back
        let found_blocks = pool.find_blocks(&[100]);
        assert_eq!(found_blocks.len(), 1);
        assert_eq!(found_blocks[0].block_id(), 1);

        assert_pool_state(&pool, 0, 0);
    }

    // =========================================================================
    // Return Blocks Variants Tests
    // =========================================================================

    #[test]
    fn test_return_blocks_batch_operation() {
        let mut pool = create_test_pool();

        // Return mixed block types
        let returnable_blocks = vec![
            ReturnableBlock::Reset(create_reset_block(1)),
            ReturnableBlock::SharedRegistered(Arc::new(create_registered_block(3, 300))),
        ];
        pool.return_blocks(returnable_blocks);

        // Return registered block directly (since it's no longer in ReturnableBlock enum)
        pool.return_registered_block(create_registered_block(2, 200));

        assert_pool_state(&pool, 1, 2);
        assert!(pool.has_block(200));
        assert!(pool.has_block(300));
    }

    #[test]
    fn test_return_different_block_types() {
        let mut pool = create_test_pool();

        // Return each type of block individually
        pool.return_block(ReturnableBlock::Reset(create_reset_block(1)));
        pool.return_registered_block(create_registered_block(2, 200));
        pool.return_block(ReturnableBlock::SharedRegistered(Arc::new(
            create_registered_block(3, 300),
        )));

        assert_pool_state(&pool, 1, 2);

        // Verify we can allocate and find blocks correctly
        let allocated = pool.allocate_blocks(1).unwrap();
        assert_eq!(allocated[0].block_id(), 1); // Reset block

        let found = pool.find_blocks(&[200]);
        assert_eq!(found[0].seq_hash(), 200);

        assert_pool_state(&pool, 0, 1); // One registered block remaining
    }

    // =========================================================================
    // Performance & Scale Tests
    // =========================================================================

    #[test]
    fn test_large_pool_operations() {
        let mut pool = create_test_pool();

        // Add 100 reset blocks and 100 registered blocks
        for id in 1..=100 {
            let reset_block = create_reset_block(id);
            pool.return_block(ReturnableBlock::Reset(reset_block));
        }

        for id in 101..=200 {
            let registered_block = create_registered_block(id, id * 10);
            pool.return_registered_block(registered_block);
        }

        assert_pool_state(&pool, 100, 100);

        // Allocate half the blocks
        let allocated = pool.allocate_blocks(100).unwrap();
        assert_eq!(allocated.len(), 100);

        assert_pool_state(&pool, 0, 100);

        // Find some registered blocks
        let seq_hashes: Vec<u64> = (1010..=1020).map(|id| id * 10).collect();
        let found = pool.find_blocks(&seq_hashes);
        assert_eq!(found.len(), 11); // Found all requested blocks

        assert_pool_state(&pool, 0, 89); // 100 - 11 found blocks
    }

    #[test]
    fn test_allocation_deallocation_cycles() {
        let mut pool = create_test_pool();

        // Perform multiple allocation/deallocation cycles
        for cycle in 1..=10 {
            // Add blocks
            for id in 1..=5 {
                let block_id = cycle * 100 + id;
                let block = create_reset_block(block_id);
                pool.return_block(ReturnableBlock::Reset(block));
            }

            assert_eq!(pool.available_blocks(), 5);

            // Allocate all blocks
            let allocated = pool.allocate_blocks(5).unwrap();
            assert_eq!(allocated.len(), 5);
            assert_eq!(pool.available_blocks(), 0);

            // Return them as registered blocks
            for block in allocated {
                let seq_hash = (block.block_id() * 10) as u64;
                let complete_block = block.complete(seq_hash);
                let registry = BlockRegistry::new();
                let handle = registry.register_sequence(seq_hash);
                let registered_block = complete_block.register(handle);
                pool.return_registered_block(registered_block);
            }

            assert_eq!(pool.available_blocks(), 5);
        }

        // Final state check
        assert_pool_state(&pool, 0, 50); // 10 cycles * 5 blocks each
    }
}
