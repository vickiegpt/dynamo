// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Basic test to verify RAII pattern functionality

#[cfg(test)]
mod tests {
    use super::super::reset::ResetPool;
    use super::super::{Block, Reset};
    use crate::tokens::TokenBlock;

    #[derive(Debug, Clone, PartialEq)]
    struct TestData {
        value: u64,
    }

    fn create_test_blocks(count: usize) -> Vec<Block<TestData, Reset>> {
        (0..count)
            .map(|i| Block::new(i as u64))
            .collect()
    }

    #[test]
    fn test_mutable_block_raii_return() {
        let blocks = create_test_blocks(3);
        let pool = ResetPool::new(blocks);

        // Initial state: pool should have 3 blocks
        assert_eq!(pool.len(), 3);

        {
            // Allocate blocks
            let allocated = pool.allocate_blocks(2).unwrap();
            assert_eq!(allocated.len(), 2);
            assert_eq!(pool.len(), 1); // 1 block remaining

            // Blocks will be automatically returned when they go out of scope
        } // allocated blocks are dropped here

        // Give a moment for async return (if any)
        std::thread::sleep(std::time::Duration::from_millis(10));

        // After drop, blocks should be returned to pool
        assert_eq!(pool.len(), 3); // All blocks returned
    }

    #[test]
    fn test_mutable_block_state_transitions() {
        let blocks = create_test_blocks(1);
        let pool = ResetPool::new(blocks);

        let allocated = pool.allocate_blocks(1).unwrap();
        let mut block = allocated.into_iter().next().unwrap();

        // Test completing a block
        let token_block = TokenBlock::default(); // Create a default token block
        block = block.complete(token_block);

        // Verify we can access token block after completion
        assert!(block.token_block().is_some());
        assert!(block.sequence_hash().is_some());

        // Block will be automatically reset and returned when dropped
    }

    #[test]
    fn test_pool_allocation_and_return_cycle() {
        let blocks = create_test_blocks(5);
        let pool = ResetPool::new(blocks);

        // Multiple allocation/return cycles
        for _ in 0..3 {
            assert_eq!(pool.len(), 5);

            {
                let allocated = pool.allocate_blocks(2).unwrap();
                assert_eq!(allocated.len(), 2);
                assert_eq!(pool.len(), 3);

                // Blocks automatically returned on drop
            }

            // Brief wait for async return
            std::thread::sleep(std::time::Duration::from_millis(5));

            // Should be back to full capacity
            assert_eq!(pool.len(), 5);
        }
    }
}