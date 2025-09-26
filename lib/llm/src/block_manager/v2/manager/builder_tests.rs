// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for BlockManager builder pattern

#[cfg(test)]
mod tests {
    use super::super::BlockManager;

    #[derive(Debug, Clone, PartialEq)]
    struct TestBlockData {
        value: u32,
    }

    #[test]
    fn test_builder_default() {
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .build()
            .expect("Should build with defaults");

        // Verify we can allocate blocks
        let blocks = manager.allocate_blocks(5);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 5);
    }

    #[test]
    fn test_builder_with_lru_backend() {
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .with_lru_backend()
            .build()
            .expect("Should build with LRU backend");

        // Verify we can allocate blocks
        let blocks = manager.allocate_blocks(10);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 10);
    }

    #[test]
    fn test_builder_with_multi_lru_backend() {
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .frequency_tracker_size(1 << 20) // 2^20
            .with_multi_lru_backend()
            .build()
            .expect("Should build with MultiLRU backend");

        // Verify we can allocate blocks
        let blocks = manager.allocate_blocks(8);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 8);
    }

    #[test]
    fn test_builder_with_custom_multi_lru_thresholds() {
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .frequency_tracker_size(1 << 21) // 2^21 (default)
            .with_multi_lru_backend_custom_thresholds(2, 6, 12)
            .build()
            .expect("Should build with custom thresholds");

        // Verify we can allocate blocks
        let blocks = manager.allocate_blocks(4);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 4);
    }

    #[test]
    fn test_builder_validation_zero_blocks() {
        let result = BlockManager::<TestBlockData>::builder()
            .block_count(0)
            .build();

        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("block_count must be greater than 0"));
        }
    }

    #[test]
    #[should_panic(expected = "must be <= 15")]
    fn test_builder_invalid_threshold_too_high() {
        BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .with_multi_lru_backend_custom_thresholds(2, 6, 20); // 20 > 15, should panic
    }

    #[test]
    #[should_panic(expected = "must be in ascending order")]
    fn test_builder_invalid_threshold_order() {
        BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .with_multi_lru_backend_custom_thresholds(6, 2, 10); // Not ascending, should panic
    }

    #[test]
    #[should_panic(expected = "must be between 2^18 and 2^24")]
    fn test_builder_invalid_frequency_tracker_size() {
        BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .frequency_tracker_size(1000); // Not a valid size, should panic
    }

    #[test]
    #[should_panic(expected = "must be a power of 2")]
    fn test_builder_non_power_of_two_frequency_tracker() {
        BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .frequency_tracker_size((1 << 20) + 1); // Not power of 2, should panic
    }
}