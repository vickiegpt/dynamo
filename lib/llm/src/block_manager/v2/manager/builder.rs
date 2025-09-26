// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builder pattern for BlockManager with ergonomic backend configuration.

use std::num::NonZeroUsize;
use std::sync::Arc;


use crate::block_manager::v2::pools::{
    ActivePool, BlockDuplicationPolicy, BlockMetadata,
    block::{Block, Reset},
    frequency_sketch::TinyLFUTracker,
    inactive::{
        InactivePool,
        backends::{
            hashmap_backend::HashMapBackend,
            lru_backend::LruBackend,
            multi_lru_backend::MultiLruBackend,
            reuse::fifo::FifoReusePolicy,
            InactivePoolBackend,
            ReusePolicy,
        },
    },
    registry::BlockRegistry,
    reset::ResetPool,
};

use super::BlockManager;

/// Configuration for different inactive pool backends
pub enum InactiveBackendConfig {
    /// HashMap with configurable reuse policy
    HashMap {
        reuse_policy: Box<dyn ReusePolicy>,
    },
    /// Simple LRU - capacity automatically set to block_count
    Lru,
    /// Multi-level LRU with 4 fixed levels - capacity automatically set to block_count
    MultiLru {
        /// Frequency thresholds: [cold->warm, warm->hot, hot->very_hot]
        /// Default: [3, 8, 15]
        frequency_thresholds: [u8; 3],
    },
}

/// Builder for BlockManager configuration
pub struct BlockManagerConfigBuilder<T: BlockMetadata> {
    /// Number of blocks in the pool
    block_count: Option<usize>,

    /// Frequency tracker size for TinyLFU (must be power of 2)
    /// Default: 2^21, Min: 2^18, Max: 2^24
    frequency_tracker_size: Option<usize>,

    /// Inactive pool backend configuration
    inactive_backend: Option<InactiveBackendConfig>,

    /// Policy for handling duplicate sequence hashes
    duplication_policy: Option<BlockDuplicationPolicy>,

    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

/// Error types for BlockManager builder
#[derive(Debug, thiserror::Error)]
pub enum BlockManagerBuilderError {
    #[error("Block count must be greater than 0")]
    InvalidBlockCount,
    #[error("Invalid backend configuration: {0}")]
    InvalidBackend(String),
    #[error("Builder validation failed: {0}")]
    ValidationError(String),
}

impl<T: BlockMetadata> Default for BlockManagerConfigBuilder<T> {
    fn default() -> Self {
        Self {
            block_count: None,
            frequency_tracker_size: None,
            inactive_backend: None,
            duplication_policy: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: BlockMetadata> BlockManagerConfigBuilder<T> {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of blocks in the pool
    pub fn block_count(mut self, count: usize) -> Self {
        self.block_count = Some(count);
        self
    }

    /// Set the duplication policy
    pub fn duplication_policy(mut self, policy: BlockDuplicationPolicy) -> Self {
        self.duplication_policy = Some(policy);
        self
    }
    /// Set frequency tracker size with validation
    /// Must be a power of 2 between 2^18 and 2^24
    pub fn frequency_tracker_size(mut self, size: usize) -> Self {
        assert!(size >= (1 << 18) && size <= (1 << 24),
                "Frequency tracker size must be between 2^18 and 2^24, got: {}", size);
        assert!(size.is_power_of_two(),
                "Frequency tracker size must be a power of 2, got: {}", size);
        self.frequency_tracker_size = Some(size);
        self
    }

    /// Use simple LRU backend (capacity automatically set to block_count)
    pub fn with_lru_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::Lru);
        self
    }

    /// Use multi-level LRU backend with 4 fixed priority levels
    /// Default thresholds: [3, 8, 15] for transitions between:
    /// - Cold (0-2 hits) -> Warm (3-7 hits) -> Hot (8-14 hits) -> Very Hot (15 hits)
    pub fn with_multi_lru_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::MultiLru {
            frequency_thresholds: [3, 8, 15],
        });
        self
    }

    /// Use multi-level LRU with custom frequency thresholds
    ///
    /// # Requirements
    /// - Thresholds must be in ascending order: cold_to_warm < warm_to_hot < hot_to_very_hot
    /// - hot_to_very_hot must be <= 15 (4-bit counter maximum)
    /// - cold_to_warm must be >= 1 (to distinguish from never-accessed blocks)
    ///
    /// # Arguments
    /// * `cold_to_warm` - Minimum frequency to move from Cold to Warm level
    /// * `warm_to_hot` - Minimum frequency to move from Warm to Hot level
    /// * `hot_to_very_hot` - Minimum frequency to move from Hot to Very Hot level
    ///
    /// # Panics
    /// Panics if thresholds don't meet the requirements above
    pub fn with_multi_lru_backend_custom_thresholds(
        mut self,
        cold_to_warm: u8,
        warm_to_hot: u8,
        hot_to_very_hot: u8,
    ) -> Self {
        // Validate ascending order
        assert!(
            cold_to_warm < warm_to_hot && warm_to_hot < hot_to_very_hot,
            "Thresholds must be in ascending order: {} < {} < {} failed",
            cold_to_warm, warm_to_hot, hot_to_very_hot
        );

        // Validate maximum value (4-bit counter limit)
        assert!(
            hot_to_very_hot <= 15,
            "hot_to_very_hot threshold ({}) must be <= 15 (4-bit counter maximum)",
            hot_to_very_hot
        );

        // Additional validation: ensure reasonable gaps between levels
        assert!(
            cold_to_warm >= 1,
            "cold_to_warm threshold must be >= 1 to distinguish from zero-access blocks"
        );

        self.inactive_backend = Some(InactiveBackendConfig::MultiLru {
            frequency_thresholds: [cold_to_warm, warm_to_hot, hot_to_very_hot],
        });
        self
    }

    /// Use HashMap backend with custom reuse policy
    pub fn with_hashmap_backend(mut self, reuse_policy: Box<dyn ReusePolicy>) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::HashMap { reuse_policy });
        self
    }

    /// Validate the configuration
    fn validate(&self) -> Result<(), String> {
        let block_count = self.block_count
            .ok_or("block_count is required")?;

        if block_count == 0 {
            return Err("block_count must be greater than 0".to_string());
        }

        // Additional validation for MultiLRU thresholds at build time
        if let Some(InactiveBackendConfig::MultiLru { frequency_thresholds }) = &self.inactive_backend {
            let [t1, t2, t3] = frequency_thresholds;
            if !(*t1 < *t2 && *t2 < *t3) {
                return Err(format!(
                    "Invalid thresholds [{}, {}, {}]: must be in ascending order",
                    t1, t2, t3
                ));
            }
            if *t3 > 15 {
                return Err(format!(
                    "Invalid threshold {}: maximum frequency is 15 (4-bit counter)",
                    t3
                ));
            }
        }

        Ok(())
    }

    /// Build the BlockManager
    pub fn build(mut self) -> Result<BlockManager<T>, BlockManagerBuilderError> {
        // First validate the configuration
        self.validate()
            .map_err(BlockManagerBuilderError::ValidationError)?;

        let block_count = self.block_count.unwrap();

        // Create registry with frequency tracking
        let freq_size = self.frequency_tracker_size.unwrap_or(2_097_152);
        let frequency_tracker = Arc::new(TinyLFUTracker::new(freq_size));
        let registry = BlockRegistry::with_frequency_tracker(frequency_tracker.clone());

        // Create reset pool
        let blocks: Vec<Block<T, Reset>> = (0..block_count as u64)
            .map(|id| Block::new(id))
            .collect();
        let reset_pool = ResetPool::new(blocks);

        // Create backend based on configuration
        let backend: Box<dyn InactivePoolBackend<T>> = match self.inactive_backend.take() {
            Some(InactiveBackendConfig::HashMap { reuse_policy }) => {
                Box::new(HashMapBackend::new(reuse_policy))
            }
            Some(InactiveBackendConfig::Lru) => {
                // Capacity automatically set to block_count
                let capacity = NonZeroUsize::new(block_count)
                    .expect("block_count must be > 0");
                Box::new(LruBackend::new(capacity))
            }
            Some(InactiveBackendConfig::MultiLru { frequency_thresholds }) => {
                // Total capacity = block_count, distributed across 4 levels
                let capacity_per_level = (block_count + 3) / 4; // Round up division
                let level_capacity = NonZeroUsize::new(capacity_per_level)
                    .expect("capacity per level must be > 0");

                Box::new(MultiLruBackend::new_with_thresholds(
                    level_capacity,
                    &frequency_thresholds,
                    frequency_tracker,
                ))
            }
            None => {
                // Default to HashMap with FIFO
                Box::new(HashMapBackend::new(Box::new(FifoReusePolicy::new())))
            }
        };

        // Create pools
        let inactive_pool = InactivePool::new(backend, &reset_pool);
        let active_pool = ActivePool::new(registry.clone(), inactive_pool.return_fn());

        Ok(BlockManager {
            reset_pool,
            active_pool,
            inactive_pool,
            block_registry: registry,
            duplication_policy: self.duplication_policy.unwrap_or(BlockDuplicationPolicy::Allow),
        })
    }
}

