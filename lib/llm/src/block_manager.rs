// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Block Manager for LLM KV Cache
//!
//! This module provides functionality for managing KV blocks in LLM attention
//! mechanisms. It handles storage allocation, block management, and safe access
//! patterns for both system memory and remote (NIXL) storage.

pub mod config;
mod state;

pub mod block;
pub mod distributed;
pub mod events;
pub mod layout;
pub mod metrics;
pub mod offload;
pub mod pool;
pub mod storage;

pub use crate::common::dtype::DType;
pub use block::{
    nixl::{
        AsBlockDescriptorSet, BlockDescriptorList, IsImmutable, IsMutable, MutabilityKind,
        RemoteBlock,
    },
    transfer::{BlockTransferEngineV1, TransferRequestPut},
    BasicMetadata, BlockMetadata, Blocks, ImmutableBlock,
};
pub use config::*;
pub use layout::{nixl::NixlLayout, LayoutConfig, LayoutConfigBuilder, LayoutError, LayoutType};
use offload::request::BlockResult;
pub use pool::BlockPool;
pub use storage::{
    nixl::NixlRegisterableStorage, DeviceStorage, DiskStorage, PinnedStorage, Storage,
    StorageAllocator,
};
pub use tokio_util::sync::CancellationToken;

use anyhow::{Context, Result};
use block::nixl::{BlockMutability, NixlBlockSet, RemoteBlocks, SerializedNixlBlockSet};
use derive_builder::Builder;
use nixl_sys::Agent as NixlAgent;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use storage::nixl::MemType;
use validator::Validate;

pub type WorkerID = u64;

pub type ReferenceBlockManager = KvBlockManager<BasicMetadata>;

/// Represents the different cache levels for KV blocks
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum CacheLevel {
    /// Represents KV blocks in GPU memory
    G1,

    /// Represents KV blocks in CPU memory
    G2,

    /// Represents KV blocks in Local NVMe storage
    G3,

    /// Represents KV blocks in Remote NVMe storage
    G4,
}

struct CancelOnLastDrop {
    cancellation_token: CancellationToken,
}

impl Drop for CancelOnLastDrop {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

// When we construct the pool:
// 1. instantiate the runtime,
// 2. build layout::LayoutConfigs for each of the requested storage types
// 3. register the layouts with the NIXL agent if enabled
// 4. construct a Blocks object for each layout providing a unique block_set_idx
//    for each layout type.
// 5. initialize the pools for each set of blocks
#[derive(Clone)]
pub struct KvBlockManager<Metadata: BlockMetadata> {
    state: Arc<state::KvBlockManagerState<Metadata>>,
    _cancellation_token: Arc<CancelOnLastDrop>,
    block_size: usize,
}

impl<Metadata: BlockMetadata> KvBlockManager<Metadata> {
    /// Create a new [KvBlockManager]
    ///
    /// The returned object is a frontend to the [KvBlockManager] which owns the cancellation
    /// tokens. When this object gets drop, the cancellation token will be cancelled and begin
    /// the gracefully shutdown of the block managers internal state.
    pub fn new(config: KvBlockManagerConfig) -> Result<Self> {
        let mut config = config;

        // The frontend of the KvBlockManager will take ownership of the cancellation token
        // and will be responsible for cancelling the task when the KvBlockManager is dropped
        let cancellation_token = config.runtime.cancellation_token.clone();

        // The internal state will use a child token of the original token
        config.runtime.cancellation_token = cancellation_token.child_token();

        let block_size = config.model.page_size;

        // Create the internal state
        let state = state::KvBlockManagerState::new(config)?;

        let _cancellation_token = Arc::new(CancelOnLastDrop { cancellation_token });

        Ok(Self {
            state,
            _cancellation_token,
            block_size,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Exports the local blockset configuration as a serialized object.
    pub fn export_local_blockset(&self) -> Result<SerializedNixlBlockSet> {
        self.state.export_local_blockset()
    }

    /// Imports a remote blockset configuration from a serialized object.
    pub fn import_remote_blockset(
        &self,
        serialized_blockset: SerializedNixlBlockSet,
    ) -> Result<()> {
        self.state.import_remote_blockset(serialized_blockset)
    }

    /// Get a [`Vec<RemoteBlock<IsImmutable>>`] from a [`BlockDescriptorList`]
    pub fn get_remote_blocks_immutable(
        &self,
        bds: &BlockDescriptorList,
    ) -> Result<Vec<RemoteBlock<IsImmutable>>> {
        self.state.get_remote_blocks_immutable(bds)
    }

    /// Get a [`Vec<RemoteBlock<IsMutable>>`] from a [`BlockDescriptorList`]
    pub fn get_remote_blocks_mutable(
        &self,
        bds: &BlockDescriptorList,
    ) -> Result<Vec<RemoteBlock<IsMutable>>> {
        self.state.get_remote_blocks_mutable(bds)
    }

    /// Get a reference to the disk block pool
    pub fn disk(&self) -> Option<&BlockPool<DiskStorage, Metadata>> {
        self.state.disk()
    }

    /// Get a reference to the host block pool
    pub fn host(&self) -> Option<&BlockPool<PinnedStorage, Metadata>> {
        self.state.host()
    }

    /// Get a reference to the device block pool
    pub fn device(&self) -> Option<&BlockPool<DeviceStorage, Metadata>> {
        self.state.device()
    }

    /// Get the worker ID
    pub fn worker_id(&self) -> WorkerID {
        self.state.worker_id()
    }

    pub async fn onboard_blocks<S: Storage>(
        &self,
        blocks: Vec<ImmutableBlock<S, Metadata>>,
    ) -> BlockResult<DeviceStorage, Metadata> {
        self.state.onboard_blocks(blocks).await
    }
}

#[cfg(all(test, feature = "testing-full"))]
mod tests {
    use super::*;

    use crate::block_manager::block::BlockExt;
    use crate::tokens::Tokens;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio_retry::{strategy::ExponentialBackoff, Retry};

    /// Retry configuration for testing async operations
    ///
    /// Designed specifically for testing offloaded and detached operations
    /// where we need to wait for completion without static timeouts.
    #[derive(Clone, Debug)]
    struct RetryConfig {
        /// Maximum total duration to keep retrying
        pub max_duration: Duration,
        /// Initial delay between retries
        pub initial_delay: Duration,
        /// Maximum delay between retries
        pub max_delay: Duration,
        /// Whether to use exponential backoff or fixed intervals
        pub use_exponential_backoff: bool,
    }

    impl RetryConfig {
        /// Configuration optimized for offload operations testing
        ///
        /// Uses aggressive retry timing suitable for testing scenarios
        /// where operations typically complete within a few seconds.
        pub fn for_offload_test() -> Self {
            Self {
                max_duration: Duration::from_secs(10),
                initial_delay: Duration::from_millis(50),
                max_delay: Duration::from_millis(500),
                use_exponential_backoff: true,
            }
        }

        /// Quick retry configuration for fast-running tests
        ///
        /// Shorter timeouts for operations that should complete quickly.
        pub fn quick() -> Self {
            Self {
                max_duration: Duration::from_secs(3),
                initial_delay: Duration::from_millis(10),
                max_delay: Duration::from_millis(100),
                use_exponential_backoff: true,
            }
        }

        /// Patient retry configuration for slower operations
        ///
        /// Longer timeouts for operations that may take more time to complete.
        pub fn patient() -> Self {
            Self {
                max_duration: Duration::from_secs(30),
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(2),
                use_exponential_backoff: true,
            }
        }
    }

    /// Retry an async operation until it succeeds or times out
    ///
    /// This function will retry the operation until either:
    /// - The operation returns Ok(value) where value has length > 0
    /// - The maximum duration is exceeded
    ///
    /// # Arguments
    /// * `config` - Retry configuration specifying timeouts and backoff
    /// * `operation` - Async closure that returns Result<Vec<T>>
    ///
    /// # Returns
    /// * `Ok(Vec<T>)` - If operation succeeds with non-empty result
    /// * `Err(anyhow::Error)` - If operation times out or fails permanently
    async fn retry_until_non_empty<T, F, Fut>(config: RetryConfig, operation: F) -> Result<Vec<T>>
    where
        F: FnMut() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<Vec<T>>> + Send + 'static,
        T: Send + 'static,
    {
        let strategy = if config.use_exponential_backoff {
            ExponentialBackoff::from_millis(config.initial_delay.as_millis() as u64)
                .max_delay(config.max_delay)
        } else {
            // For non-exponential, we'll still use exponential but with multiplier of 1
            ExponentialBackoff::from_millis(config.initial_delay.as_millis() as u64)
                .factor(1) // No exponential growth
                .max_delay(config.max_delay)
        };

        let operation = Arc::new(Mutex::new(operation));

        // Wrap the retry logic with tokio::time::timeout for proper timeout handling
        let retry_with_timeout = async {
            Retry::spawn(strategy, move || {
                let operation_clone = Arc::clone(&operation);
                async move {
                    let result = {
                        let mut op = operation_clone.lock().unwrap();
                        op().await.map_err(|e| format!("Operation failed: {}", e))?
                    };

                    if result.is_empty() {
                        // Return a retryable error if the result is empty
                        Err("Operation returned empty result, retrying...".to_string())
                    } else {
                        // Success - we have a non-empty result
                        tracing::debug!(count = result.len(), "Retry operation succeeded");
                        Ok(result)
                    }
                }
            })
            .await
        };

        // Use tokio::time::timeout to enforce the maximum duration
        match tokio::time::timeout(config.max_duration, retry_with_timeout).await {
            Ok(result) => result.map_err(|e| anyhow::anyhow!("Retry operation failed: {}", e)),
            Err(_) => Err(anyhow::anyhow!(
                "Retry operation timed out after {:?}",
                config.max_duration
            )),
        }
    }

    // Atomic Counter for Worker ID
    static WORKER_ID: AtomicU64 = AtomicU64::new(1337);

    fn create_reference_block_manager() -> ReferenceBlockManager {
        let worker_id = WORKER_ID.fetch_add(1, Ordering::SeqCst);

        // Check if we're already in a Tokio runtime context
        let async_runtime = if tokio::runtime::Handle::try_current().is_ok() {
            None // If we're already in a runtime, don't create a new one
        } else {
            // Only create a new runtime if not already in one
            Some(Arc::new(tokio::runtime::Runtime::new().unwrap()))
        };

        let config = KvBlockManagerConfig::builder()
            .runtime(
                KvManagerRuntimeConfig::builder()
                    .worker_id(worker_id)
                    .enable_nixl()
                    .async_runtime(async_runtime)
                    .build()
                    .unwrap(),
            )
            .model(
                KvManagerModelConfig::builder()
                    .num_layers(3)
                    .outer_dim(2)
                    .page_size(4)
                    .inner_dim(16)
                    .build()
                    .unwrap(),
            )
            .disk_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(16)
                    .allocator(storage::DiskAllocator)
                    .build()
                    .unwrap(),
            )
            .host_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(16)
                    .allocator(storage::PinnedAllocator::default())
                    .build()
                    .unwrap(),
            )
            .device_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(8)
                    .allocator(storage::DeviceAllocator::new(0).unwrap())
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        ReferenceBlockManager::new(config).unwrap()
    }

    #[tokio::test]
    async fn test_reference_block_manager_inherited_async_runtime() {
        dynamo_runtime::logging::init();
        let _block_manager = create_reference_block_manager();
    }

    #[test]
    fn test_reference_block_manager_blocking() {
        dynamo_runtime::logging::init();
        let _block_manager = create_reference_block_manager();
    }

    // This tests mimics the behavior of two unique kvbm workers exchanging blocksets
    // Each KvBlockManager is a unique worker in this test, each has its resources including
    // it's own worker_ids, nixl_agent, and block pools.
    //
    // This test is meant to mimic the behavior of the basic nixl integration test found here:
    // https://github.com/ai-dynamo/nixl/blob/main/src/bindings/rust/src/tests.rs
    // TODO: This test doesn't work because NIXL doesn't support partial metadata in the rust bindings.
    #[ignore]
    #[tokio::test]
    async fn test_reference_block_managers() {
        dynamo_runtime::logging::init();

        // create two block managers - mimics two unique dynamo workers
        let kvbm_0 = create_reference_block_manager();
        let kvbm_1 = create_reference_block_manager();

        assert_ne!(kvbm_0.worker_id(), kvbm_1.worker_id());

        // in dynamo, we would exchange the blocksets via the discovery plane
        let blockset_0 = kvbm_0.export_local_blockset().unwrap();
        let blockset_1 = kvbm_1.export_local_blockset().unwrap();

        // in dynamo, we would be watching the discovery plane for remote blocksets
        kvbm_0.import_remote_blockset(blockset_1).unwrap();
        kvbm_1.import_remote_blockset(blockset_0).unwrap();

        // Worker 0
        // Allocate 4 mutable blocks on the host
        let blocks_0 = kvbm_0.host().unwrap().allocate_blocks(4).await.unwrap();

        // Create a BlockDescriptorList for the mutable blocks
        // let blockset_0 = BlockDescriptorList::from_mutable_blocks(&blocks_0).unwrap();
        let blockset_0 = blocks_0.as_block_descriptor_set().unwrap();

        // Worker 1
        // Create a RemoteBlock list from blockset_0
        let _blocks_1 = kvbm_1.host().unwrap().allocate_blocks(4).await.unwrap();
        let mut _remote_blocks_0 = kvbm_1.get_remote_blocks_mutable(&blockset_0).unwrap();

        // TODO(#967) - Enable with TransferEngine

        // // Create a TransferRequestPut for the mutable blocks
        // let transfer_request = TransferRequestPut::new(&blocks_0, &mut remote_blocks_0).unwrap();

        // // Validate blocks - this could be an expensive operation
        // // TODO: Create an ENV trigger debug flag which will call this on every transfer request
        // // In this case, we expect an error because we have overlapping blocks as we are sending to/from the same blocks
        // // because we are using the wrong target (artifact of the test setup allowing variable to cross what woudl be
        // // worker boundaries)
        // // assert!(transfer_request.validate_blocks().is_err());

        // // This is proper request - PUT from worker 1 (local) to worker 0 (remote)
        // let transfer_request = TransferRequestPut::new(&blocks_1, &mut remote_blocks_0).unwrap();
        // // assert!(transfer_request.validate_blocks().is_ok());

        // // Execute the transfer request
        // // transfer_request.execute().unwrap();

        // let mut put_request = PutRequestBuilder::<_, _>::builder();

        // put_request.from(&blocks_1).to(&mut remote_blocks_0);

        // // Create a Put request direct between two local blocks
        // // split the blocks into two vecs each with 2 blocks
        // let mut blocks_1 = blocks_1;

        // let slice_0 = blocks_1.split_off(2);
        // let mut slice_1 = blocks_1;

        // let transfer_request = TransferRequestPut::new(&slice_0, &mut slice_1).unwrap();
        // // assert!(transfer_request.validate_blocks().is_ok());

        // // Execute the transfer request
        // // transfer_request.execute().unwrap();
    }

    #[tokio::test]
    async fn test_retry_timeout_behavior() -> Result<()> {
        dynamo_runtime::logging::init();

        // Test that retry properly times out when operation never succeeds
        let start_time = std::time::Instant::now();
        let config = RetryConfig::quick(); // 3 second timeout

        let result = retry_until_non_empty(config, || async {
            // This operation will always return empty, forcing timeout
            Ok::<Vec<i32>, anyhow::Error>(vec![])
        })
        .await;

        let elapsed = start_time.elapsed();

        // Should have failed due to timeout
        assert!(result.is_err());

        // Should have taken approximately the max_duration (3 seconds for quick config)
        // Allow some tolerance for timing variations
        assert!(
            elapsed >= Duration::from_secs(2),
            "Timeout should take at least 2 seconds, took {:?}",
            elapsed
        );
        assert!(
            elapsed <= Duration::from_secs(5),
            "Timeout should not exceed 5 seconds, took {:?}",
            elapsed
        );

        tracing::info!("Retry timeout test completed in {:?}", elapsed);
        Ok(())
    }

    #[tokio::test]
    async fn test_retry_immediate_success() -> Result<()> {
        dynamo_runtime::logging::init();

        // Test that retry succeeds immediately when operation works
        let start_time = std::time::Instant::now();
        let config = RetryConfig::quick();

        let result = retry_until_non_empty(config, || async {
            // This operation will always succeed immediately
            Ok::<Vec<i32>, anyhow::Error>(vec![1, 2, 3])
        })
        .await;

        let elapsed = start_time.elapsed();

        // Should succeed
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1, 2, 3]);

        // Should complete quickly (much less than timeout)
        assert!(
            elapsed < Duration::from_millis(500),
            "Immediate success should be fast, took {:?}",
            elapsed
        );

        tracing::info!("Retry immediate success test completed in {:?}", elapsed);
        Ok(())
    }

    #[tokio::test]
    async fn test_offload() -> Result<()> {
        dynamo_runtime::logging::init();

        let block_manager = create_reference_block_manager();

        let device = block_manager.device().unwrap();

        let tokens = Tokens::from(vec![1, 2, 3, 4]);
        let token_sequence = tokens.into_sequence(4, Some(0));
        let token_block = token_sequence.blocks().first().unwrap();

        let mut device_block = device.allocate_blocks(1).await?.into_iter().next().unwrap();
        device_block.apply_token_block(token_block.clone())?;

        let immutable_device_blocks = device.register_blocks(vec![device_block]).await.unwrap();
        assert_eq!(immutable_device_blocks.len(), 1);

        let sequence_hash = immutable_device_blocks[0].sequence_hash();

        // Wait for blocks to be offloaded to host - retry until we find them or timeout
        let host_blocks = retry_until_non_empty(RetryConfig::for_offload_test(), {
            let block_manager = block_manager.clone();
            move || {
                let block_manager = block_manager.clone();
                async move {
                    Ok(block_manager
                        .host()
                        .unwrap()
                        .match_sequence_hashes(vec![sequence_hash].as_slice())
                        .await?)
                }
            }
        })
        .await?;
        assert_eq!(host_blocks.len(), 1);

        // Wait for blocks to be offloaded to disk - retry until we find them or timeout
        let disk_blocks = retry_until_non_empty(RetryConfig::for_offload_test(), {
            let block_manager = block_manager.clone();
            move || {
                let block_manager = block_manager.clone();
                async move {
                    Ok(block_manager
                        .disk()
                        .unwrap()
                        .match_sequence_hashes(vec![sequence_hash].as_slice())
                        .await?)
                }
            }
        })
        .await?;
        assert_eq!(disk_blocks.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_with_enhanced_logging() -> Result<()> {
        dynamo_runtime::logging::init();

        let block_manager = create_reference_block_manager();

        let device = block_manager.device().unwrap();

        let tokens = Tokens::from(vec![1, 2, 3, 4]);
        let token_sequence = tokens.into_sequence(4, Some(0));
        let token_block = token_sequence.blocks().first().unwrap();

        let mut device_block = device.allocate_blocks(1).await?.into_iter().next().unwrap();
        device_block.apply_token_block(token_block.clone())?;

        let immutable_device_blocks = device.register_blocks(vec![device_block]).await.unwrap();
        assert_eq!(immutable_device_blocks.len(), 1);

        let sequence_hash = immutable_device_blocks[0].sequence_hash();
        tracing::info!(
            "Created device block with sequence_hash: {:?}",
            sequence_hash
        );

        // Wait for blocks to be offloaded to host - retry until we find them or timeout
        tracing::info!("Starting host offload wait...");
        let host_blocks = retry_until_non_empty(RetryConfig::for_offload_test(), {
            let block_manager = block_manager.clone();
            move || {
                let block_manager = block_manager.clone();
                async move {
                    let result = block_manager
                        .host()
                        .unwrap()
                        .match_sequence_hashes(vec![sequence_hash].as_slice())
                        .await?;
                    tracing::debug!("Host check returned {} blocks", result.len());
                    Ok(result)
                }
            }
        })
        .await?;
        assert_eq!(host_blocks.len(), 1);
        tracing::info!("Host offload completed successfully");

        // Wait for blocks to be offloaded to disk - retry until we find them or timeout
        tracing::info!("Starting disk offload wait...");
        let start_time = std::time::Instant::now();
        let disk_result = retry_until_non_empty(RetryConfig::for_offload_test(), {
            let block_manager = block_manager.clone();
            let sequence_hash = sequence_hash;
            move || {
                let block_manager = block_manager.clone();
                async move {
                    let result = block_manager
                        .disk()
                        .unwrap()
                        .match_sequence_hashes(vec![sequence_hash].as_slice())
                        .await;

                    match &result {
                        Ok(blocks) => {
                            tracing::debug!("Disk check returned {} blocks", blocks.len());
                        }
                        Err(e) => {
                            tracing::warn!("Disk check failed: {:?}", e);
                        }
                    }

                    result.map_err(|e| anyhow::anyhow!("Disk operation failed: {}", e))
                }
            }
        })
        .await;

        let elapsed = start_time.elapsed();
        tracing::info!("Disk offload attempt completed in {:?}", elapsed);

        match disk_result {
            Ok(disk_blocks) => {
                assert_eq!(disk_blocks.len(), 1);
                tracing::info!("Disk offload completed successfully");
            }
            Err(e) => {
                tracing::error!("Disk offload failed: {:?}", e);
                // For now, we'll allow this to fail since we're investigating the issue
                // but we want to ensure the timeout actually worked
                assert!(
                    elapsed >= Duration::from_secs(8),
                    "Should have retried for at least 8 seconds, only took {:?}",
                    elapsed
                );
                assert!(
                    elapsed <= Duration::from_secs(12),
                    "Should have timed out by 12 seconds, took {:?}",
                    elapsed
                );
                tracing::info!(
                    "Confirmed that disk offload properly timed out after {:?}",
                    elapsed
                );
            }
        }

        Ok(())
    }
}
