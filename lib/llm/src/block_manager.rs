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

pub mod block;
pub mod events;
pub mod layout;
pub mod pool;
pub mod storage;

pub use crate::common::dtype::DType;
pub use block::{BasicMetadata, BlockMetadata, Blocks};
pub use layout::{nixl::NixlLayout, LayoutConfig, LayoutConfigBuilder, LayoutError, LayoutType};
pub use pool::BlockPool;
pub use storage::{
    nixl::NixlEnabledStorage, DeviceStorage, PinnedStorage, Storage, StorageAllocator,
};
pub use tokio_util::sync::CancellationToken;

use anyhow::{Context, Result};
use block::nixl::{NixlBlockSet, RemoteBlocks, SerializedNixlBlockSet};
use derive_builder::Builder;
use nixl_sys::Agent as NixlAgent;
use std::{collections::HashMap, sync::Arc};
use storage::nixl::MemType;
use validator::Validate;

pub type WorkerID = u64;

#[derive(Debug, Clone, Builder, Validate)]
#[builder(pattern = "owned")]
pub struct KvManagerRuntimeConfig {
    worker_id: u64,

    #[builder(default)]
    cancellation_token: CancellationToken,

    #[builder(default = "true")]
    enable_nixl: bool,
}

impl KvManagerRuntimeConfig {
    pub fn builder() -> KvManagerRuntimeConfigBuilder {
        KvManagerRuntimeConfigBuilder::default()
    }
}

#[derive(Debug, Clone, Builder, Validate)]
#[builder(pattern = "owned")]
pub struct KvManagerModelConfig {
    #[validate(range(min = 1))]
    pub num_layers: usize,

    #[validate(range(min = 1))]
    pub page_size: usize,

    #[validate(range(min = 1))]
    pub inner_dim: usize,

    #[builder(default = "DType::FP16")]
    pub dtype: DType,
}

impl KvManagerModelConfig {
    pub fn builder() -> KvManagerModelConfigBuilder {
        KvManagerModelConfigBuilder::default()
    }
}

#[derive(Builder, Validate)]
#[builder(pattern = "owned", build_fn(validate = "Self::validate"))]
pub struct KvManagerLayoutConfig<S: Storage + NixlEnabledStorage> {
    #[validate(range(min = 1))]
    pub num_blocks: usize,

    #[builder(default = "LayoutType::FullyContiguous")]
    pub layout_type: LayoutType,

    /// Storage for the blocks
    /// If provided, the blocks will be allocated from the provided storage
    /// Otherwise, the blocks will be allocated from
    #[builder(default)]
    pub storage: Option<Vec<S>>,

    /// If provided, the blocks will be allocated from the provided allocator
    /// This option is mutually exclusive with the `storage` option
    #[builder(default, setter(custom))]
    pub allocator: Option<Arc<dyn StorageAllocator<S>>>,
}

impl<S: Storage + NixlEnabledStorage> KvManagerLayoutConfig<S> {
    pub fn builder() -> KvManagerLayoutConfigBuilder<S> {
        KvManagerLayoutConfigBuilder::default()
    }
}

// Implement the validation and build functions on the generated builder type
// Note: derive_builder generates KvManagerBlockConfigBuilder<S>
impl<S: Storage + NixlEnabledStorage> KvManagerLayoutConfigBuilder<S> {
    /// Custom setter for the `allocator` field
    fn allocator(mut self, allocator: impl StorageAllocator<S> + 'static) -> Self {
        self.allocator = Some(Some(Arc::new(allocator)));
        self
    }

    // Validation function
    fn validate(&self) -> Result<(), String> {
        match (self.storage.is_some(), self.allocator.is_some()) {
            (true, false) | (false, true) => Ok(()), // XOR condition met
            (true, true) => Err("Cannot provide both `storage` and `allocator`.".to_string()),
            (false, false) => Err("Must provide either `storage` or `allocator`.".to_string()),
        }
    }
}

#[derive(Builder, Validate)]
#[builder(pattern = "owned")]
pub struct KvBlockManagerConfig {
    runtime: KvManagerRuntimeConfig,
    model: KvManagerModelConfig,

    #[builder(default, setter(strip_option))]
    device_layout: Option<KvManagerLayoutConfig<DeviceStorage>>,

    #[builder(default, setter(strip_option))]
    host_layout: Option<KvManagerLayoutConfig<PinnedStorage>>,
}

impl KvBlockManagerConfig {
    pub fn builder() -> KvBlockManagerConfigBuilder {
        KvBlockManagerConfigBuilder::default()
    }
}

pub type ReferenceBlockManager = KvBlockManager<BasicMetadata, BasicMetadata>;

// When we construct the pool:
// 1. instantiate the runtime,
// 2. build layout::LayoutConfigs for each of the requested storage types
// 3. register the layouts with the NIXL agent if enabled
// 4. construct a Blocks object for each layout providing a unique block_set_idx
//    for each layout type.
// 5. initialize the pools for each set of blocks
pub struct KvBlockManager<HostMetadata: BlockMetadata, DeviceMetadata: BlockMetadata> {
    worker_id: WorkerID,
    cancellation_token: CancellationToken,

    nixl_agent: Option<NixlAgent>,
    nixl_backends: HashMap<String, nixl_sys::Backend>,

    host_pool: Option<BlockPool<PinnedStorage, HostMetadata>>,
    device_pool: Option<BlockPool<DeviceStorage, DeviceMetadata>>,

    local_block_set: NixlBlockSet,
    remote_block_sets: HashMap<WorkerID, HashMap<usize, RemoteBlocks>>,
}

impl<HostMetadata: BlockMetadata, DeviceMetadata: BlockMetadata>
    KvBlockManager<HostMetadata, DeviceMetadata>
{
    pub fn new(config: KvBlockManagerConfig) -> Result<Self> {
        config
            .runtime
            .validate()
            .context("Validating runtime config")?;

        config.model.validate().context("Validating model config")?;

        let worker_id = config.runtime.worker_id;
        let cancellation_token = config.runtime.cancellation_token;

        // Create a map of NIXL backends
        let mut nixl_backends: HashMap<String, nixl_sys::Backend> = HashMap::new();

        // Create a NIXL agent if NIXL is enabled and instantiate requested backends
        // TODO: Build a map of NIXL backends to block pools/sets
        let nixl_agent = if config.runtime.enable_nixl {
            let agent = NixlAgent::new(&worker_id.to_string())?;

            // Create NIXL nixl_backends
            // TODO: Expose this to API for configuration
            tracing::debug!("Creating NIXL UCX backend");
            let (_ucx_mem_list1, ucx_params) = agent.get_plugin_params("UCX")?;
            let backend = agent.create_backend("UCX", &ucx_params)?;
            nixl_backends.insert("UCX".to_string(), backend);

            Some(agent)
        } else {
            None
        };

        // Initialize model-specific layout config. The layout_builder is incomplete at this point.
        // We will clone this builder and apply the storage-specific configs to each clone in the
        // following steps.
        let model = &config.model;
        let mut layout_builder = LayoutConfig::builder();

        layout_builder
            .num_layers(model.num_layers)
            .page_size(model.page_size)
            .inner_dim(model.inner_dim)
            .dtype(model.dtype);

        let mut next_block_set_idx = 0;
        let mut local_block_set = block::nixl::NixlBlockSet::new(worker_id);

        // Create the host block pool if a host layout is provided
        let host_pool = if let Some(config) = config.host_layout {
            next_block_set_idx += 1;
            tracing::debug!("Constructing host pool.");
            let layout = create_layout(layout_builder.clone(), config, nixl_agent.as_ref())?;
            local_block_set.add_block_set(next_block_set_idx, layout.serialize()?);
            let block_pool = create_block_pool::<_, HostMetadata>(
                layout,
                next_block_set_idx,
                cancellation_token.clone(),
            )?;
            Some(block_pool)
        } else {
            tracing::debug!("No host layout provided; will not allocate host blocks.");
            None
        };

        // Create the device block pool if a device layout is provided
        let device_pool = if let Some(config) = config.device_layout {
            next_block_set_idx += 1;
            tracing::debug!("Constructing device pool.");
            let layout = create_layout(layout_builder.clone(), config, nixl_agent.as_ref())?;
            local_block_set.add_block_set(next_block_set_idx, layout.serialize()?);
            let block_pool = create_block_pool::<_, DeviceMetadata>(
                layout,
                next_block_set_idx,
                cancellation_token.clone(),
            )?;
            Some(block_pool)
        } else {
            tracing::debug!("No device layout provided; will not allocate device blocks.");
            None
        };

        // Finalize the local block set by adding NIXL metadata
        if let Some(nixl_agent) = &nixl_agent {
            tracing::debug!("Finalize NixlBlockSet: adding NIXL metadata.");
            local_block_set.set_nixl_metadata(nixl_agent.get_local_md()?);
        }

        Ok(Self {
            worker_id,
            cancellation_token,
            nixl_agent,
            nixl_backends,
            host_pool,
            device_pool,
            local_block_set,
            remote_block_sets: HashMap::new(),
        })
    }

    /// Exports the local blockset configuration as a serialized object.
    pub fn export_local_blockset(&self) -> Result<SerializedNixlBlockSet> {
        SerializedNixlBlockSet::try_from(&self.local_block_set)
            .context("Failed to serialize local blockset")
    }

    /// Imports a remote blockset configuration from a serialized object.
    pub fn import_remote_blockset(
        &mut self,
        serialized_blockset: SerializedNixlBlockSet,
    ) -> Result<()> {
        let remote = NixlBlockSet::try_from(serialized_blockset)
            .context("Failed to deserialize remote blockset")?;

        let (block_sets, metadata, worker_id) = remote.dissolve();
        tracing::debug!("Importing remote blockset from worker {}", worker_id);

        assert_ne!(
            worker_id, self.worker_id,
            "Cannot import blockset from self"
        );

        let agent = self
            .nixl_agent
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("NIXL agent not initialized"))?;

        if self.remote_block_sets.contains_key(&worker_id) {
            anyhow::bail!(
                "Worker ID {} already exists; cannot update remote blockset",
                worker_id
            );
        }

        let mut inner_map = HashMap::new();

        for (block_set_idx, block_set_layout) in block_sets {
            // Deserialize the individual layout and create RemoteBlocks
            let remote_blocks =
                RemoteBlocks::from_serialized(block_set_layout.clone(), block_set_idx)?;

            // check the storage type of the remote blocks
            let layout = remote_blocks.layout();
            let storage = layout.storage();

            let storage = storage
                .first()
                .ok_or_else(|| anyhow::anyhow!("No storage found in remote blockset"))?;

            match storage.mem_type() {
                MemType::Dram => {
                    tracing::trace!(block_set_idx, "Detected Host/DRAM remote descriptor");
                }
                MemType::Vram => {
                    tracing::trace!(block_set_idx, "Detected GPU/Device/VRAM remote descriptor");
                }
                _ => {
                    tracing::warn!(
                        block_set_idx,
                        "Detected unknown remote descriptor; skipping blockset..."
                    );
                    continue;
                }
            }

            inner_map.insert(block_set_idx, remote_blocks);
        }

        let agent_id = agent
            .load_remote_md(&metadata)
            .context("Loading remote metadata")?;

        // try to convert the agent_id (String) to a WorkerID (u64)
        let agent_id: WorkerID =
            agent_id // Assuming agent_id is String here
                .parse() // Parse the String into u64 (WorkerID)
                .context("Failed to parse agent ID string into WorkerID (u64)")?;

        assert_eq!(agent_id, worker_id, "Mismatch with remote worker ID");

        self.remote_block_sets.insert(worker_id, inner_map);

        Ok(())
    }
}

impl<HostMetadata: BlockMetadata, DeviceMetadata: BlockMetadata>
    KvBlockManager<HostMetadata, DeviceMetadata>
{
    pub fn host(&self) -> Option<&BlockPool<PinnedStorage, HostMetadata>> {
        self.host_pool.as_ref()
    }

    pub fn device(&self) -> Option<&BlockPool<DeviceStorage, DeviceMetadata>> {
        self.device_pool.as_ref()
    }
}

impl<HostMetadata: BlockMetadata, DeviceMetadata: BlockMetadata> Drop
    for KvBlockManager<HostMetadata, DeviceMetadata>
{
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

fn create_layout<S: Storage + NixlEnabledStorage>(
    mut builder: LayoutConfigBuilder,
    config: KvManagerLayoutConfig<S>,
    nixl_agent: Option<&NixlAgent>,
) -> Result<Arc<dyn NixlLayout<StorageType = S>>> {
    let layout = builder.num_blocks(config.num_blocks).build()?;
    if let Some(storage) = config.storage {
        let mut layout = layout.create_layout(config.layout_type, storage)?;
        if let Some(nixl_agent) = nixl_agent {
            layout.nixl_register(nixl_agent, None)?;
        }
        return Ok(Arc::new(layout));
    }

    if let Some(allocator) = config.allocator {
        let mut layout = layout.allocate_layout(config.layout_type, allocator)?;
        if let Some(nixl_agent) = nixl_agent {
            layout.nixl_register(nixl_agent, None)?;
        }
        return Ok(Arc::new(layout));
    }

    anyhow::bail!("failed to create layout");
}

fn create_block_pool<S: Storage + NixlEnabledStorage, M: BlockMetadata>(
    layout: Arc<dyn NixlLayout<StorageType = S>>,
    block_set_idx: usize,
    cancellation_token: CancellationToken,
) -> Result<BlockPool<S, M>> {
    let blocks = block::layout_to_blocks::<_, M>(layout, block_set_idx)?;
    Ok(BlockPool::builder()
        .blocks(blocks)
        .cancel_token(cancellation_token)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};

    // Atomic Counter for Worker ID
    static WORKER_ID: AtomicU64 = AtomicU64::new(1337);

    fn create_reference_block_manager() -> ReferenceBlockManager {
        let worker_id = WORKER_ID.fetch_add(1, Ordering::SeqCst);
        let config = KvBlockManagerConfig::builder()
            .runtime(
                KvManagerRuntimeConfig::builder()
                    .worker_id(worker_id)
                    .build()
                    .unwrap(),
            )
            .model(
                KvManagerModelConfig::builder()
                    .num_layers(3)
                    .page_size(4)
                    .inner_dim(16)
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

    // This tests mimics the behavior of two unique dynamo workers exchanging blocksets
    // Each KvBlockManager is a unique worker in this test, each has its resources including
    // it's own worker_ids, nixl_agent, and block pools.
    //
    // This test is meant to mimic the behavior of the basic nixl integration test found here:
    // https://github.com/ai-dynamo/nixl/blob/main/src/bindings/rust/src/tests.rs
    #[tokio::test]
    async fn test_reference_block_managers() {
        dynamo_runtime::logging::init();

        // create two block managers - mimics two unique dynamo workers
        let mut kvbm_0 = create_reference_block_manager();
        let mut kvbm_1 = create_reference_block_manager();

        assert_ne!(kvbm_0.worker_id, kvbm_1.worker_id);

        // in dynamo, we would exchange the blocksets via the discovery plane
        let blockset_0 = kvbm_0.export_local_blockset().unwrap();
        let blockset_1 = kvbm_1.export_local_blockset().unwrap();

        // in dynamo, we would be watching the discovery plane for remote blocksets
        kvbm_0.import_remote_blockset(blockset_1).unwrap();
        kvbm_1.import_remote_blockset(blockset_0).unwrap();
    }
}
