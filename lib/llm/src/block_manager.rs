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
pub use block::{BasicMetadata, BlockMetadata};
pub use layout::{LayoutConfig, LayoutType};
pub use pool::BlockPool;
pub use storage::{
    nixl::NixlEnabledStorage, DeviceStorage, PinnedStorage, Storage, StorageAllocator,
};
pub use tokio_util::sync::CancellationToken;

use anyhow::{Context, Result};
use derive_builder::Builder;
use nixl_sys::Agent as NixlAgent;
use std::sync::Arc;
use validator::Validate;

#[derive(Debug, Clone, Builder)]
pub struct KvManagerRuntimeConfig {
    worker_id: u64,

    #[builder(default)]
    cancellation_token: CancellationToken,

    #[builder(default = "true")]
    enable_nixl: bool,
}

#[derive(Debug, Clone, Builder, Validate)]
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

#[derive(Clone, Builder, Validate)]
#[builder(build_fn(validate = "Self::validate"))]
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

// Implement the validation and build functions on the generated builder type
// Note: derive_builder generates KvManagerBlockConfigBuilder<S>
impl<S: Storage + NixlEnabledStorage> KvManagerLayoutConfigBuilder<S> {
    /// Custom setter for the `allocator` field
    fn allocator(&mut self, allocator: impl StorageAllocator<S> + 'static) -> &mut Self {
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
pub struct KvManagerPoolConfig {
    runtime: KvManagerRuntimeConfig,
    model: KvManagerModelConfig,

    #[builder(default, setter(strip_option))]
    device_layout: Option<KvManagerLayoutConfig<DeviceStorage>>,

    #[builder(default, setter(strip_option))]
    host_layout: Option<KvManagerLayoutConfig<PinnedStorage>>,
}

// When we construct the pool:
// 1. instantiate the runtime,
// 2. build layout::LayoutConfigs for each of the requested storage types
// 3. register the layouts with the NIXL agent if enabled
// 4. construct a Blocks object for each layout providing a unique block_set_idx
//    for each layout type.
// 5. initialize the pools for each set of blocks
pub struct KvBlockManager<HostMetadata: BlockMetadata, DeviceMetadata: BlockMetadata> {
    worker_id: u64,
    cancellation_token: CancellationToken,

    nixl_agent: Option<NixlAgent>,

    host_blocks: BlockPool<PinnedStorage, HostMetadata>,
    device_blocks: BlockPool<DeviceStorage, DeviceMetadata>,
}

// impl<HostMetadata: BlockMetadata, DeviceMetadata: BlockMetadata>
//     KvBlockManager<HostMetadata, DeviceMetadata>
// {
//     pub fn new(config: KvManagerPoolConfig) -> Result<Self> {
//         config
//             .runtime
//             .validate()
//             .context("Validating runtime config")?;

//         config.model.validate().context("Validating model config")?;

//         let worker_id = config.runtime.worker_id;
//         let cancellation_token = config.runtime.cancellation_token;

//         let nixl_agent = if config.runtime.enable_nixl {
//             Some(NixlAgent::new(&worker_id.to_string())?)
//         } else {
//             None
//         };

//         let model = &config.model;
//         let mut layout_builder = LayoutConfig::builder();

//         layout_builder
//             .num_layers(model.num_layers)
//             .page_size(model.page_size)
//             .inner_dim(model.inner_dim)
//             .dtype(model.dtype);

//         if let Some(host_layout) = &config.host_layout {
//             let layout = layout_builder
//                 .clone()
//                 .num_blocks(host_layout.num_blocks)
//                 .build()?;

//             if let Some(storage) = &host_layout.storage {
//             }
//         }

//         let host_layout = layout_builder
//             .clone()
//             .num_blocks(config.host_layout.num_blocks);

//         if let Some(host_layout) = &config.host_layout {
//             host_builder
//                 .num_blocks(host_layout.num_blocks)
//                 .page_size(host_layout.page_size)
//                 .inner_dim(host_layout.inner_dim)
//         }
//     }
// }
