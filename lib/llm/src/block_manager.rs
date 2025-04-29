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

use derive_builder::Builder;
use nixl_sys::Agent as NixlAgent;
use std::sync::Arc;
use validator::Validate;

pub struct KvBlockManagerBuilder<
    DeviceMetadata: BlockMetadata,
    HostMetadata: BlockMetadata,
    // LocalStorageMetadata: BlockMetadata,
> {
    host_pool: Option<BlockPool<PinnedStorage, HostMetadata>>,
    device_pool: Option<BlockPool<DeviceStorage, DeviceMetadata>>,
    // local_storage_pool: Option<BlockPool<LocalStorageLayout, LocalStorageMetadata>>,
    nixl_agent: Option<NixlAgent>,
}

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
pub struct KvManagerBlockConfig<S: Storage + NixlEnabledStorage> {
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
impl<S: Storage> KvManagerBlockConfigBuilder<S> {
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

#[derive(Debug, Clone, Builder, Validate)]
pub struct KvManagerPoolConfig {
    model_config: KvManagerModelConfig,
}
