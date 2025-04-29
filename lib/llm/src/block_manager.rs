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

// pub use block::{Block, BlockMetadata};
// pub use layout::BlockLayout;
// pub use pool::BlockPool;

// pub struct KvBlockManager<
//     DeviceLayout: BlockLayout,
//     DeviceMetadata: BlockMetadata,
//     HostLayout: BlockLayout,
//     HostMetadata: BlockMetadata,
//     LocalStorageLayout: BlockLayout,
//     LocalStorageMetadata: BlockMetadata,
// > {
//     device_pool: BlockPool<DeviceLayout, DeviceMetadata>,
//     host_pool: BlockPool<HostLayout, HostMetadata>,
//     local_storage_pool: BlockPool<LocalStorageLayout, LocalStorageMetadata>,
// }
