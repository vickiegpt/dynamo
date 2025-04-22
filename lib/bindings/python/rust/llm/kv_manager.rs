// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use super::*;

use pyo3::exceptions::PyRuntimeError;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[pyclass]
pub struct KvManager {
    inner: Arc<dyn dynamo_kv_manager::layout::BlockLayout + Send + Sync>,
}

#[pymethods]
impl KvManager {
    #[new]
    #[pyo3(signature = (num_blocks, num_layers, page_size, inner_dim, alignment, dtype, device, pin_memory=false))]
    fn new(
        num_blocks: usize,
        num_layers: usize,
        page_size: usize,
        inner_dim: usize,
        alignment: usize,
        dtype: String,
        device: String,
        pin_memory: bool
    ) -> PyResult<Self> {
        let layout_config = dynamo_kv_manager::layout::LayoutConfig {
            num_blocks,
            num_layers,
            page_size,
            inner_dim,
            alignment,
            dtype: dynamo_kv_manager::dtype::DType::from_str(&dtype)
                .ok_or_else(|| PyRuntimeError::new_err(format!("Invalid dtype: {}", dtype)))?,
        };

        let block_layout = match device.as_str() {
            d if d.to_uppercase() == "CPU" => {
                // Use System Allocator
                let allocator = dynamo_kv_manager::storage::SystemAllocator {};

                let layout = dynamo_kv_manager::layout::contiguous::FullyContiguous::allocate(layout_config, &allocator)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate SystemStorage layout: {}", e)))?;

                Arc::new(layout) as Arc<dyn dynamo_kv_manager::layout::BlockLayout + Send + Sync>
            }
            d if d.to_uppercase().starts_with("CUDA") => {
                if pin_memory {
                    // Use PinnedAllocator
                    let allocator = dynamo_kv_manager::storage::PinnedAllocator {};

                    let layout = dynamo_kv_manager::layout::contiguous::FullyContiguous::allocate(layout_config, &allocator)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate PinnedStorage layout: {}", e)))?;

                    Arc::new(layout) as Arc<dyn dynamo_kv_manager::layout::BlockLayout + Send + Sync>
                } else {
                    // Extract CUDA device ID
                    let device_id = d.split(':')
                    .nth(1)
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);

                    // Use DeviceAllocator
                    let allocator = dynamo_kv_manager::storage::DeviceAllocator::try_new(device_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DeviceAllocator for device {}: {}", device_id, e)))?;

                    let layout = dynamo_kv_manager::layout::contiguous::FullyContiguous::allocate(layout_config, &allocator)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate DeviceStorage layout: {}", e)))?;

                    Arc::new(layout) as Arc<dyn dynamo_kv_manager::layout::BlockLayout + Send + Sync>
                }
            }
            _ => {
                // Unsupported device
                return Err(PyRuntimeError::new_err(format!("Unsupported device: {}", device)));
            }
        };

        Ok(KvManager { inner: block_layout })
    }
}
