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
    inner: Arc<dynamo_kv_manager::layout::LayerConfiguration>,
}

#[pymethods]
impl KvManager {
    #[new]
    #[pyo3(signature = (device))]
    fn new(device: String) -> PyResult<Self> {
        let layout_config = dynamo_kv_manager::layout::LayoutConfig {
            num_blocks: 7,
            num_layers: 5,
            page_size: 4,
            inner_dim: 13,
            alignment: 1,
            dtype: dynamo_kv_manager::dtype::DType::FP32,
        };

        let layer_configuration = match device.as_str() {
            d if d == "CPU" => {
                // Create a PinnedAllocator for CPU
                let allocator = dynamo_kv_manager::storage::cuda::PinnedAllocator::try_new(0)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create PinnedAllocator: {}", e)))?;

                // Use allocate function instead of new
                dynamo_kv_manager::layout::contiguous::FullyContiguous::allocate(layout_config, &allocator)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate FullyContiguous layout: {}", e)))?
            }
            d if d.starts_with("CUDA") => {
                // Extract device ID if in format "CUDA:N"
                let device_id = if d.contains(':') {
                    // Split by ':' and try to parse the second part as usize
                    let parts: Vec<&str> = d.split(':').collect();
                    if parts.len() >= 2 {
                        // Try to parse the second part as usize
                        match parts[1].parse::<usize>() {
                            Ok(id) => id,
                            Err(_) => 0, // Default to 0 if parsing fails
                        }
                    } else {
                        0 // Default to 0 if no second part
                    }
                } else {
                    0 // Default to 0 if no ':' in the string
                };

                // Create a DeviceAllocator for CUDA with the specific device ID
                let allocator = dynamo_kv_manager::storage::cuda::DeviceAllocator::try_new(device_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DeviceAllocator for device {}: {}", device_id, e)))?;

                // Use allocate function with the device allocator
                dynamo_kv_manager::layout::contiguous::FullyContiguous::allocate(layout_config, &allocator)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate FullyContiguous layout on CUDA device {}: {}", device_id, e)))?
            }
            _ => {
                // Default case - return error for unsupported device
                return Err(PyRuntimeError::new_err(format!("Unsupported device: {}", device)));
            }
        };

        Ok(KvManager {
            inner: Arc::new(layer_configuration),
        })
    }
}
