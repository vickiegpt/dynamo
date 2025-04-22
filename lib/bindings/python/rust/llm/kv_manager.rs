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
    // Using Arc<dyn BlockLayout> to hold any type that implements BlockLayout
    inner: Arc<dyn dynamo_kv_manager::layout::BlockLayout + Send + Sync>,
}

#[pymethods]
impl KvManager {
    #[new]
    #[pyo3(signature = (device, pin_memory=false))]
    fn new(device: String, pin_memory: bool) -> PyResult<Self> {
        // TODO: Should this be provided by the user? The values are from contiguous.rs tests.
        let layout_config = dynamo_kv_manager::layout::LayoutConfig {
            num_blocks: 7,
            num_layers: 5,
            page_size: 4,
            inner_dim: 13,
            alignment: 1,
            dtype: dynamo_kv_manager::dtype::DType::FP32,
        };

        let inner = match device.as_str() {
            d if d == "CUDA_PINNED" => {  // TODO: Use the pin_memory flag with device arg, this should be CPU memory.
                // Use PinnedAllocator for CPU
                let allocator = dynamo_kv_manager::storage::PinnedAllocator::try_new(0)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create PinnedAllocator: {}", e)))?;

                // Create the concrete layout with PinnedStorage
                let layout = dynamo_kv_manager::layout::contiguous::FullyContiguous::allocate(layout_config, &allocator)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate PinnedStorage layout: {}", e)))?;

                // Return the layout as a trait object
                Arc::new(layout) as Arc<dyn dynamo_kv_manager::layout::BlockLayout + Send + Sync>
            }
            d if d.starts_with("CUDA") => {
                // Extract device ID for CUDA
                let device_id = d.split(':')
                    .nth(1)
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);

                // Use DeviceAllocator for CUDA
                let allocator = dynamo_kv_manager::storage::DeviceAllocator::try_new(device_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DeviceAllocator for device {}: {}", device_id, e)))?;

                // Create the concrete layout with DeviceStorage
                let layout = dynamo_kv_manager::layout::contiguous::FullyContiguous::allocate(layout_config, &allocator)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate DeviceStorage layout: {}", e)))?;

                // Return the layout as a trait object
                Arc::new(layout) as Arc<dyn dynamo_kv_manager::layout::BlockLayout + Send + Sync>
            }
            _ => {
                // Default case - return error for unsupported device
                return Err(PyRuntimeError::new_err(format!("Unsupported device: {}", device)));
            }
        };

        Ok(KvManager { inner })
    }
}
