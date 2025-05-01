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

// Silence warnings about deprecated features (like pyo3::IntoPy::into_py)
#![allow(deprecated)]

use super::*;

use pyo3::{exceptions::PyRuntimeError, Python, PyObject, PyResult};
use dlpark::{ffi::DeviceType, prelude::{Device, DataType, ManagerCtx, ShapeAndStrides, ToTensor}};
use std::{collections::HashMap, sync::{Arc, Mutex, Weak}};

// Create a shared memory tracker type
type RegionRefTracker = Arc<Mutex<HashMap<u64, usize>>>;

struct DlPackTensor {
    memory_region: u64,
    region_shape: Vec<usize>,
    dtype: dynamo_kv_manager::dtype::DType,
    storage_type: dynamo_kv_manager::storage::StorageType,
    region_ref_tracker: Weak<Mutex<HashMap<u64, usize>>>,
}

impl ToTensor for DlPackTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.memory_region as *mut std::ffi::c_void
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        match self.storage_type {
            dynamo_kv_manager::storage::StorageType::Device(_) => Device::cuda(0),
            dynamo_kv_manager::storage::StorageType::Pinned => Device {
                device_type: DeviceType::CudaHost,
                device_id: 0,
            },
            dynamo_kv_manager::storage::StorageType::System => Device::CPU,
            _ => panic!("Unsupported storage type"),
        }
    }

    fn dtype(&self) -> DataType {
        match self.dtype {
            dynamo_kv_manager::dtype::DType::FP8 => DataType::U8,
            dynamo_kv_manager::dtype::DType::FP16 => DataType::F16,
            dynamo_kv_manager::dtype::DType::BF16 => DataType::BF16,
            dynamo_kv_manager::dtype::DType::FP32 => DataType::F32,
            dynamo_kv_manager::dtype::DType::FP64 => DataType::F64,
            dynamo_kv_manager::dtype::DType::U8 => DataType::U8,
            dynamo_kv_manager::dtype::DType::U16 => DataType::U16,
            dynamo_kv_manager::dtype::DType::U32 => DataType::U32,
            dynamo_kv_manager::dtype::DType::U64 => DataType::U64,
            dynamo_kv_manager::dtype::DType::I8 => DataType::I8,
            dynamo_kv_manager::dtype::DType::I16 => DataType::I16,
            dynamo_kv_manager::dtype::DType::I32 => DataType::I32,
            dynamo_kv_manager::dtype::DType::I64 => DataType::I64,
        }
    }

    fn shape_and_strides(&self) -> ShapeAndStrides {
        let shape_i64: Vec<i64> = self.region_shape.iter().map(|x| *x as i64).collect();
        ShapeAndStrides::new_contiguous(&shape_i64)
    }
}

impl Drop for DlPackTensor {
    fn drop(&mut self) {
        // Decrement the reference count for this memory region
        if let Some(tracker) = self.region_ref_tracker.upgrade() {
            let mut tracker_map = tracker.lock().unwrap();
            if let Some(count) = tracker_map.get_mut(&self.memory_region) {
                *count -= 1;
                if *count == 0 {
                    // Remove the entry if the count reaches zero
                    tracker_map.remove(&self.memory_region);
                    println!("Removed memory_region {:?} from tracking map", self.memory_region);
                } else {
                    println!("Decremented memory_region {:?} count to {}", self.memory_region, *count);
                }
            } else {
                panic!("Failed to get count for memory_region {:?}", self.memory_region);
            }
        } else {
            panic!("Failed to upgrade weak reference to RegionRefTracker");
        }
    }
}

#[pyclass]
pub struct BlockManager {
    inner: Arc<dyn dynamo_kv_manager::layout::BlockLayout + Send + Sync>,
    region_ref_tracker: RegionRefTracker,
}

#[pymethods]
impl BlockManager {
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

        // Create allocator based on device type
        let allocator: Box<dyn dynamo_kv_manager::storage::StorageAllocator + Send + Sync> = match device.as_str() {
            d if d.to_uppercase() == "CPU" => {
                // Use System Allocator
                Box::new(dynamo_kv_manager::storage::SystemAllocator {})
            }
            d if d.to_uppercase().starts_with("CUDA") => {
                if pin_memory {
                    // Use PinnedAllocator
                    Box::new(dynamo_kv_manager::storage::PinnedAllocator {})
                } else {
                    // Extract CUDA device ID
                    let device_id = d.split(':')
                        .nth(1)
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(0);

                    // Use DeviceAllocator
                    Box::new(dynamo_kv_manager::storage::DeviceAllocator::try_new(device_id)
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create DeviceAllocator for device {}: {}", device_id, e)))?)
                }
            }
            _ => {
                // Unsupported device
                return Err(PyRuntimeError::new_err(format!("Unsupported device: {}", device)));
            }
        };

        // Create layout with the selected allocator
        let block_layout = dynamo_kv_manager::layout::contiguous::FullyContiguous::allocate(layout_config, &*allocator)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate storage layout: {}", e)))?;

        Ok(BlockManager {
            inner: Arc::from(block_layout),
            region_ref_tracker: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn py_capsule(&self, block_idx: usize, layer_idx: usize) -> PyResult<PyObject> {
        // Get memory region and metadata for tensor construction
        let memory_region = self.inner.get_memory_region(block_idx, layer_idx)
            .expect(&format!("Failed to get memory region for block {}, layer {}", block_idx, layer_idx));
        let region_shape = vec![self.inner.page_size(), self.inner.inner_dim()];
        let dtype = self.inner.dtype();
        let storage_type = self.inner.storage_type();

        // Update the memory region reference tracker
        {
            let mut tracker = self.region_ref_tracker.lock().unwrap();
            let count = tracker.entry(memory_region).or_insert(0);
            *count += 1;
            println!("Tracking memory_region {:?}: count = {}", memory_region, *count);
        }

        // Create DLPack PyCapsule
        let manager_ctx = ManagerCtx::new(DlPackTensor {
            memory_region,
            region_shape,
            dtype,
            storage_type,
            region_ref_tracker: Arc::downgrade(&self.region_ref_tracker),
        });
        let py_capsule = Python::with_gil(|py| {
            manager_ctx.into_py(py)
        });
        Ok(py_capsule)
    }
}

impl Drop for BlockManager {
    fn drop(&mut self) {
        // Check if there are any memory regions still being tracked
        let tracker = self.region_ref_tracker.lock().unwrap();
        if !tracker.is_empty() {
            let regions: Vec<_> = tracker.iter().collect();
            panic!("BlockManager dropped while still tracking memory regions: {:?}", regions);
        }
    }
}
