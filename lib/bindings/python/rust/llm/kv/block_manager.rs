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
use crate::dlpack::{ffi, ManagerCtx, ShapeAndStrides, ToTensor};
use std::sync::Arc;

struct DlPackTensor {
    memory_region: u64,
    region_shape: Vec<usize>,
    dtype: dynamo_kv_manager::dtype::DType,
    storage_type: dynamo_kv_manager::storage::StorageType,
}

impl ToTensor for DlPackTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.memory_region as *mut std::ffi::c_void
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> ffi::Device {
        match self.storage_type {
            dynamo_kv_manager::storage::StorageType::Device(_) => ffi::Device {
                device_type: ffi::DeviceType::Cuda,
                device_id: 0,
            },
            dynamo_kv_manager::storage::StorageType::Pinned => ffi::Device {
                device_type: ffi::DeviceType::CudaHost,
                device_id: 0,
            },
            dynamo_kv_manager::storage::StorageType::System => ffi::Device {
                device_type: ffi::DeviceType::Cpu,
                device_id: 0,
            },
            _ => panic!("Unsupported storage type"),
        }
    }

    fn dtype(&self) -> ffi::DataType {
        match self.dtype {
            dynamo_kv_manager::dtype::DType::FP8 => ffi::DataType::U8,
            dynamo_kv_manager::dtype::DType::FP16 => ffi::DataType::F16,
            dynamo_kv_manager::dtype::DType::BF16 => ffi::DataType::BF16,
            dynamo_kv_manager::dtype::DType::FP32 => ffi::DataType::F32,
            dynamo_kv_manager::dtype::DType::FP64 => ffi::DataType::F64,
            dynamo_kv_manager::dtype::DType::U8 => ffi::DataType::U8,
            dynamo_kv_manager::dtype::DType::U16 => ffi::DataType::U16,
            dynamo_kv_manager::dtype::DType::U32 => ffi::DataType::U32,
            dynamo_kv_manager::dtype::DType::U64 => ffi::DataType::U64,
            dynamo_kv_manager::dtype::DType::I8 => ffi::DataType::I8,
            dynamo_kv_manager::dtype::DType::I16 => ffi::DataType::I16,
            dynamo_kv_manager::dtype::DType::I32 => ffi::DataType::I32,
            dynamo_kv_manager::dtype::DType::I64 => ffi::DataType::I64,
        }
    }

    fn shape_and_strides(&self) -> ShapeAndStrides {
        let shape_i64: Vec<i64> = self.region_shape.iter().map(|x| *x as i64).collect();
        ShapeAndStrides::new_contiguous(&shape_i64)
    }
}

#[pyclass]
pub struct BlockManager {
    inner: Arc<dyn dynamo_kv_manager::layout::BlockLayout + Send + Sync>,
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

        Ok(BlockManager { inner: Arc::from(block_layout) })
    }

    fn py_capsule(&self, block_idx: usize, layer_idx: usize) -> PyResult<PyObject> {
        // Get memory region and metadata for tensor construction
        let memory_region = self.inner.get_memory_region(block_idx, layer_idx)
            .expect(&format!("Failed to get memory region for block {}, layer {}", block_idx, layer_idx));
        let region_shape = vec![self.inner.page_size(), self.inner.inner_dim()];
        let dtype = self.inner.dtype();
        let storage_type = self.inner.storage_type();

        // Create a DlPackTensor instance
        let dlpack_tensor = DlPackTensor {
            memory_region,
            region_shape,
            dtype,
            storage_type,
        };

        // Convert to Python object using into_py
        let manager_ctx = ManagerCtx::new(dlpack_tensor);
        let py_capsule = Python::with_gil(|py| {
            manager_ctx.into_py(py)
        });
        Ok(py_capsule)
    }
}
