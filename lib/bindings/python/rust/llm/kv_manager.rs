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

    fn tensor(&self, block_idx: usize, layer_idx: usize) -> PyResult<PyObject> {
        let memory_region = self.inner.get_memory_region(block_idx, layer_idx)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get memory region: {}", e)))?;
        let region_size = self.inner.memory_region_size();
        let region_shape = vec![self.inner.page_size(), self.inner.inner_dim()];
        let dtype = self.inner.dtype();

        let torch_tensor = match self.inner.storage_type() {
            dynamo_kv_manager::storage::StorageType::Device(_) => {
                Python::with_gil(|py| -> PyResult<PyObject> {
                    let cupy = py.import("cupy")?;
                    let cupy_type = match dtype {
                        dynamo_kv_manager::dtype::DType::FP8 => cupy.getattr("uint8")?,
                        dynamo_kv_manager::dtype::DType::FP16 => cupy.getattr("float16")?,
                        dynamo_kv_manager::dtype::DType::BF16 => cupy.getattr("float16")?,
                        dynamo_kv_manager::dtype::DType::FP32 => cupy.getattr("float32")?,
                        dynamo_kv_manager::dtype::DType::FP64 => cupy.getattr("float64")?,
                        dynamo_kv_manager::dtype::DType::U8 => cupy.getattr("uint8")?,
                        dynamo_kv_manager::dtype::DType::U16 => cupy.getattr("uint16")?,
                        dynamo_kv_manager::dtype::DType::U32 => cupy.getattr("uint32")?,
                        dynamo_kv_manager::dtype::DType::U64 => cupy.getattr("uint64")?,
                        dynamo_kv_manager::dtype::DType::I8 => cupy.getattr("int8")?,
                        dynamo_kv_manager::dtype::DType::I16 => cupy.getattr("int16")?,
                        dynamo_kv_manager::dtype::DType::I32 => cupy.getattr("int32")?,
                        dynamo_kv_manager::dtype::DType::I64 => cupy.getattr("int64")?,
                    };
                    let cupy_mem = cupy.getattr("cuda")?.getattr("UnownedMemory")?.call1((memory_region, region_size, py.None()))?;
                    let cupy_ptr = cupy.getattr("cuda")?.getattr("MemoryPointer")?.call1((cupy_mem, 0))?;
                    let cupy_array = cupy.getattr("ndarray")?.call1((region_shape, cupy_type, cupy_ptr))?;
                    let torch = py.import("torch")?;
                    let torch_tensor = torch.getattr("from_dlpack")?.call1((cupy_array,))?;
                    Ok(torch_tensor.into())
                })?
            }
            dynamo_kv_manager::storage::StorageType::Pinned => {
                return Err(PyRuntimeError::new_err("Pinned memory tensor retrieval not implemented"));
            }
            dynamo_kv_manager::storage::StorageType::System => {
                Python::with_gil(|py| -> PyResult<PyObject> {
                    let ctypes = py.import("ctypes")?;
                    let ctypes_type = match dtype {
                        dynamo_kv_manager::dtype::DType::FP8 => ctypes.getattr("c_ubyte")?,
                        dynamo_kv_manager::dtype::DType::FP16 => ctypes.getattr("c_ushort")?,
                        dynamo_kv_manager::dtype::DType::BF16 => ctypes.getattr("c_ushort")?,
                        dynamo_kv_manager::dtype::DType::FP32 => ctypes.getattr("c_float")?,
                        dynamo_kv_manager::dtype::DType::FP64 => ctypes.getattr("c_double")?,
                        dynamo_kv_manager::dtype::DType::U8 => ctypes.getattr("c_uint8")?,
                        dynamo_kv_manager::dtype::DType::U16 => ctypes.getattr("c_uint16")?,
                        dynamo_kv_manager::dtype::DType::U32 => ctypes.getattr("c_uint32")?,
                        dynamo_kv_manager::dtype::DType::U64 => ctypes.getattr("c_uint64")?,
                        dynamo_kv_manager::dtype::DType::I8 => ctypes.getattr("c_int8")?,
                        dynamo_kv_manager::dtype::DType::I16 => ctypes.getattr("c_int16")?,
                        dynamo_kv_manager::dtype::DType::I32 => ctypes.getattr("c_int32")?,
                        dynamo_kv_manager::dtype::DType::I64 => ctypes.getattr("c_int64")?,
                    };
                    let ctypes_ptr_type = ctypes.getattr("POINTER")?.call1((ctypes_type,))?;
                    let ctypes_ptr = ctypes.getattr("cast")?.call1((memory_region, ctypes_ptr_type))?;
                    let np = py.import("numpy")?;
                    let np_array = np.getattr("ctypeslib")?.getattr("as_array")?.call1((ctypes_ptr, region_shape))?;
                    let torch = py.import("torch")?;
                    let torch_tensor = torch.getattr("from_dlpack")?.call1((np_array,))?;
                    Ok(torch_tensor.into())
                })?
            }
            _ => {
                return Err(PyRuntimeError::new_err("Unsupported storage type"));
            }
        };

        Ok(torch_tensor)
    }
}
