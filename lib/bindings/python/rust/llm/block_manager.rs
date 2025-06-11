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
use pyo3::PyResult;

mod block;
mod block_list;
mod dlpack;
mod layer;

pub mod vllm;

use llm_rs::block_manager::{storage::{torch::{TorchDevice, TorchTensor}, DeviceStorage, DeviceAllocator}, LayoutType};

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<layer::Layer>()?;
    m.add_class::<block::Block>()?;
    m.add_class::<block_list::BlockList>()?;
    m.add_class::<BlockManager>()?;
    m.add_class::<BlockManagerConfig>()?;

    vllm::add_to_module(m)?;

    Ok(())
}

#[derive(Clone, Debug)]
struct VllmTensor {
    _py_tensor: Py<PyAny>,
    device: TorchDevice,
    data_ptr: u64,
    size_bytes: usize,
    shape: Vec<usize>,
}

impl VllmTensor {
    fn new(py_tensor: Py<PyAny>) -> anyhow::Result<Self> {
        Python::with_gil(|py| {
            let device = py_tensor.getattr(py, "device")?;
            let device_type = device.getattr(py, "type")?.extract::<String>(py)?;

            let device = if device_type == "cuda" {
                TorchDevice::Cuda(device.getattr(py, "index")?.extract::<usize>(py)?)
            } else {
                TorchDevice::Other(device_type)
            };

            let data_ptr = py_tensor.call_method0(py, "data_ptr")?.extract::<u64>(py)?;
            let size_bytes = py_tensor.call_method0(py, "size_bytes")?.extract::<usize>(py)?;
            let shape = py_tensor.getattr(py, "shape")?.extract::<Vec<usize>>(py)?;

            Ok(Self {
                _py_tensor: py_tensor,
                device,
                data_ptr,
                size_bytes,
                shape,
            })
        })
    }
}

impl TorchTensor for VllmTensor {
    fn device(&self) -> TorchDevice {
        self.device.clone()
    }

    fn data_ptr(&self) -> u64 {
        self.data_ptr
    }
    
    fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

#[pyclass]
#[derive(Clone)]
struct BlockManagerConfig {
    pub num_layers: usize,
    pub num_blocks: usize,
    pub outer_dim: usize,
    pub page_size: usize,
    pub inner_dim: usize,
    pub tensors: Vec<VllmTensor>,
    pub layout_type: LayoutType,
    pub dtype: Option<String>,
}

#[pymethods]
impl BlockManagerConfig {
    #[new]
    #[pyo3(signature = (num_layers, num_blocks, page_size, vllm_tensors, dtype="fp16".to_string()))]
    fn new(num_layers: usize, num_blocks: usize, page_size: usize, vllm_tensors: Vec<Py<PyAny>>, dtype: Option<String>) -> PyResult<Self> {
        if num_layers == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of layers must be greater than 0"
            ));
        }
        
        if num_layers != vllm_tensors.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Number of layers ({}) does not match number of tensors ({})",
                num_layers, vllm_tensors.len()
            )));
        }

        let mut shape = None;
        let mut tensors = Vec::with_capacity(vllm_tensors.len());
        for tensor in vllm_tensors {
            let tensor = VllmTensor::new(tensor).map_err(to_pyerr)?;

            if let Some(shape) = shape.as_ref() {
                if tensor.shape != *shape {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Tensor shape ({:?}) does not match previous tensor shape ({:?})",
                        tensor.shape, shape
                    )));
                }
            } else {
                shape = Some(tensor.shape.clone());
            }

            tensors.push(tensor);
        }

        let shape = shape.as_ref().unwrap();

        let layout_type;
        let inner_dim;
        let outer_dim;

        // We need to handle 4 different possible layouts here:
        // 1. [2, num_blocks, block_size, num_kv_heads, head_size] (FlashAttn + variants)
        // 2. [num_blocks, 2, block_size, num_kv_heads, head_size] (FlashInfer)
        // 3. [2, num_blocks, block_size * num_kv_heads * head_size] (PagedAttention + variants)
        // 4. [num_blocks, block_size, head_size] (MLA)
        // TODO: This is cursed.
        if shape[0] == num_blocks {
            layout_type = LayoutType::LayerSeparate{ outer_contiguous: false };
            if shape.len() == 3 {
                outer_dim = 1;
                inner_dim = shape[2];
            } else if shape.len() == 5 {
                if shape[1] != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Expected outer_dim to be 2, got {}", shape[1]
                    )));
                }
                outer_dim = shape[1];
                inner_dim = shape[3] * shape[4];
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unrecognized kv layer shape: {:?}", shape
                )));
            }
        } else if shape[0] == 2 {
            outer_dim = 2;
            layout_type = LayoutType::LayerSeparate{ outer_contiguous: true };
            if shape.len() == 3 {
                inner_dim = shape[2] / page_size;
            } else if shape.len() == 5 {
                inner_dim = shape[3] * shape[4];
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unrecognized kv layer shape: {:?}", shape
                )));
            }
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unrecognized kv layer shape: {:?}", shape
            )));
        }

        tracing::info!("Inferred layout type: {:?}, outer_dim: {}, inner_dim: {}", layout_type, outer_dim, inner_dim);

        Ok(Self {
            num_layers,
            num_blocks,
            outer_dim,
            page_size,
            inner_dim,
            tensors,
            layout_type,
            dtype,
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BlockManager {
    inner: Arc<dynamo_llm::block_manager::ReferenceBlockManager>,
    // TODO: Metadata should be stored in the block manager?
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
}

#[pymethods]
impl BlockManager {
    #[new]
    #[pyo3(signature = (worker_id, config, host_num_blocks=None, device_id=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        worker_id: u64,
        config: BlockManagerConfig,
        host_num_blocks: Option<usize>,
        device_id: usize,
    ) -> PyResult<Self> {
        let mut kvbm_config = dynamo_llm::block_manager::KvBlockManagerConfig::builder().runtime(
            dynamo_llm::block_manager::KvManagerRuntimeConfig::builder()
                .worker_id(worker_id)
                .build()
                .map_err(to_pyerr)?,
        );
        let mut model_config = dynamo_llm::block_manager::KvManagerModelConfig::builder()
            .num_layers(config.num_layers)
            .outer_dim(config.outer_dim)
            .page_size(config.page_size)
            .inner_dim(config.inner_dim);

        let mut dtype_ = dynamo_llm::common::dtype::DType::FP16; // Default in block_manager config
        if let Some(dtype_str) = config.dtype {
            dtype_ = match dtype_str.as_str() {
                "fp8" | "FP8" => dynamo_llm::common::dtype::DType::FP8,
                "fp16" | "FP16" => dynamo_llm::common::dtype::DType::FP16,
                "bf16" | "BF16" => dynamo_llm::common::dtype::DType::BF16,
                "fp32" | "FP32" => dynamo_llm::common::dtype::DType::FP32,
                "u8" | "U8" => dynamo_llm::common::dtype::DType::U8,
                "u16" | "U16" => dynamo_llm::common::dtype::DType::U16,
                "u32" | "U32" => dynamo_llm::common::dtype::DType::U32,
                "u64" | "U64" => dynamo_llm::common::dtype::DType::U64,
                "i8" | "I8" => dynamo_llm::common::dtype::DType::I8,
                "i16" | "I16" => dynamo_llm::common::dtype::DType::I16,
                "i32" | "I32" => dynamo_llm::common::dtype::DType::I32,
                "i64" | "I64" => dynamo_llm::common::dtype::DType::I64,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unsupported dtype: {}",
                        dtype_str
                    )))
                }
            };
        }
        model_config = model_config.dtype(dtype_);
        kvbm_config = kvbm_config.model(model_config.build().map_err(to_pyerr)?);
        if let Some(host_num_blocks) = host_num_blocks {
            kvbm_config = kvbm_config.host_layout(
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(host_num_blocks)
                    .layout_type(config.layout_type)
                    .allocator(
                        dynamo_llm::block_manager::storage::PinnedAllocator::new()
                            .map_err(to_pyerr)?,
                    )
                    .build()
                    .map_err(to_pyerr)?,
            );
        }

        let device_allocator = DeviceAllocator::new(device_id).map_err(to_pyerr)?;

        let mut device_tensors = Vec::with_capacity(config.tensors.len());

        for tensor in config.tensors {
            device_tensors.push(DeviceStorage::new_from_torch(device_allocator.ctx(), Box::new(tensor.clone())).map_err(to_pyerr)?);
        }

        kvbm_config = kvbm_config.device_layout(
            dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                .num_blocks(config.num_blocks)
                .layout_type(config.layout_type)
                .storage(Some(device_tensors))
                .build()
                .map_err(to_pyerr)?,
        );
        
        let kvbm_config = kvbm_config.build().map_err(to_pyerr)?;
        let tokio_runtime = pyo3_async_runtimes::tokio::get_runtime();
        Ok(BlockManager {
            inner: Arc::from(
                tokio_runtime
                    .block_on(async {
                        dynamo_llm::block_manager::ReferenceBlockManager::new(kvbm_config)
                    })
                    .map_err(to_pyerr)?,
            ),
            dtype: dtype_,
            device_id,
        })
    }

    fn allocate_host_blocks_blocking(&self, count: usize) -> PyResult<block_list::BlockList> {
        let blocks = self
            .inner
            .host()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Host allocator not available")
            })?
            .allocate_blocks_blocking(count)
            .map_err(to_pyerr)?;
        // Wrap each block in an enum accounting for Pinned & Device block
        let blocks = blocks.into_iter().map(block::BlockType::Pinned).collect();
        Ok(block_list::BlockList::from_rust(
            blocks,
            self.dtype,
            self.device_id,
        ))
    }

    #[pyo3(signature = (count))]
    fn allocate_host_blocks<'py>(
        &self,
        py: Python<'py>,
        count: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let dtype = self.dtype;
        let device_id = self.device_id;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let blocks = inner
                .host()
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Host allocator not available")
                })?
                .allocate_blocks(count)
                .await
                .map_err(to_pyerr)?;
            // Wrap each block in an enum accounting for Pinned & Device block
            let blocks = blocks.into_iter().map(block::BlockType::Pinned).collect();
            Ok(block_list::BlockList::from_rust(blocks, dtype, device_id))
        })
    }

    fn allocate_device_blocks_blocking(&self, count: usize) -> PyResult<block_list::BlockList> {
        let blocks = self
            .inner
            .device()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Device allocator not available")
            })?
            .allocate_blocks_blocking(count)
            .map_err(to_pyerr)?;
        // Wrap each block in an enum accounting for Pinned & Device block
        let blocks = blocks.into_iter().map(block::BlockType::Device).collect();
        Ok(block_list::BlockList::from_rust(
            blocks,
            self.dtype,
            self.device_id,
        ))
    }

    #[pyo3(signature = (count))]
    fn allocate_device_blocks<'py>(
        &self,
        py: Python<'py>,
        count: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let dtype = self.dtype;
        let device_id = self.device_id;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let blocks = inner
                .device()
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Device allocator not available")
                })?
                .allocate_blocks(count)
                .await
                .map_err(to_pyerr)?;
            // Wrap each block in an enum accounting for Pinned & Device block
            let blocks = blocks.into_iter().map(block::BlockType::Device).collect();
            Ok(block_list::BlockList::from_rust(blocks, dtype, device_id))
        })
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }
}

impl BlockManager {
    #[inline(always)]
    pub fn get_block_manager(
        &self,
    ) -> &dynamo_llm::block_manager::KvBlockManager<dynamo_llm::block_manager::BasicMetadata> {
        self.inner.as_ref()
    }
}
