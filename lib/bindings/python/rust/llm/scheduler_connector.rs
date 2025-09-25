// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for scheduler worker device blocks.

use std::sync::Arc;

use pyo3::prelude::*;

use dynamo_llm::integrations::vllm::scheduler::worker::WorkerDeviceBlocks as RustWorkerDeviceBlocks;
use dynamo_llm::block_manager::storage::torch::{TorchDevice, TorchTensor};

use crate::to_pyerr;

/// A wrapper around a Torch tensor for scheduler connector.
/// We hold onto the py object to ensure it doesn't get GCed.
#[derive(Clone, Debug)]
pub struct SchedulerTensor {
    _py_tensor: Py<PyAny>,
    device: TorchDevice,
    data_ptr: u64,
    size_bytes: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl SchedulerTensor {
    pub fn new(py_tensor: Py<PyAny>) -> anyhow::Result<Self> {
        Python::with_gil(|py| {
            let device = py_tensor.getattr(py, "device")?;
            let device_type = device.getattr(py, "type")?.extract::<String>(py)?;

            let device = if device_type == "cuda" {
                TorchDevice::Cuda(device.getattr(py, "index")?.extract::<usize>(py)?)
            } else {
                TorchDevice::Other(device_type)
            };

            let data_ptr = py_tensor.call_method0(py, "data_ptr")?.extract::<u64>(py)?;
            let size_bytes = py_tensor.getattr(py, "nbytes")?.extract::<usize>(py)?;
            let shape = py_tensor.getattr(py, "shape")?.extract::<Vec<usize>>(py)?;
            let stride = py_tensor
                .call_method0(py, "stride")?
                .extract::<Vec<usize>>(py)?;

            Ok(Self {
                _py_tensor: py_tensor,
                device,
                data_ptr,
                size_bytes,
                shape,
                stride,
            })
        })
    }
}

impl TorchTensor for SchedulerTensor {
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

    fn stride(&self) -> Vec<usize> {
        self.stride.clone()
    }
}

/// Python wrapper for WorkerDeviceBlocks.
///
/// This class provides worker device block construction for the scheduler
/// without requiring leader/worker synchronization.
#[pyclass]
pub struct WorkerDeviceBlocks {
    inner: Arc<RustWorkerDeviceBlocks>,
}

#[pymethods]
impl WorkerDeviceBlocks {
    /// Create local blocks from KV cache tensors.
    ///
    /// Args:
    ///     tensors: List of torch tensors (one per layer)
    ///     num_device_blocks: Number of device blocks
    ///     page_size: Page size (typically 16)
    ///     device_id: CUDA device ID
    ///     dtype_width_bytes: Bytes per dtype element (e.g., 2 for fp16)
    ///     is_fully_contiguous: Whether layout is fully contiguous
    #[new]
    #[pyo3(signature = (tensors, num_device_blocks, page_size, device_id=0, dtype_width_bytes=2, is_fully_contiguous=false))]
    fn new(
        tensors: Vec<Py<PyAny>>,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        is_fully_contiguous: bool,
    ) -> PyResult<Self> {
        // Convert Python tensors to Rust tensors
        let mut rust_tensors: Vec<Arc<dyn TorchTensor>> = Vec::with_capacity(tensors.len());

        for tensor in tensors {
            let scheduler_tensor = SchedulerTensor::new(tensor).map_err(to_pyerr)?;
            rust_tensors.push(Arc::new(scheduler_tensor));
        }

        // Build worker device blocks
        let worker_blocks = RustWorkerDeviceBlocks::from_tensors(
            rust_tensors,
            num_device_blocks,
            page_size,
            device_id,
            dtype_width_bytes,
            is_fully_contiguous,
        )
        .map_err(to_pyerr)?;

        Ok(Self {
            inner: Arc::new(worker_blocks),
        })
    }

    /// Get the number of device blocks.
    #[getter]
    fn num_device_blocks(&self) -> usize {
        self.inner.num_device_blocks
    }

    /// Get the number of layers.
    #[getter]
    fn num_layers(&self) -> usize {
        self.inner.num_layers
    }

    /// Get the outer dimension.
    #[getter]
    fn outer_dim(&self) -> usize {
        self.inner.outer_dim
    }

    /// Get the page size.
    #[getter]
    fn page_size(&self) -> usize {
        self.inner.page_size
    }

    /// Get the inner dimension.
    #[getter]
    fn inner_dim(&self) -> usize {
        self.inner.inner_dim
    }

    /// Get the dtype width in bytes.
    #[getter]
    fn dtype_width_bytes(&self) -> usize {
        self.inner.dtype_width_bytes
    }

    /// Get the total bytes per block.
    #[getter]
    fn bytes_per_block(&self) -> usize {
        self.inner.bytes_per_block
    }

    /// Get the number of blocks that were created.
    fn num_blocks(&self) -> usize {
        self.inner.device_blocks.len()
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        format!(
            "WorkerDeviceBlocks(num_blocks={}, num_layers={}, outer_dim={}, page_size={}, inner_dim={}, dtype_width_bytes={}, bytes_per_block={})",
            self.inner.device_blocks.len(),
            self.inner.num_layers,
            self.inner.outer_dim,
            self.inner.page_size,
            self.inner.inner_dim,
            self.inner.dtype_width_bytes,
            self.inner.bytes_per_block
        )
    }
}

/// Register the module with Python.
pub fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "scheduler_connector")?;
    m.add_class::<WorkerDeviceBlocks>()?;
    parent_module.add_submodule(&m)?;
    Ok(())
}