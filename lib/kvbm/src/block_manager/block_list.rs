// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "block-manager")]

use super::*;
use pyo3::{types::PyList, PyResult, Python};
use std::sync::{Arc, Mutex};

#[pyclass]
pub struct BlockList {
    inner: Vec<Arc<Mutex<block::BlockType>>>,
    // TODO: Metadata should be stored in the block manager?
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
    // Python iterator state
    py_itr_idx: usize,
}

impl BlockList {
    pub fn from_rust(
        block_list: Vec<block::BlockType>,
        dtype: dynamo_llm::common::dtype::DType,
        device_id: usize,
    ) -> Self {
        Self {
            inner: block_list
                .into_iter()
                .map(|b| Arc::new(Mutex::new(b)))
                .collect(),
            dtype,
            device_id,
            py_itr_idx: 0,
        }
    }
}

#[pymethods]
impl BlockList {
    #[pyo3(signature = ())]
    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let blocks: Vec<block::Block> = self
            .inner
            .iter()
            .map(|b| block::Block::from_rust(b.clone(), self.dtype, self.device_id))
            .collect();
        PyList::new(py, blocks)
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    fn __getitem__(&self, index: usize) -> PyResult<block::Block> {
        if index >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range for BlockList of length {}",
                index,
                self.inner.len()
            )));
        }
        let block = block::Block::from_rust(self.inner[index].clone(), self.dtype, self.device_id);
        Ok(block)
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        // Reset iterator index at the beginning of each iteration
        // Use to_list() for iterating concurrently
        slf.py_itr_idx = 0;
        Ok(slf)
    }

    fn __next__(&mut self) -> PyResult<block::Block> {
        if self.py_itr_idx >= self.inner.len() {
            return Err(pyo3::exceptions::PyStopIteration::new_err(
                "No more items in BlockList",
            ));
        }
        let block = block::Block::from_rust(
            self.inner[self.py_itr_idx].clone(),
            self.dtype,
            self.device_id,
        );
        self.py_itr_idx += 1;
        Ok(block)
    }
}
