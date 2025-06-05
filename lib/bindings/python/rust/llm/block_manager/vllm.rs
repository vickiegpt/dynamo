// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::{prelude::*, wrap_pymodule};

use dynamo_llm::block_manager::{self as bm};

use crate::llm::block_manager::BlockManager as PyBlockManager;

use crate::to_pyerr;

mod block_list;
mod request;

pub use block_list::{BlockListType, BlockState, BlockStates, KvbmBlockList};
pub use request::KvbmRequest;

#[pymodule]
fn _vllm_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KvbmCacheManager>()?;
    m.add_class::<KvbmRequest>()?;
    m.add_class::<KvbmBlockList>()?;
    m.add_class::<BlockState>()?;
    m.add_class::<BlockStates>()?;
    Ok(())
}

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(_vllm_integration))?;
    Ok(())
}

#[pyclass]
pub struct KvbmCacheManager {
    block_manager: PyBlockManager,
}

impl KvbmCacheManager {
    #[inline(always)]
    pub fn block_manager(&self) -> &bm::KvBlockManager<bm::BasicMetadata> {
        self.block_manager.get_block_manager()
    }
}

#[pymethods]
impl KvbmCacheManager {
    #[new]
    #[pyo3(signature = (block_manager))]
    pub fn new(block_manager: PyBlockManager) -> PyResult<Self> {
        Ok(Self { block_manager })
    }

    pub fn get_computed_blocks(&self, request: KvbmRequest) -> PyResult<KvbmBlockList> {
        let sequence_hashes = request.sequence_hashes();
        tracing::debug!("sequence_hashes: {:?}", sequence_hashes);

        let blocks = self
            .block_manager()
            .device()
            .unwrap()
            .match_sequence_hashes_blocking(&sequence_hashes)
            .map_err(to_pyerr)?;
        tracing::debug!("blocks: {:?}", blocks);

        Ok(KvbmBlockList::new(BlockListType::Immutable(blocks)))
    }
}
