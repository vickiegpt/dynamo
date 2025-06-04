#![cfg(feature = "block-manager")]


use pyo3::prelude::*;

use dynamo_llm::block_manager::{self as bm};

use crate::llm::block_manager::BlockManager as PyBlockManager;
use crate::to_pyerr;
use crate::llm::kvbm_utils::*;


/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DynamoVllmKvCacheManager>()?;
    Ok(())
}


#[pyclass]
struct DynamoVllmKvCacheManager {
    block_manager: PyBlockManager,
}

impl DynamoVllmKvCacheManager {
    #[inline(always)]
    fn block_manager(&self) -> &bm::KvBlockManager<bm::BasicMetadata> {
        self.block_manager.get_block_manager()
    }
}

#[pymethods]
impl DynamoVllmKvCacheManager {
    #[new]
    #[pyo3(signature = (block_manager))]
    fn new(block_manager: PyBlockManager) -> PyResult<Self> {
        Ok(Self { block_manager })
    }

    fn get_computed_blocks(&self, request: KvRequest) -> PyResult<DynamoKvBlockList> {
        let sequence_hashes = request.sequence_hashes();
        let blocks = self
            .block_manager()
            .device()
            .unwrap()
            .match_sequence_hashes_blocking(&sequence_hashes)
            .map_err(to_pyerr)?;

        Ok(DynamoKvBlockList::new(BlockListType::Immutable(blocks)))
    }
}
