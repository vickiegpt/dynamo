use pyo3::prelude::*;

use super::utils::*;
use crate::llm::block_manager::BlockManager as PyBlockManager;
use crate::to_pyerr;

#[pyclass]
struct DynamoTrtllmKvCacheManager {
    block_manager: PyBlockManager,
}

#[pymethods]
impl DynamoTrtllmKvCacheManager {
    #[new]
    fn new(block_manager: PyBlockManager) -> PyResult<Self> {
        Ok(Self { block_manager })
    }

    fn add_sequence(&self, request: KvRequest) -> PyResult<(DynamoKvBlockList, DynamoKvBlockList)> {
        let block_manager = self.block_manager.get_block_manager();

        let total_blocks = request.tbs().blocks().len() + 1;

        let immutable = block_manager
            .device()
            .unwrap()
            .match_sequence_hashes_blocking(&request.sequence_hashes())
            .map_err(to_pyerr)?;
        let num_immutable = immutable.len();
        let immutable_list = DynamoKvBlockList::new(BlockListType::Immutable(immutable));

        let num_blocks_to_allocate = total_blocks - num_immutable;

        let mutable = block_manager
            .device()
            .unwrap()
            .allocate_blocks_blocking(num_blocks_to_allocate)
            .map_err(to_pyerr)?;
        let mutable_list = DynamoKvBlockList::new(BlockListType::Mutable(mutable));

        Ok((immutable_list, mutable_list))
    }
}
