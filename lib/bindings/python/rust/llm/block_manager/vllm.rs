use super::*;
use dynamo_llm::block_manager::vllm::RequestID;
use pyo3::PyResult;

use dynamo_llm::block_manager::{self as bm, block::BlockIdentifier};
use dynamo_llm::tokens::{compute_hash_v2, SaltHash, TokenBlockSequence, Tokens};

use super::BlockManager as PyBlockManager;

type DeviceStorageType = bm::storage::DeviceStorage;

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

    fn get_computed_blocks(&self, request: KvRequest) -> PyResult<DynamoVllmKvBlockList> {
        let sequence_hashes = request.sequence_hashes();
        let blocks = self
            .block_manager()
            .device()
            .unwrap()
            .match_sequence_hashes_blocking(&sequence_hashes)
            .map_err(to_pyerr)?;

        Ok(DynamoVllmKvBlockList::new(BlockListType::Immutable(blocks)))
    }
}

/// Request Inputs
#[pyclass]
#[derive(Debug, Clone)]
pub struct KvRequest {
    lora_name: Option<String>,
    salt_hash: u64,
    tbs: Arc<TokenBlockSequence>,
}

#[pymethods]
impl KvRequest {
    #[new]
    #[pyo3(signature = (tokens, block_size, lora_name=None, salt_hash=None))]
    fn new(
        tokens: Vec<usize>,
        block_size: usize,
        lora_name: Option<String>,
        salt_hash: Option<String>,
    ) -> Self {
        let tokens: Tokens = tokens
            .into_iter()
            .map(|t| t as u32)
            .collect::<Vec<_>>()
            .into();

        // compute salt
        #[derive(serde::Serialize)]
        struct Salt {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<String>,
        }

        let salt = Salt {
            salt: salt_hash,
            lora_name: lora_name.clone(),
        };

        let salt_bytes = serde_json::to_vec(&salt).unwrap();
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        let sequence = Arc::new(TokenBlockSequence::new(tokens, block_size, Some(salt_hash)));

        Self {
            lora_name,
            salt_hash,
            tbs: sequence,
        }
    }

    fn sequence_hashes(&self) -> Vec<u64> {
        self.tbs
            .blocks()
            .iter()
            .map(|b| b.sequence_hash())
            .collect()
    }
}

#[derive(Debug)]
enum BlockListType {
    Immutable(Vec<bm::block::ImmutableBlock<DeviceStorageType, bm::BasicMetadata>>),
    Mutable(Vec<bm::block::MutableBlock<DeviceStorageType, bm::BasicMetadata>>),
}

#[pyclass]
#[derive(Debug, Clone)]
struct DynamoVllmKvBlockList {
    blocks: Arc<std::sync::Mutex<Option<BlockListType>>>,
    count: usize,
}

impl DynamoVllmKvBlockList {
    fn new(blocks: BlockListType) -> Self {
        let count = match &blocks {
            BlockListType::Immutable(blocks) => blocks.len(),
            BlockListType::Mutable(blocks) => blocks.len(),
        };

        Self {
            blocks: Arc::new(std::sync::Mutex::new(Some(blocks))),
            count,
        }
    }

    fn take_blocks(&self) -> Option<BlockListType> {
        let mut blocks = self.blocks.lock().unwrap();
        blocks.take()
    }
}

#[pymethods]
impl DynamoVllmKvBlockList {
    fn get_block_id(&self, block_idx: usize) -> PyResult<usize> {
        let blocks = self.blocks.lock().unwrap();
        let block_id = match &*blocks {
            Some(BlockListType::Immutable(blocks)) => blocks.get(block_idx).map(|b| b.block_id()),
            Some(BlockListType::Mutable(blocks)) => blocks.get(block_idx).map(|b| b.block_id()),
            None => None,
        };

        Ok(block_id.ok_or_else(|| to_pyerr("block not found"))?)
    }

    fn get_block_hash(&self, block_idx: usize) -> PyResult<Option<u64>> {
        let blocks = self.blocks.lock().unwrap();
        let block_id = match &*blocks {
            Some(BlockListType::Immutable(blocks)) => blocks
                .get(block_idx)
                .ok_or(to_pyerr("block not found"))?
                .sequence_hash()
                .ok(),
            Some(BlockListType::Mutable(blocks)) => blocks
                .get(block_idx)
                .ok_or(to_pyerr("block not found"))?
                .sequence_hash()
                .ok(),
            None => None,
        };

        Ok(block_id)
    }

    #[pyo3(name = "__len__")]
    fn len(&self) -> usize {
        self.count
    }
}

// struct SlotUpdate {
//     /// The request ID.
//     request_id: RequestID,

//     /// External state about the number of computed tokens in the request.
//     /// This should match the slots expectation.
//     request_num_computed_tokens: usize,

//     /// The number of new tokens which advances the sequence state.
//     /// This is the number of tokens which will be computed in the near future.
//     /// When [BaseKvCacheManager::update_slot] is called again, these tokens will be committed.
//     num_new_tokens: usize,

//     /// The number of new computed tokens in the request.
//     /// The `num_new_tokens / block_size` should be equal to the length of the `new_computed_blocks`,
//     /// it may have a remainder for the partial block state.
//     num_new_computed_tokens: Option<usize>,

//     /// The new computed blocks which advance the sequence state.
//     new_computed_blocks: Option<BlockList<DeviceStorage, BasicMetadata>>,

//     /// The number of lookahead blocks to cache.
//     num_lookahead_blocks: Option<usize>,

//     /// Whether to delay caching the blocks.
//     delay_cache_blocks: Option<bool>,
// }

// struct Slot {
//     num_computed_tokens: usize,
//     sequence_blocks: Vec<bm::block::ImmutableBlock<DeviceStorageType, bm::BasicMetadata>>,
//     inflight_blocks: Vec<bm::block::MutableBlock<DeviceStorage, BasicMetadata>>,
// }
