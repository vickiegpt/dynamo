use super::*;
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

        Ok(DynamoVllmKvBlockList {
            blocks: BlockListType::Immutable(blocks),
        })
    }
}

/// Request Inputs
#[pyclass]
#[derive(Debug, Clone)]
pub struct KvRequest {
    tokens: Tokens,
    lora_name: Option<String>,
    salt_hash: Option<String>,
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
        let tokens: Tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<_>>().into();

        // compute salt
        #[derive(serde::Serialize)]
        struct Salt {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<String>,
        }

        let salt = Salt {
            salt: salt_hash.clone(),
            lora_name: lora_name.clone(),
        };

        let salt_bytes = serde_json::to_vec(&salt).unwrap();
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        let sequence = Arc::new(TokenBlockSequence::new(tokens, block_size, Some(salt_hash)));

        Self {
            tokens,
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

enum BlockListType {
    Immutable(Vec<bm::block::ImmutableBlock<DeviceStorageType, bm::BasicMetadata>>),
    Mutable(Vec<bm::block::MutableBlock<DeviceStorageType, bm::BasicMetadata>>),
}

#[pyclass]
#[derive(Debug, Clone)]
struct DynamoVllmKvBlockList {
    blocks: BlockListType,
}

#[pymethods]
impl DynamoVllmKvBlockList {
    fn get_block_id(&self, block_idx: u64) -> PyResult<u64> {
        match &self.blocks {
            BlockListType::Immutable(blocks) => {
                blocks.get(block_idx as usize).map(|b| b.block_id())
            }
            BlockListType::Mutable(blocks) => blocks.get(block_idx as usize).map(|b| b.block_id()),
        }
    }

    fn get_block_hash(&self, block_idx: u64) -> PyResult<Option<u64>> {
        match &self.blocks {
            BlockListType::Immutable(blocks) => {
                blocks.get(block_idx as usize).map(|b| b.sequence_hash())
            }
            BlockListType::Mutable(blocks) => {
                blocks.get(block_idx as usize).map(|b| b.sequence_hash())
            }
        }
    }

    fn block_ids(&self) -> Vec<u64> {
        match &self.blocks {
            BlockListType::Immutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
            BlockListType::Mutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
        }
    }

    fn unhashed_block_ids(&self) -> Vec<u64> {
        match &self.blocks {
            BlockListType::Immutable(blocks) => vec![],
            BlockListType::Mutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
        }
    }

    #[pyo3(name = "__len__")]
    fn len(&self) -> usize {
        match &self.blocks {
            BlockListType::Immutable(blocks) => blocks.len(),
            BlockListType::Mutable(blocks) => blocks.len(),
        }
    }
}
