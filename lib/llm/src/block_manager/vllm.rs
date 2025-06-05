use std::collections::VecDeque;

use super::block::{self, list::BlockList, BlockId, BlockIdentifier, BlockState, MutableBlock};
use crate::tokens::{SaltHash, SequenceHash, TokenBlockSequence, Tokens};

use super::{pool::BlockPoolError, *};

pub type RequestID = u64;

#[derive(Debug, thiserror::Error)]
pub enum CacheManagerError {
    #[error(transparent)]
    BlockPoolError(#[from] BlockPoolError),

    #[error("Slot not found for request ID: {0}")]
    SlotNotFound(RequestID),

    #[error("Slot already exists for request ID: {0}")]
    SlotAlreadyExists(RequestID),
    // #[error("Block list error: {0}")]
    // BlockListError(#[from] BlockListError),
}

/// Stateful implementation which is close, but not exactly conformant to the
/// vLLM's KVCacheManagerProtocol.
///
/// A Python class will be needed to complete the protocol implementation.
pub struct BaseKvCacheManager {
    block_manager: KvBlockManager<BasicMetadata>,
    slots: HashMap<RequestID, Slot>,
}

impl BaseKvCacheManager {
    pub fn new(block_manager: KvBlockManager<BasicMetadata>) -> Self {
        Self {
            block_manager,
            slots: HashMap::new(),
        }
    }

    pub fn get_computed_blocks(
        &self,
        token: Tokens,
        salt_hash: SaltHash,
    ) -> Result<BlockList<DeviceStorage, BasicMetadata>, CacheManagerError> {
        unimplemented!()

        // Ok(self
        //     .block_manager
        //     .device()
        //     .unwrap()
        //     .match_sequence_hashes_blocking(sequence_hashes)?
        //     .into())
    }

    // pub fn create_slot(
    //     &mut self,
    //     request_id: RequestID,
    //     tokens: Tokens,
    //     salt_hash: SaltHash,
    // ) -> Result<(), CacheManagerError> {
    //     if self.slots.contains_key(&request_id) {
    //         return Err(CacheManagerError::SlotAlreadyExists(request_id));
    //     }

    //     let sequence =
    //         TokenBlockSequence::new(tokens, self.block_manager.page_size(), Some(salt_hash));

    //     let sequence_hashes = sequence
    //         .blocks()
    //         .iter()
    //         .map(|b| b.sequence_hash())
    //         .collect::<Vec<_>>();

    //     let sequence_blocks = self.get_computed_blocks(&sequence_hashes)?;

    //     let slot = Slot {
    //         sequence,
    //         sequence_blocks,
    //         inflight_blocks: Vec::new(),
    //     };

    //     assert!(self.slots.insert(request_id, slot).is_none());

    //     Ok(())
    // }

    // todo: this would be better if split into a create and update set of methods; work with vllm
    // to see if we can get this abstraction.
    // todo: vllm requires visiting this method in order to to enqueue cache offloading; however, this
    // would be better triggered by an event issued at the end of the .forward and before the sampling,
    // in this way, we can perform kv cache commits sooner and thus improve compute/control overlap.
    pub fn allocate_slots(&mut self, slot_update: SlotUpdate) -> Result<(), CacheManagerError> {
        let slot = self.slots.get(&slot_update.request_id);

        match (slot, slot_update.request_num_computed_tokens) {
            (None, 0) => self.create_slot(slot_update),
            (None, _) => {
                // if the external state is telling us that there should be computed tokens, but we
                // don't have a slot, then we have a problem.
                return Err(CacheManagerError::SlotNotFound(slot_update.request_id));
            }
            (Some(slot), _) => self.update_slot(slot_update),
        }
    }

    fn create_slot(&mut self, slot_update: SlotUpdate) -> Result<(), CacheManagerError> {
        unimplemented!()
    }

    fn update_slot(&mut self, slot_update: SlotUpdate) -> Result<(), CacheManagerError> {
        unimplemented!()
    }
}

#[derive(Debug)]
struct Slot {
    /// The number of tokens which have been computed and the kv state is populated.
    num_computed_tokens: usize,

    /// The blocks which have been computed and the kv state is populated.
    sequence_blocks: BlockList<DeviceStorage, BasicMetadata>,

    /// The blocks which are in flight and will be committed when the request is complete.
    inflight_blocks: Vec<MutableBlock<DeviceStorage, BasicMetadata>>,
}

pub struct SlotUpdate {
    /// The request ID.
    request_id: RequestID,

    /// External state about the number of computed tokens in the request.
    /// This should match the slots expectation.
    request_num_computed_tokens: usize,

    /// The number of new tokens which advances the sequence state.
    /// This is the number of tokens which will be computed in the near future.
    /// When [BaseKvCacheManager::update_slot] is called again, these tokens will be committed.
    num_new_tokens: usize,

    /// The number of new computed tokens in the request.
    /// The `num_new_tokens / block_size` should be equal to the length of the `new_computed_blocks`,
    /// it may have a remainder for the partial block state.
    num_new_computed_tokens: Option<usize>,

    /// The new computed blocks which advance the sequence state.
    new_computed_blocks: Option<BlockList<DeviceStorage, BasicMetadata>>,

    /// The number of lookahead blocks to cache.
    num_lookahead_blocks: Option<usize>,

    /// Whether to delay caching the blocks.
    delay_cache_blocks: Option<bool>,
}

pub struct ComputedBlocks {
    blocks: Vec<ImmutableBlock<DeviceStorage, BasicMetadata>>,
}

#[derive(Debug)]
enum KVCacheBlockType {
    Mutable(MutableBlock<DeviceStorage, BasicMetadata>),
    Immutable(ImmutableBlock<DeviceStorage, BasicMetadata>),
}

#[derive(Debug, Default)]
pub struct KVCacheBlocks {
    blocks: Vec<KVCacheBlockType>,
}

impl KVCacheBlocks {
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.blocks
            .iter()
            .map(|b| match b {
                KVCacheBlockType::Mutable(b) => b.block_id(),
                KVCacheBlockType::Immutable(b) => b.block_id(),
            })
            .collect()
    }

    // ///  def get_unhashed_block_ids(self) -> list[int]:
    // """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
    // return [
    //     block.block_id for block in self.blocks if block.block_hash is None
    // ]
    pub fn unhashed_block_ids(&self) -> Vec<BlockId> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                KVCacheBlockType::Mutable(b) => match b.state() {
                    BlockState::Reset => None,
                    BlockState::Partial(_) => Some(b.block_id()),
                    BlockState::Complete(_) | BlockState::Registered(_, _) => None,
                },
                KVCacheBlockType::Immutable(b) => None,
            })
            .collect()
    }
}

pub enum SlotState {
    Empty,
}

/// An ordered collection of blocks which are part of a sequence.
///
/// The root block is initialized with the `salt_hash` and consists of a series
/// of ordered immutable blocks followed by ordered mutable blocks.
///
/// Mutable blocks can be popped from the front of and registered with the block manager,
/// then added to the end (push back) of the commited sequence.
///
/// New blocks can be added to the ned of the mutable list.
pub struct KVCacheSlot<S: Storage> {
    salt_hash: SaltHash,
    commited: Vec<ImmutableBlock<S, BasicMetadata>>,
    mutable: VecDeque<MutableBlock<S, BasicMetadata>>,
}

// impl<S: Storage> KVCacheSlot<S> {
//     pub fn new(salt_hash: SaltHash) -> Self {
//         Self {
//             salt_hash,
//             commited: Vec::new(),
//             mutable: VecDeque::new(),
//         }
//     }

//     pub fn match_sequence_hashes(&self, sequence_hashes: &[SequenceHash]) -> Result<(), KVCacheSlotError> {

// }
