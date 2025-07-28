// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Debug, thiserror::Error)]
pub enum SlotError {
    #[error("slot not found")]
    NotFound,

    #[error("slot is in an invalid state: {0}")]
    InvalidState(String),

    #[error("slot operation failed: {0}")]
    InvalidOperation(String),

    #[error(transparent)]
    BlockPoolError(#[from] BlockPoolError),
}

pub trait SlotManager<R: RequestKey>: Send + Sync {
    type SlotType: Slot + ?Sized;

    fn has_slot(&self, request_id: &R) -> bool;

    /// Create a new slot for the given request ID, initial tokens and salt hash.
    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError>;

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError>;
    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// The slot was not scheduled in the previous iteration.
    Initialized,

    /// The slot was previously scheduled, but not in the last iteration.
    NotScheduled,

    /// The slot is prepared to load kv blocks from external storage; however, the onboarding operation
    /// has not been triggered yet. The usize is the number of tokens that are ready for onboarding.
    OnboardStaged(usize),

    /// The slot is actively copying blocks to device storage from some external storage(s).
    /// The u64 is the iteration at which the onboarding operation was triggered.
    Onboarding(u64),

    /// The slot is actively prefilling the sequence.
    Prefilling,

    /// The slot is actively participating in a forward pass which will result in one more more tokens
    /// to be applied to the sequence.
    Decoding,

    /// The slot is marked as finished, but not all resources have been released.
    Finishing,

    /// The slot is finished and all resources have been released.
    Finished,
}

pub trait Slot: std::fmt::Debug {
    fn state(&self) -> SlotState;

    fn sequence(&self) -> &TokenBlockSequence;

    /// The number of tokens that have been computed on the device, i.e. the number of tokens for which we have ownership
    /// of computed kv blocks in the device storage.
    fn computed_tokens(&self) -> usize;

    fn mark_as_scheduled(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_prefilling(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_decoding(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_onboarding(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_not_scheduled(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_finished(&mut self, iteration: u64) -> Result<(), SlotError>;

    /// The number of device blocks that have been allocated to the slot.
    fn num_device_blocks_allocated(&self) -> usize;

    /// Find all possible block matches for remaining known tokens in some local storage, i.e. look up and take ownership
    /// of any kv blocks for tokens in the isl that are not already in memory on the device, but on some local storage.
    ///
    /// If external tokens are matched, then the slot will transition to the [`SlotState::Onboarding`] state.
    fn acquire_all_local_matches(&mut self) -> Result<(), SlotError>;

    /// Take all pending operations for the slot.
    fn take_pending_operations(&mut self) -> Vec<String>;
}

pub trait ExternallyManagedDeviceSlot: Slot {
    /// Since we do not control the device pool, nor do we have insight in how the device pool is managed,
    /// we must accept external updates to the computed position.
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError>;

    /// Append the given block ids to the slot.
    ///
    /// The external device block manager has provided a set of mutable blocks to the slot.
    fn append_mutable_device_blocks(&mut self, block_ids: Vec<BlockId>) -> Result<(), SlotError>;
}

pub struct ConnectorSlotManager<R: RequestKey> {
    slots: Mutex<HashMap<R, Arc<Mutex<VllmConnectorSlot>>>>,
    block_manager: VllmBlockManager,
}

impl<R: RequestKey> ConnectorSlotManager<R> {
    pub fn new(block_manager: VllmBlockManager) -> Self {
        tracing::debug!(
            "creating slot manager with block size: {}",
            block_manager.block_size()
        );
        Self {
            slots: Mutex::new(HashMap::new()),
            block_manager,
        }
    }
}

impl<R: RequestKey> SlotManager<R> for ConnectorSlotManager<R> {
    type SlotType = dyn ExternallyManagedDeviceSlot;

    fn has_slot(&self, request_id: &R) -> bool {
        self.slots.lock().unwrap().contains_key(request_id)
    }

    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError> {
        let slot = VllmConnectorSlot::new(tokens.into(), salt_hash, self.block_manager.clone());
        self.slots
            .lock()
            .unwrap()
            .insert(request_id.clone(), Arc::new(Mutex::new(slot)));
        Ok(())
    }

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError> {
        let slots = self.slots.lock().unwrap();
        let slot = slots.get(request_id).ok_or(SlotError::NotFound)?;
        Ok(slot.clone())
    }

    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError> {
        self.slots.lock().unwrap().remove(request_id);
        Ok(())
    }
}

pub struct VllmConnectorSlot {
    /// The state of the slot.
    state: SlotState,

    /// Current position in the sequence of tokens that have been computed.
    /// When the slot is initialized, we populate the sequence with the prefill tokens.
    /// However, those tokens are not yet prefilled, so they are not yet represented
    /// in the sequence_position.
    computed_position: usize,

    /// The sequence of token blocks
    sequence: TokenBlockSequence,

    /// The immutable blocks id (device)
    immutable: Vec<usize>,

    /// The mutable blocks id (device)
    mutable: VecDeque<usize>,

    /// Blocks to be onboarded from the host
    /// We must hold these blocks in the slot state until the scheduler trigger the onboarding.
    staging_from_host: Option<Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>>,

    /// Blocks to be onboarded from the disk
    /// We must hold these blocks in the slot state until the scheduler trigger the onboarding.
    staging_from_disk: Option<Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>>,

    /// Blocks to be offloaded to the host
    /// We must hold these blocks in the slot state until the scheduler trigger the offloading.
    offload_to_host: Vec<MutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>,

    /// The number of blocks cached from the device
    blocks_cached_from_device: usize,

    /// The number of blocks cached from the host
    blocks_cached_from_host: usize,

    /// The number of blocks cached from the disk
    blocks_cached_from_disk: usize,

    /// Phantom data to ensure the storage type is correct.
    block_manager: VllmBlockManager,

    block_size: usize,

    iteration_first_scheduled: Option<u64>,

    pending_operations: Vec<String>,
}

impl VllmConnectorSlot {
    pub fn new(tokens: Tokens, salt_hash: SaltHash, block_manager: VllmBlockManager) -> Self {
        assert!(!tokens.is_empty(), "tokens must be non-empty");
        let block_size = block_manager.block_size();
        debug_assert!(block_size.is_power_of_two() && block_size <= 1024);
        let sequence = TokenBlockSequence::new(tokens, block_size as u32, Some(salt_hash));

        Self {
            sequence,
            block_manager,
            block_size,

            // default values
            state: SlotState::Initialized,
            iteration_first_scheduled: None,
            computed_position: 0,
            immutable: Vec::new(),
            mutable: VecDeque::new(),
            staging_from_host: None,
            staging_from_disk: None,
            offload_to_host: Vec::new(),
            pending_operations: Vec::new(),
            blocks_cached_from_device: 0,
            blocks_cached_from_host: 0,
            blocks_cached_from_disk: 0,
        }
    }
}

impl std::fmt::Debug for VllmConnectorSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VllmConnectorSlot")
            .field("state", &self.state)
            .field("computed_position", &self.computed_position)
            .field("num_tokens", &self.sequence.total_tokens())
            .finish()
    }
}

impl Slot for VllmConnectorSlot {
    fn state(&self) -> SlotState {
        self.state
    }

    fn mark_as_prefilling(&mut self, iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Prefilling;

        // we can now aggress

        Ok(())
    }

    fn mark_as_decoding(&mut self, iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Decoding;
        Ok(())
    }

    fn mark_as_onboarding(&mut self, iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Onboarding(iteration);
        Ok(())
    }

    fn mark_as_scheduled(&mut self, iteration: u64) -> Result<(), SlotError> {
        if self.iteration_first_scheduled.is_none() {
            self.iteration_first_scheduled = Some(iteration);
        }
        Ok(())
    }

    fn mark_as_not_scheduled(&mut self, _iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::NotScheduled;
        Ok(())
    }

    fn mark_as_finished(&mut self, _iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Finishing;
        Ok(())
    }

    fn sequence(&self) -> &TokenBlockSequence {
        &self.sequence
    }

    fn computed_tokens(&self) -> usize {
        self.computed_position
    }

    fn num_device_blocks_allocated(&self) -> usize {
        self.immutable.len() + self.mutable.len()
    }

    fn take_pending_operations(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_operations)
    }

    #[tracing::instrument(level = "debug", skip_all)]
    fn acquire_all_local_matches(&mut self) -> Result<(), SlotError> {
        if !matches!(self.state(), SlotState::Initialized) {
            return Err(SlotError::InvalidOperation(format!(
                "slot must be in the NotScheduled state to acquire local matches; got {:?}",
                self.state()
            )));
        }

        let block_size = self.block_manager.block_size();
        let num_computed_tokens = self.computed_tokens();
        let num_computed_blocks = num_computed_tokens / block_size;
        debug_assert!(num_computed_tokens % block_size == 0);

        let sequence_hashes = self
            .sequence()
            .blocks()
            .iter()
            .skip(num_computed_blocks)
            .map(|b| b.sequence_hash())
            .collect::<Vec<_>>();

        tracing::debug!("matching against {} block hashes", sequence_hashes.len());

        // we should do this opportunistically after this operation is done
        // ideally it was triggered by the match_sequence_hashes_blocking calls directly

        // if let Some(host) = self.block_manager.host() {
        //     host.touch_blocks_blocking(&sequence_hashes)?;
        // }

        // if let Some(disk) = self.block_manager.disk() {
        //     disk.touch_blocks_blocking(&sequence_hashes)?;
        // }

        // we start matching non-device blocks after the device blocks
        let search_offset = num_computed_blocks;

        let mut host_blocks = self
            .block_manager
            .host()
            .map(|host| host.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_host_blocks = host_blocks.len();

        // advance the search offset by the number of matched host blocks
        let search_offset = search_offset + num_matched_host_blocks;

        // start at host offset
        let mut disk_blocks = self
            .block_manager
            .disk()
            .map(|disk| disk.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_disk_blocks = disk_blocks.len();

        let num_matched_blocks = num_matched_host_blocks + num_matched_disk_blocks;

        tracing::debug!(
            "matched {} host blocks and {} disk blocks; {} total blocks",
            num_matched_host_blocks,
            num_matched_disk_blocks,
            num_matched_blocks
        );

        // early exit if we did not match any blocks
        if num_matched_blocks == 0 {
            return Ok(());
        }

        let mut num_new_matched_tokens = num_matched_blocks * block_size;

        // we are on a block boundary, so we need to throw away the last block
        if num_computed_tokens + num_new_matched_tokens == self.sequence().total_tokens() {
            tracing::debug!("on a block boundary, throwing away the last block");

            // we should have matched at least one block
            assert!(!host_blocks.is_empty() || !disk_blocks.is_empty());

            // pop from disk, or if there are none, then from host
            if disk_blocks.is_empty() {
                host_blocks.pop();
            } else {
                disk_blocks.pop();
            }

            // decrement the number of new matched tokens by the block size
            num_new_matched_tokens -= block_size;
        }

        self.staging_from_host = Some(host_blocks);
        self.staging_from_disk = Some(disk_blocks);
        self.state = SlotState::OnboardStaged(num_new_matched_tokens);

        Ok(())
    }
}

impl ExternallyManagedDeviceSlot for VllmConnectorSlot {
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError> {
        if self.computed_position + num_tokens > self.sequence().total_tokens() {
            return Err(SlotError::InvalidOperation(format!(
                "cannot advance computed position by {num_tokens} tokens, total tokens is {}",
                self.sequence().total_tokens()
            )));
        }

        self.computed_position += num_tokens;
        Ok(())
    }

    fn append_mutable_device_blocks(&mut self, block_ids: Vec<BlockId>) -> Result<(), SlotError> {
        tracing::debug!(
            "appending {} mutable device blocks to slot; total device blocks: {}",
            block_ids.len(),
            self.num_device_blocks_allocated()
        );
        self.mutable.extend(block_ids);
        Ok(())
    }
}
