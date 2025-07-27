pub mod slot;

use super::*;
use slot::{ConnectorSlotManager, SlotError, SlotManager, SlotState};

use crate::llm::block_manager::BlockManager as PyBlockManager;
use crate::llm::block_manager::{vllm::KvbmRequest, VllmBlockManager};

use dynamo_llm::block_manager::{
    block::{
        data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
        locality::Logical,
    },
    connector::*,
    BasicMetadata, DiskStorage, ImmutableBlock, PinnedStorage,
};
use dynamo_llm::tokens::{SaltHash, TokenBlockSequence, Tokens};

use std::collections::HashSet;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

type VllmLocality = Logical<DistributedLeaderWorkerResources>;

impl From<SlotError> for PyErr {
    fn from(err: SlotError) -> Self {
        to_pyerr(err)
    }
}

#[pyclass]
pub struct KvConnectorLeader {
    slot_manager: ConnectorSlotManager<String>,
    block_manager: PyBlockManager,
    block_size: usize,
    inflight_requests: HashSet<String>,
    iteration_counter: u64,
}

#[pymethods]
impl KvConnectorLeader {
    #[new]
    #[pyo3(signature = (worker_id, block_manager))]
    pub fn new(worker_id: String, block_manager: PyBlockManager) -> Self {
        tracing::info!(
            "KvConnectorLeader initialized with worker_id: {}",
            worker_id
        );

        let block_size = block_manager.get_block_manager().block_size();

        Self {
            slot_manager: ConnectorSlotManager::new(block_manager.get_block_manager().clone()),
            block_manager,
            block_size,
            inflight_requests: HashSet::new(),
            iteration_counter: 0,
        }
    }

    /// Match the tokens in the request with the available block pools.
    /// Note: the necessary details of the request are captured prior to this call. For vllm,
    /// we make a create slot call prior to this call, so a slot is guaranteed to exist.
    ///
    /// To align with the connector interface, we must ensure that if no blocks are matched, we return (0, false).
    /// In our implementation, if we match any block, we return (num_matched_tokens, true).
    #[tracing::instrument(level = "debug", skip(self, request_num_tokens, num_computed_tokens))]
    pub fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> PyResult<(usize, bool)> {
        tracing::debug!(
            "request_num_tokens: {request_num_tokens}; num_computed_tokens: {num_computed_tokens}"
        );

        // the number of device matched tokens should be less than or equal to the number of tokens in the request
        debug_assert!(num_computed_tokens % self.block_size == 0);

        let shared_slot = self.slot_manager.get_slot(&request_id).map_err(to_pyerr)?;
        let mut slot = shared_slot.lock().map_err(to_pyerr)?;

        // vllm is telling us that the tokens have been computed, since we do not have insight into the device pool
        // we accept this and advance the computed position
        slot.advance_computed_position(num_computed_tokens)?;

        // early exit if we cannot match full block
        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            return Ok((0, false));
        }

        // find matches for any remaining tokens
        // this will advance the computed position and hold any newly matched blocks in the slot
        slot.acquire_all_local_matches()?;

        // return the number of external tokens that are ready for onboarding
        // we always return true here as we always asynchronously onboard matched blocks
        if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
            debug_assert!((num_computed_tokens + num_external_tokens) % self.block_size == 0);
            Ok((num_external_tokens, true))
        } else {
            Ok((0, false))
        }
    }

    /// We drop the need to pass in the KvCacheBlocks and the num_external_tokens as they are captured
    /// statefully in the [`VllmLeaderKvCacheManagerAndConnector::get_num_new_matched_tokens`] function.
    #[tracing::instrument(level = "debug", skip_all, fields(request_id))]
    pub fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: u64,
    ) -> PyResult<()> {
        tracing::debug!(
            request_id,
            "num_device_blocks: {}; num_external_tokens: {}",
            block_ids.len(),
            num_external_tokens
        );

        let shared_slot = self.slot_manager.get_slot(&request_id).map_err(to_pyerr)?;
        let mut slot = shared_slot.lock().map_err(to_pyerr)?;

        slot.append_mutable_device_blocks(block_ids)?;

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all)]
    pub fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> PyResult<Vec<u8>> {
        // the iteration counter is used to track the number of times we have built the connector metadata
        // all connetor operations have the iteration counter at which they were issued.
        // this allows operations to be lazily enqueued to the transfer engine
        // the worker side of the connector will track all operations for completion before the request is
        // allowed to be marked as finished.
        self.iteration_counter += 1;
        let iteration = self.iteration_counter;

        tracing::debug!("Building connector metadata; iteration {iteration}");
        tracing::trace!("{scheduler_output:#?}");

        let mut inflight_requests = self.inflight_requests.clone();
        let mut md = ConnectorMetadata::new(iteration);

        for new_req in &scheduler_output.new_requests {
            let request_id = &new_req.request_id;
            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager.get_slot(request_id).map_err(to_pyerr)?;
            let mut slot = shared_slot.lock().map_err(to_pyerr)?;

            md.create_slot(new_req.request_id.clone());
            slot.mark_as_scheduled(iteration)?;

            debug_assert!(
                matches!(
                    slot.state(),
                    SlotState::Initialized | SlotState::OnboardStaged(_)
                ),
                "current slot state: {:?}",
                slot.state()
            );

            if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
                tracing::debug!(
                    request_id,
                    iteration,
                    "initializing onboarding of {num_external_tokens} tokens"
                );
                slot.mark_as_onboarding(iteration)?;
                continue;
            }

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            if scheduled_tokens == 0 {
                tracing::debug!(
                    request_id,
                    iteration,
                    "no tokens scheduled; mark as not scheduled"
                );
                slot.mark_as_not_scheduled(iteration)?;
            } else if new_req.num_computed_tokens < slot.sequence().total_tokens() {
                tracing::debug!(request_id, iteration, "mark as prefilling");
                slot.mark_as_prefilling(iteration)?;
            } else {
                tracing::debug!(request_id, iteration, "mark as decoding");
                slot.mark_as_decoding(iteration)?;
            }

            md.add_operations(
                request_id.clone(),
                iteration,
                slot.take_pending_operations(),
            );
        }

        for cached_req in &scheduler_output.cached_requests {
            let request_id = &cached_req.request_id;
            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );
        }

        tracing::debug!("scheduler_output: {scheduler_output:#?}");
        serde_json::to_vec(&md).map_err(to_pyerr)
    }

    pub fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> PyResult<bool> {
        tracing::debug!("Request finished: {request_id}; block_ids: {block_ids:?}");
        // grab the slot
        let shared_slot = self.slot_manager.get_slot(&request_id).map_err(to_pyerr)?;

        // mark the slot as finished
        let mut slot = shared_slot.lock().map_err(to_pyerr)?;
        slot.mark_as_finished(self.iteration_counter)?;

        // remove it from the manager as we will never use it again
        self.slot_manager.remove_slot(&request_id)?;

        // if the slot has finished, we can return false to vllm, indicating all gpu blocks are free to be reused
        // otherwise, we return false, which means there are still outstanding operations on gpu blocks which
        // must be awaited before the gpu blocks can be reused. if we return true, then it is the worker side
        // of the connector api which will be used to inform vllm that the request is finished.
        if let SlotState::Finished = slot.state() {
            Ok(false)
        } else {
            debug_assert!(matches!(slot.state(), SlotState::Finishing));
            Ok(true)
        }
    }

    pub fn has_slot(&self, request_id: String) -> bool {
        self.slot_manager.has_slot(&request_id)
    }

    /// Create a new slot for the given request ID.
    /// This is used to create a new slot for the request.
    pub fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> PyResult<()> {
        self.slot_manager
            .create_slot(&request.request_id, tokens, request.salt_hash)?;

        self.inflight_requests.insert(request.request_id);

        Ok(())
    }
}

impl KvConnectorLeader {
    #[inline(always)]
    pub fn block_manager(&self) -> &VllmBlockManager {
        self.block_manager.get_block_manager()
    }
}
