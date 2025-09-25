// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod recorder;
pub mod slot;

use super::*;
use dynamo_llm::block_manager::metrics_kvbm::KvbmMetrics;
use dynamo_runtime::DistributedRuntime;
use slot::{ConnectorSlotManager, SlotError, SlotManager, SlotState};

use dynamo_runtime::DistributedRuntime as PyDistributedRuntime;
use crate::block_manager::BlockManagerBuilder;
use crate::block_manager::{
    VllmBlockManager, distributed::KvbmLeader as PyKvbmLeader, vllm::KvbmRequest,
    vllm::connector::leader::slot::VllmConnectorSlot,
};
use dynamo_runtime::metrics::prometheus_names::kvbm_connector;

use dynamo_llm::block_manager::{
    BasicMetadata, DiskStorage, ImmutableBlock, PinnedStorage,
    block::{
        data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
        locality::Logical,
    },
    connector::*,
};
use dynamo_llm::tokens::{SaltHash, TokenBlockSequence, Tokens};
use std::sync::{Arc, OnceLock};
use std::{collections::HashSet, sync::Mutex};
use tokio;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

type VllmLocality = Logical<DistributedLeaderWorkerResources>;

impl From<SlotError> for PyErr {
    fn from(err: SlotError) -> Self {
        to_pyerr(err)
    }
}
use anyhow;
use dynamo_llm::recorder::Recorder;
use tokio_util::sync::CancellationToken;

pub trait Leader: Send + Sync + std::fmt::Debug {
    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(usize, bool)>;

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()>;

    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>>;

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool>;

    fn has_slot(&self, request_id: String) -> bool;

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()>;

    fn slot_manager(&self) -> &ConnectorSlotManager<String>;
}

#[derive(Debug)]
pub struct KvConnectorLeader {
    slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
    block_size: usize,
    inflight_requests: HashSet<String>,
    onboarding_slots: HashSet<String>,
    iteration_counter: u64,
    kvbm_metrics: KvbmMetrics,
}

impl KvConnectorLeader {
    fn new(
        worker_id: String,
        drt: PyDistributedRuntime,
        page_size: usize,
        leader_py: PyKvbmLeader,
    ) -> Self {
        tracing::info!(
            "KvConnectorLeader initialized with worker_id: {}",
            worker_id
        );

        let leader = leader_py.get_inner().clone();
        let drt = drt.inner().clone();
        let handle: Handle = drt.runtime().primary();

        let ns = drt
            .namespace(kvbm_connector::KVBM_CONNECTOR_LEADER)
            .unwrap();

        let kvbm_metrics = KvbmMetrics::new(&ns);
        let kvbm_metrics_clone = kvbm_metrics.clone();

        let slot_manager_cell = Arc::new(OnceLock::new());
        let (leader_ready_tx, leader_ready_rx) = oneshot::channel::<String>();

        {
            let slot_manager_cell = slot_manager_cell.clone();

            handle.spawn(async move {
                let ready = leader.wait_worker_sync_ready().await;
                if !ready {
                    tracing::error!(
                        "KvConnectorLeader init aborted: leader worker barrier not ready!",
                    );
                    return;
                }

                let block_manager = match BlockManagerBuilder::new()
                    .worker_id(0)
                    .leader(leader_py)
                    .page_size(page_size)
                    .disable_device_pool(false)
                    .build()
                    .await
                {
                    Ok(bm) => bm,
                    Err(e) => {
                        tracing::error!("Failed to build BlockManager: {}", e);
                        return;
                    }
                };

                // Create the slot manager now that everything is ready
                let sm = ConnectorSlotManager::new(
                    block_manager.get_block_manager().clone(),
                    leader.clone(),
                    drt.clone(),
                    kvbm_metrics_clone.clone(),
                );

                let _ = slot_manager_cell.set(sm);

                // another barrier sync to make sure worker init won't return before leader is ready
                let _ = leader.run_leader_readiness_barrier_blocking(drt);

                if leader_ready_tx.send("finished".to_string()).is_err() {
                    tracing::error!("main routine receiver dropped before result was sent");
                }
            });
        }

        tokio::task::block_in_place(|| {
            handle.block_on(async {
                match leader_ready_rx.await {
                    Ok(_) => tracing::info!("KvConnectorLeader init complete."),
                    Err(_) => tracing::warn!("KvConnectorLeader init channel dropped"),
                }
            });
        });

        Self {
            slot_manager: slot_manager_cell,
            block_size: page_size,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            iteration_counter: 0,
            kvbm_metrics,
        }
    }
}

impl Leader for KvConnectorLeader {
    #[inline]
    fn slot_manager(&self) -> &ConnectorSlotManager<String> {
        self.slot_manager
            .get()
            .expect("slot_manager not initialized")
    }

    /// Match the tokens in the request with the available block pools.
    /// Note: the necessary details of the request are captured prior to this call. For vllm,
    /// we make a create slot call prior to this call, so a slot is guaranteed to exist.
    ///
    /// To align with the connector interface, we must ensure that if no blocks are matched, we return (0, false).
    /// In our implementation, if we match any block, we return (num_matched_tokens, true).
    #[tracing::instrument(level = "debug", skip(self, request_num_tokens, num_computed_tokens))]
    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(usize, bool)> {
        tracing::debug!(
            "request_num_tokens: {request_num_tokens}; num_computed_tokens: {num_computed_tokens}"
        );

        // the number of device matched tokens should be less than or equal to the number of tokens in the request
        debug_assert!(num_computed_tokens % self.block_size == 0);

        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        debug_assert!(
            slot.state() != SlotState::Prefilling && slot.state() != SlotState::Decoding,
            "slot is in the Prefilled state or Decoding; shouldn't happen"
        );

        if slot.state() == SlotState::SkippedPrefill || slot.state() == SlotState::SkippedDecode {
            tracing::debug!(
                "slot is in the SkippedPrefill or SkippedDecode state; will resume from skipped and return early"
            );
            match slot.state() {
                SlotState::SkippedPrefill => {
                    slot.mark_as_prefilling(self.iteration_counter)?;
                    return Ok((0, false));
                }
                SlotState::SkippedDecode => {
                    slot.mark_as_decoding(self.iteration_counter)?;
                    return Ok((0, false));
                }
                _ => unreachable!("slot is not in the SkippedPrefill or SkippedDecode state"),
            }
        }

        // early exit if we cannot match full block
        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            return Ok((0, false));
        }

        // find matches for any remaining tokens
        // this will advance the computed position and hold any newly matched blocks in the slot
        slot.acquire_local_matches(num_computed_tokens)?;

        // return the number of external tokens that are ready for onboarding
        // we always return true here as we always asynchronously onboard matched blocks
        if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
            debug_assert!((num_computed_tokens + num_external_tokens) % self.block_size == 0);
            tracing::debug!(
                request_id = request_id,
                "scheduling onboarding for {} external tokens",
                num_external_tokens
            );
            self.kvbm_metrics
                .matched_tokens
                .inc_by(num_external_tokens as u64);
            Ok((num_external_tokens, true))
        } else {
            Ok((0, false))
        }
    }

    /// Note: vLLM will not provide any scheduler output data for requests that are onboarding. it is entirely
    /// on the connector's implementation to handle this case.
    #[tracing::instrument(level = "debug", skip_all, fields(request_id))]
    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()> {
        tracing::debug!(
            request_id,
            "num_device_blocks: {}; num_external_tokens: {}",
            block_ids.len(),
            num_external_tokens
        );

        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        // we have not yet advanced the computed position, but now we can, since we have an indication that we have
        // necessary gpu blocks into which we will load the external tokens.

        slot.append_mutable_device_blocks(&block_ids)?;

        // the second call will show num_external_tokens == 0
        // this call is just letting us know the other blocks that are being used for the remainder of the prefill
        if num_external_tokens > 0 {
            let num_computed_tokens = block_ids.len() * self.block_size - num_external_tokens;
            slot.record_cached_device_tokens(num_computed_tokens);
            slot.advance_computed_position(num_computed_tokens)?;

            tracing::debug!(
                request_id = request_id,
                "triggering onboarding for {} external tokens",
                num_external_tokens
            );
            slot.trigger_onboarding(num_external_tokens)?;
            self.onboarding_slots.insert(request_id);
        }

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(iteration = self.iteration_counter + 1))]
    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>> {
        // the iteration counter is used to track the number of times we have built the connector metadata
        // all connetor operations have the iteration counter at which they were issued.
        // this allows operations to be lazily enqueued to the transfer engine
        // the worker side of the connector will track all operations for completion before the request is
        // allowed to be marked as finished.
        self.iteration_counter += 1;
        let iteration = self.iteration_counter;

        tracing::debug!("Building connector metadata");
        tracing::debug!("SchedulerOutput: {scheduler_output:#?}");

        let mut inflight_requests = self.inflight_requests.clone();
        let mut md = ConnectorMetadata::new(iteration);

        let onboarding_slots = std::mem::take(&mut self.onboarding_slots);

        // Worker-side - we create a request slot for onboarding, then delete it when onboarding is finished, then
        // recreate it again when we start the prefill/decode phase.
        //
        // This is kind of a nice abstraction as it keeps the events simplier; however, we now create the request-slot
        // once for onboarding (this loop), then again for prefill/decode (new_requests loop).
        for request_id in onboarding_slots.iter() {
            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            md.create_slot(request_id.clone());

            if let Some(pending_ops) = slot.take_pending_operations() {
                tracing::debug!("adding {} pending onboarding operations", pending_ops.len());
                md.add_operations(pending_ops);
            }

            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );
        }

        // vLLM provides us with "new_requests" which are "new" after onboarding, but not before or during.
        // this makes the lifecyle a potentially two-phase lifecycle.
        //
        // todo: update the code and abstraction to account for this two-phase lifecycle.
        for new_req in &scheduler_output.new_requests {
            let request_id = &new_req.request_id;
            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            // inform the worker that a new request-slot should be created
            md.create_slot(new_req.request_id.clone());

            slot.record_start_iteration(iteration)?;

            debug_assert!(
                matches!(
                    slot.state(),
                    SlotState::Initialized | SlotState::Onboarding(_)
                ),
                "current slot state: {:?}",
                slot.state()
            );

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(&[], &[], new_req.num_computed_tokens, scheduled_tokens)?;

            if let Some(pending_ops) = slot.take_pending_operations() {
                tracing::debug!(
                    "adding {} pending operations for slot {}",
                    pending_ops.len(),
                    new_req.request_id
                );
                md.add_operations(pending_ops);
            }
        }

        for cached_req in &scheduler_output.cached_requests {
            let request_id = &cached_req.request_id;

            if cached_req.resumed_from_preemption {
                // we really do not know what to expect here:
                // first let's try to get the slot, it might fail because maybe preemption put us thru
                // a finished cycle -- who knows
                let shared_slot = self.slot_manager().get_slot(request_id);
                match &shared_slot {
                    Ok(_) => {
                        tracing::info!("after preemption, slot is still alive");
                    }
                    Err(_) => {
                        tracing::info!("after preemption, slot is not alive");
                    }
                }

                let shared_slot = shared_slot?;
                let mut slot = shared_slot
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

                // todo: we probably need to reset the slot state and reload it from `cache_req`; however, we do not
                // know if it will take another pass at `get_num_new_matched_tokens` or `update_state_after_alloc`.
                slot.reset_after_preemption();

                // note, we can not trigger onboarding here -- perhaps we are supposed to or perhaps will get another
                // pass at `get_num_new_matched_tokens` or `update_state_after_alloc`.
            }

            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &cached_req.new_token_ids,
                &cached_req.new_block_ids,
                cached_req.num_computed_tokens,
                scheduled_tokens,
            )?;

            if let Some(pending_ops) = slot.take_pending_operations() {
                tracing::debug!(
                    "adding {} pending operations for slot {}",
                    pending_ops.len(),
                    request_id
                );
                md.add_operations(pending_ops);
            }
        }

        for unscheduled_req in inflight_requests.iter() {
            let shared_slot = self.slot_manager().get_slot(unscheduled_req)?;
            let mut slot_guard = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            let slot = slot_guard
                .as_any_mut()
                .downcast_mut::<VllmConnectorSlot>()
                .ok_or_else(|| anyhow::anyhow!("Expected VllmConnectorSlot, got different type"))?;

            slot.mark_as_skipped()?;
        }

        tracing::debug!("metadata: {md:#?}");
        serde_json::to_vec(&md)
            .map_err(|e| anyhow::anyhow!("Failed to serialize connector metadata: {}", e))
    }

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool> {
        tracing::debug!("Request finished: {request_id}; block_ids: {block_ids:?}");

        if !self.slot_manager().has_slot(&request_id) {
            tracing::warn!(
                "request_finished called for request_id: {request_id} but slot is not found"
            );
            self.inflight_requests.remove(&request_id);
            return Ok(false);
        }

        // grab the slot
        let shared_slot = self.slot_manager().get_slot(&request_id)?;

        // mark the slot as finished
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;
        slot.mark_as_finished(self.iteration_counter)?;

        // todo: allow the request to resolve when it should exit
        // the request may have some outstanding operations
        // we would like to inform it to shutdown, then have it signal to the work that is officially gone,
        // then we can remove the slot and trigger the worker to clean up as well.

        // remove the request from the inflight requests
        self.inflight_requests.remove(&request_id);

        // remove it from the manager as we will never use it again
        self.slot_manager().remove_slot(&request_id)?;

        // if the slot has finished, we can return false to vllm, indicating all gpu blocks are free to be reused
        // otherwise, we return true, which means there are still outstanding operations on gpu blocks which
        // must be awaited before the gpu blocks can be reused. if we return true, then it is the worker side
        // of the connector api which will be used to inform vllm that the request is finished.
        if let SlotState::Finished = slot.state() {
            Ok(false)
        } else {
            debug_assert!(matches!(slot.state(), SlotState::Finishing));
            Ok(true)
        }
    }

    fn has_slot(&self, request_id: String) -> bool {
        self.slot_manager().has_slot(&request_id)
    }

    /// Create a new slot for the given request ID.
    /// This is used to create a new slot for the request.
    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()> {
        self.slot_manager()
            .create_slot(&request.request_id, tokens, request.salt_hash)?;

        self.inflight_requests.insert(request.request_id);

        Ok(())
    }
}

#[pyclass]
pub struct PyKvConnectorLeader {
    connector_leader: Box<dyn Leader>,
}

#[pymethods]
impl PyKvConnectorLeader {
    #[new]
    #[pyo3(signature = (worker_id, drt, page_size, leader))]
    pub fn new(
        worker_id: String,
        drt: PyDistributedRuntime,
        page_size: usize,
        leader: PyKvbmLeader,
    ) -> Self {
        let enable_kvbm_record = std::env::var("ENABLE_KVBM_RECORD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let connector_leader: Box<dyn Leader> = if enable_kvbm_record {
            Box::new(recorder::KvConnectorLeaderRecorder::new(
                worker_id, drt, page_size, leader,
            ))
        } else {
            Box::new(KvConnectorLeader::new(worker_id, drt, page_size, leader))
        };
        Self { connector_leader }
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> PyResult<(usize, bool)> {
        self.connector_leader
            .get_num_new_matched_tokens(request_id, request_num_tokens, num_computed_tokens)
            .map_err(to_pyerr)
    }

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> PyResult<()> {
        self.connector_leader
            .update_state_after_alloc(request_id, block_ids, num_external_tokens)
            .map_err(to_pyerr)
    }

    fn build_connector_metadata(&mut self, scheduler_output: SchedulerOutput) -> PyResult<Vec<u8>> {
        self.connector_leader
            .build_connector_metadata(scheduler_output)
            .map_err(to_pyerr)
    }

    fn request_finished(&mut self, request_id: &str, block_ids: Vec<BlockId>) -> PyResult<bool> {
        self.connector_leader
            .request_finished(request_id.to_string(), block_ids)
            .map_err(to_pyerr)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.connector_leader.has_slot(request_id.to_string())
    }

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> PyResult<()> {
        self.connector_leader
            .create_slot(request, tokens)
            .map_err(to_pyerr)
    }
}
