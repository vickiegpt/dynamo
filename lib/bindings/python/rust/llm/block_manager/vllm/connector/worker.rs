// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod slot;
use slot::{CreateEngineSlotRequest, EngineSlot, WorkerSlot};

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use super::*;
use crate::llm::block_manager::distributed::get_barrier_id;
use crate::llm::block_manager::vllm::connector::worker::slot::WorkerSlotState;
use crate::{
    llm::block_manager::distributed::VllmTensor, to_pyerr,
    DistributedRuntime as PyDistributedRuntime,
};

use dynamo_llm::block_manager::distributed::{
    BlockTransferHandler, BlockTransferPool, BlockTransferRequest, KvbmWorker, KvbmWorkerConfig,
};
use dynamo_llm::block_manager::storage::torch::TorchTensor;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use dynamo_runtime::{CancellationToken, DistributedRuntime};

enum EngineMessage {
    /// Update the iteration count
    UpdateIteration(u64),

    /// Trigger a layer to be completed
    UpdateLayersCompleted(String, u32),

    /// Create a request slot with request id, counter and cancel token
    CreateEngineSlot(CreateEngineSlotRequest),

    /// Issue a request to the engine to complete a request and remove it
    RemoveEngineSlot(String),

    /// Register the transfer engine with the worker
    RegisterTransferEngine(tokio::sync::oneshot::Receiver<BlockTransferHandler>),

    /// Enqueue a block transfer request
    EnqueueBlockTransfer(ConnectorOperation),
}

#[pyclass]
pub struct KvConnectorWorker {
    drt: DistributedRuntime,
    kvbm_worker: OnceLock<KvbmWorker>,

    request_slots: HashMap<String, WorkerSlot>,

    kv_caches: HashMap<String, Arc<VllmTensor>>,

    /// Channel to send messages to the background engine
    /// We keep the touch points on the worker as small as possible to minimize impact on vllm
    engine_tx: tokio::sync::mpsc::UnboundedSender<EngineMessage>,

    /// Map of layer name to vllm tensor
    engine_task: CriticalTaskExecutionHandle,

    /// Map of request id to inflight load requests
    maybe_finished_onboarding: HashSet<String>,

    /// Map of request id to inflight finished requests
    maybe_finished_offloading: HashSet<String>,

    /// For now, offloading operations will be enqueued at the end of the forward pass
    offloading_operations: Vec<ConnectorOperation>,

    bound: bool,
    iteration: u64,
    layers_complete: usize,
}

#[pymethods]
impl KvConnectorWorker {
    #[new]
    fn new(py_drt: PyDistributedRuntime, vllm_worker_id: String) -> PyResult<Self> {
        let drt = py_drt.inner.clone();
        let runtime = drt.runtime().primary();

        let (engine_tx, engine_rx) = tokio::sync::mpsc::unbounded_channel();

        let engine_task = CriticalTaskExecutionHandle::new_with_runtime(
            move |cancel_token| KvConnectorWorker::engine_task(cancel_token, engine_rx),
            drt.primary_token(),
            "kv-connector-engine-task",
            &runtime,
        )
        .map_err(to_pyerr)?;

        tracing::info!(
            "KvConnectorWorker initialized with worker_id: {}",
            vllm_worker_id
        );

        Ok(Self {
            drt,
            kvbm_worker: OnceLock::new(),
            request_slots: HashMap::new(),
            kv_caches: HashMap::new(),
            engine_tx,
            engine_task,
            maybe_finished_onboarding: HashSet::new(),
            maybe_finished_offloading: HashSet::new(),
            offloading_operations: Vec::new(),
            bound: false,
            iteration: 0,
            layers_complete: 0,
        })
    }

    /// Registers the KV caches with the KVBM worker.
    ///
    /// The Dynamo KVBM worker is lazily initialized when the first KV cache is registered.
    /// This process establishes a connection between all KVBM workers and the leader.
    pub fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: HashMap<String, Py<PyAny>>,
    ) -> PyResult<()> {
        if self.kvbm_worker.get().is_some() {
            tracing::warn!("kvbm worker already registered");
            return Ok(());
        }

        // TODO: pass in the sorted (layer_name, tensor) such that the order of the list matches the order of layer execution in the model
        for (layer_name, torch_tensor) in kv_caches {
            let vllm_tensor = Arc::new(VllmTensor::new(torch_tensor).map_err(to_pyerr)?);
            tracing::trace!("Registering KV cache layer: {layer_name}, tensor: {vllm_tensor:?}");
            self.kv_caches.insert(layer_name, vllm_tensor);
        }

        let vllm_tensors: Vec<Arc<dyn TorchTensor>> = self
            .kv_caches
            .values()
            .map(|tensor| tensor.clone() as Arc<dyn TorchTensor>)
            .collect();

        let config = KvbmWorkerConfig::builder()
            .drt(self.drt.clone())
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .tensors(vllm_tensors)
            .device_id(device_id)
            .dtype_width_bytes(dtype_width_bytes)
            .barrier_id(get_barrier_id())
            .build()
            .map_err(to_pyerr)?;

        let mut worker = self
            .drt
            .runtime()
            .primary()
            .block_on(async move {
                let worker = KvbmWorker::new(config).await?;
                anyhow::Ok(worker)
            })
            .map_err(to_pyerr)?;

        // the block transfer handler is being initialized in the background
        let block_transfer_handler = worker
            .block_transfer_handler_rx()
            .ok_or(anyhow::anyhow!("block transfer handler not available"))
            .map_err(to_pyerr)?;

        // pass this oneshot receiver to the engine task which can await on its completion
        self.engine_tx
            .send(EngineMessage::RegisterTransferEngine(
                block_transfer_handler,
            ))
            .map_err(to_pyerr)?;

        self.kvbm_worker
            .set(worker)
            .map_err(|_| to_pyerr(anyhow::anyhow!("failed to set kvbm worker")))?;

        Ok(())
    }

    /// Loads the metadata from the leader.
    /// This action translates the metadata into a set of actions that the worker will perform.
    /// All actions much be assigned to a slot before [`KvConnectorWorker::clear_metadata`] is called.
    pub fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> PyResult<()> {
        debug_assert!(!self.bound, "connector metadata already bound");
        let metadata: ConnectorMetadata = serde_json::from_slice(&metadata).map_err(to_pyerr)?;
        self.bound = true;
        self.iteration = metadata.iteration;
        self.layers_complete = 0;
        tracing::debug!(
            iteration = self.iteration,
            "bound new metadata: {metadata:#?}"
        );

        self.engine_tx
            .send(EngineMessage::UpdateIteration(self.iteration))
            .map_err(to_pyerr)?;

        // local actions
        // - create a request slot for each new request
        // - for each action in the metadata, add the action to the request slot
        // - send the list of actions to the engine to track completion

        for slot in metadata.new_slots {
            debug_assert!(
                !self.request_slots.contains_key(&slot),
                "slot already exists"
            );
            self.create_worker_slot(slot)?;
        }

        let mut onboarding_operations = Vec::new();
        let mut offloading_operations = Vec::new();

        for operation in metadata.operations {
            tracing::debug!(
                request_id = operation.req_id, operation_id = %operation.uuid,
                "adding operation to slot: {operation:#?}"
            );

            if operation.xfer_req.to_pool == BlockTransferPool::Device {
                onboarding_operations.push(operation);
            } else if operation.xfer_req.from_pool == BlockTransferPool::Device {
                offloading_operations.push(operation);
            }
        }

        // immediately enqueue the onboarding operations
        for operation in onboarding_operations {
            let request_id = operation.req_id.clone();
            let slot = self.request_slots.get_mut(&request_id).unwrap();
            debug_assert!(matches!(
                slot.get_state(),
                WorkerSlotState::Initialized | WorkerSlotState::Onboarding
            ));
            slot.set_state(WorkerSlotState::Onboarding);
            slot.add_operation(&operation);
            self.engine_tx
                .send(EngineMessage::EnqueueBlockTransfer(operation))
                .map_err(to_pyerr)?;
            self.maybe_finished_onboarding.insert(request_id);
        }

        // delay offloading operations until the end of the forward pass
        debug_assert!(
            self.offloading_operations.is_empty(),
            "offloading operations already enqueued"
        );
        self.offloading_operations = offloading_operations;

        Ok(())
    }

    pub fn clear_connector_metadata(&mut self) {
        tracing::debug!(iteration = self.iteration, "clearing connector metadata");
        debug_assert!(self.bound, "connector metadata not bound");
        self.bound = false;
        self.iteration = 0; // always reset; leader drives the counter
        self.layers_complete = 0;
    }

    pub fn save_kv_layer(&mut self, _layer_name: String, _kv_layer: Py<PyAny>) -> PyResult<()> {
        self.layers_complete += 1;

        // self.engine_tx
        //     .send(EngineMessage::UpdateLayersCompleted(
        //         layer_name,
        //         self.layers_complete,
        //     ))
        //     .map_err(to_pyerr)?;

        if self.layers_complete == self.kv_caches.len() {
            let offloading_operations = std::mem::take(&mut self.offloading_operations);
            for operation in offloading_operations {
                let request_id = operation.req_id.clone();
                let slot = self.request_slots.get_mut(&request_id).unwrap();
                debug_assert!(matches!(
                    slot.get_state(),
                    WorkerSlotState::Initialized | WorkerSlotState::Offloading
                ));
                slot.set_state(WorkerSlotState::Offloading);
                slot.add_operation(&operation);
                self.engine_tx
                    .send(EngineMessage::EnqueueBlockTransfer(operation))
                    .map_err(to_pyerr)?;
                self.maybe_finished_offloading.insert(request_id);
            }
        }

        Ok(())
    }

    pub fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        tracing::debug!(
            iteration = self.iteration,
            "Getting finished requests: {finished_requests:?}"
        );

        // we do not have to visit every slot on every pass, just slots we are waiting on
        //
        // there are two conditions where we would be waiting:
        // 1. if we have requested a load, we need to wait for it to complete
        //    - the load request would come in via the metadata this is processsed in the bind
        // 2. if we have requested a finished event, then we need to await for all outstanding
        //    operations to complete -- either by finishing or being cancelled
        //    - the finish request is triggered by this function, it is not seen in the metadata
        //
        // under each scenario, we mark the `maybe_loading_finished` and `maybe_finished_offloading` hashsets with
        // the request id
        //
        // on each forward pass we visit the maybe slots to see if they are finished

        let mut is_finished_storing = HashSet::new();
        let mut is_finished_loading = HashSet::new();

        for request_id in finished_requests {
            tracing::debug!(request_id, "marking request as finished");

            debug_assert!(
                self.request_slots.contains_key(&request_id),
                "request slot not found"
            );

            debug_assert!(
                !self.maybe_finished_offloading.contains(&request_id),
                "request already in maybe storing finished"
            );

            // insert request into the maybe finished set
            self.maybe_finished_offloading.insert(request_id.clone());
        }

        // visit each request slot in the maybe finished set
        for request_id in self.maybe_finished_offloading.iter() {
            let slot = self.request_slots.get(request_id).unwrap();
            debug_assert!(matches!(
                slot.get_state(),
                WorkerSlotState::Initialized | WorkerSlotState::Offloading
            ));
            if slot.all_tasks_completed() {
                tracing::debug!(request_id, "request slot is finished");
                is_finished_storing.insert(request_id.clone());
            } else {
                tracing::debug!(request_id, "request slot is not finished");
            }
        }

        // remove the finished requests from the maybe finished set
        // note: when storing is finished we also remove the request from the engine state
        for request_id in &is_finished_storing {
            self.maybe_finished_offloading.remove(request_id);

            // currently chomping the error as the engine is closed and we are shutting down
            let _ = self
                .engine_tx
                .send(EngineMessage::RemoveEngineSlot(request_id.clone()));
        }

        // visit each request slot in the maybe finished set to see if it is finished
        for request_id in self.maybe_finished_onboarding.iter() {
            let slot = self.request_slots.get_mut(request_id).unwrap();
            debug_assert!(slot.get_state() == &WorkerSlotState::Onboarding);
            if slot.all_tasks_completed() {
                tracing::debug!(request_id, "request slot is finished");
                is_finished_loading.insert(request_id.clone());
                slot.set_state(WorkerSlotState::Initialized);
            } else {
                tracing::debug!(request_id, "request slot is not finished");
            }
        }

        // remove the finished requests from the maybe finished set
        for request_id in &is_finished_loading {
            self.maybe_finished_onboarding.remove(request_id);
        }

        (is_finished_storing, is_finished_loading)
    }
}

impl KvConnectorWorker {
    fn create_worker_slot(&mut self, request_id: String) -> PyResult<()> {
        // create a child token which will cancel on the parent but can be cancelled individually
        // with out effecting the parent
        let token = self.drt.primary_token().child_token();

        // create a request slot with the child token
        // this will be the local worker slot
        let slot = WorkerSlot::new(token);
        let request = slot.make_engine_slot_request(request_id.clone());

        // insert the slot into the local worker slots map
        self.request_slots.insert(request_id, slot);

        // send a request to insert the slot into the engine state
        self.engine_tx
            .send(EngineMessage::CreateEngineSlot(request))
            .map_err(to_pyerr)?;
        Ok(())
    }

    async fn engine_task(
        _cancel_token: CancellationToken,
        mut engine_rx: tokio::sync::mpsc::UnboundedReceiver<EngineMessage>,
    ) -> anyhow::Result<()> {
        let mut state = EngineState::default();
        while let Some(message) = engine_rx.recv().await {
            match message {
                EngineMessage::UpdateIteration(new_iteration) => {
                    state.update_iteration(new_iteration);
                }
                EngineMessage::CreateEngineSlot(request) => {
                    state.add_slot(request, state.iteration);
                }
                EngineMessage::RemoveEngineSlot(request_id) => {
                    state.remove_slot(request_id);
                }
                EngineMessage::UpdateLayersCompleted(last_layer_name, layers_completed) => {
                    state.update_layers_completed(last_layer_name, layers_completed);
                }
                EngineMessage::RegisterTransferEngine(block_transfer_handler_rx) => {
                    tracing::debug!("awaiting block transfer handler");
                    let block_transfer_handler = block_transfer_handler_rx.await?;
                    state.block_transfer_handler = Some(Arc::new(block_transfer_handler));
                }
                EngineMessage::EnqueueBlockTransfer(operation) => {
                    state.enqueue_block_transfer(operation);
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct EngineState {
    slots: HashMap<String, EngineSlot>,
    iteration: u64,
    layers_complete: u32,
    block_transfer_handler: Option<Arc<BlockTransferHandler>>,
}

impl EngineState {
    fn add_slot(&mut self, req: CreateEngineSlotRequest, created_at: u64) {
        let request_id = req.request_id.clone();
        debug_assert!(!self.slots.contains_key(&request_id), "slot already exists");
        tracing::debug!(request_id, "engine state adding slot");
        self.slots
            .insert(request_id, EngineSlot::new(req, created_at));
    }

    fn remove_slot(&mut self, request_id: String) {
        debug_assert!(self.slots.contains_key(&request_id), "slot not found");
        self.slots.remove(&request_id);
        tracing::debug!(request_id, "engine state removing slot");
    }

    fn update_iteration(&mut self, iteration: u64) {
        self.iteration = iteration;
        tracing::debug!(iteration, "engine state updating iteration");
    }

    fn update_layers_completed(&mut self, last_layer_name: String, layers_completed: u32) {
        self.layers_complete = layers_completed;
        tracing::trace!(
            iteration = self.iteration,
            layers_completed,
            "layer {last_layer_name} is complete"
        );
    }

    fn enqueue_block_transfer(&mut self, operation: ConnectorOperation) {
        debug_assert!(
            self.block_transfer_handler.is_some(),
            "block transfer handler not registered"
        );
        let slot = self.slots.get(&operation.req_id).unwrap();
        let completed_counter = IncrementOnDrop::new(slot.completed.clone());

        let block_transfer_handler = self.block_transfer_handler.as_ref().unwrap().clone();

        // add in a scheduler and concurency limiter

        tokio::spawn(async move {
            tracing::debug!(
                request_id = operation.req_id,
                "executing block transfer for request_id"
            );
            if let Err(e) = block_transfer_handler
                .execute_transfer(operation.xfer_req)
                .await
            {
                panic!(
                    "failed to execute block transfer for request_id {}: {e:#?}",
                    operation.req_id
                );
            }

            tracing::debug!(
                request_id = operation.req_id,
                "block transfer for request_id completed"
            );

            drop(completed_counter);
        });
    }
}

pub struct IncrementOnDrop(Arc<AtomicU64>);

impl IncrementOnDrop {
    fn new(counter: Arc<AtomicU64>) -> Self {
        Self(counter)
    }
}

impl Drop for IncrementOnDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
