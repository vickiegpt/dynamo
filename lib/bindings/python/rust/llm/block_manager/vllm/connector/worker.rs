// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod slot;
use slot::{CreateEngineSlotRequest, EngineSlot, WorkerSlot};

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use super::*;
use crate::{llm::block_manager::distributed::VllmTensor, to_pyerr};
use dynamo_llm::block_manager::distributed::{
    BlockTransferRequest, ConnectorRequestLeader, ConnectorTransferType,
};
use dynamo_llm::block_manager::{
    connector::scheduler::{Scheduler, SchedulerMessage},
    storage::torch::{TorchDevice, TorchTensor},
};
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use dynamo_runtime::CancellationToken;
use tokio::task::JoinHandle;

enum EngineMessage {
    UpdateIteration(u64),

    /// Create a request slot with request id, counter and cancel token
    CreateEngineSlot(CreateEngineSlotRequest),

    /// Issue a request to the engine to complete a request and remove it
    RemoveEngineSlot(String),

    /// Trigger a layer to be completed
    UpdateLayersCompleted(String, u32),
    // /// Create a task with request id and the operation/task name
    // CreateTask(String, String),
}

#[pyclass]
pub struct KvConnectorWorker {
    request_slots: HashMap<String, WorkerSlot>,

    kv_caches: HashMap<String, Arc<VllmTensor>>,
    cancel_token: CancellationToken,

    /// Channel to send messages to the background engine
    /// We keep the touch points on the worker as small as possible to minimize impact on vllm
    engine_tx: tokio::sync::mpsc::UnboundedSender<EngineMessage>,

    /// Map of layer name to vllm tensor
    engine_task: CriticalTaskExecutionHandle,

    /// Map of request id to inflight load requests
    maybe_loading_finished: HashSet<String>,

    /// Map of request id to inflight finished requests
    maybe_storing_finished: HashSet<String>,

    /// Runtime for the engine task - update to DRT
    runtime: tokio::runtime::Runtime,

    bound: bool,
    iteration: u64,
    layers_complete: u32,
}

#[pymethods]
impl KvConnectorWorker {
    #[new]
    fn new(worker_id: String) -> PyResult<Self> {
        // ideally we initialize the DRT here, and pass it through
        // instead we'll create a cancellation token, but in the future get the token from
        // the DRT's primary lease
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(to_pyerr)?;

        let cancel_token = CancellationToken::new();
        let cancel_token_clone = cancel_token.clone();

        let (engine_tx, engine_rx) = tokio::sync::mpsc::unbounded_channel();

        let engine_task = CriticalTaskExecutionHandle::new_with_runtime(
            move |cancel_token| KvConnectorWorker::engine_task(cancel_token, engine_rx),
            cancel_token_clone,
            "kv-connector-engine-task",
            runtime.handle(),
        )
        .map_err(to_pyerr)?;

        tracing::info!(
            "KvConnectorWorker initialized with worker_id: {}",
            worker_id
        );

        Ok(Self {
            request_slots: HashMap::new(),
            kv_caches: HashMap::new(),
            cancel_token,
            engine_tx,
            engine_task,

            maybe_loading_finished: HashSet::new(),
            maybe_storing_finished: HashSet::new(),
            runtime,
            bound: false,
            iteration: 0,
            layers_complete: 0,
        })
    }

    pub fn register_kv_caches(&mut self, kv_caches: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        for (layer_name, torch_tensor) in kv_caches {
            let vllm_tensor = Arc::new(VllmTensor::new(torch_tensor).map_err(to_pyerr)?);
            tracing::trace!("Registering KV cache layer: {layer_name}, tensor: {vllm_tensor:?}");
            self.kv_caches.insert(layer_name, vllm_tensor);
        }

        let tensors: Vec<Arc<dyn TorchTensor>> = self
            .kv_caches
            .values()
            .map(|tensor| tensor.clone() as Arc<dyn TorchTensor>)
            .collect();

        let first_tensor = tensors.first().unwrap();
        tracing::debug!("kv tensor: {first_tensor:#?}");

        // validate all tensors are on the same device
        for tensor in &tensors {
            if tensor.device() != first_tensor.device() {
                return Err(to_pyerr(anyhow::anyhow!(
                    "All tensors must be on the same device! Got {:?} and {:?}",
                    tensor.device(),
                    first_tensor.device()
                )));
            }
        }

        // refactor john's kvbm worker into just pieces
        // - build a block layout from tensors, inferring the layout dims and dtype
        // - initialize nixl agent with uxc default, if g3 enabled add gds to the agent - do this in the constructor
        // - allocate blocks for g2 and/or g3, register them with nixl
        // - construct an transfer manager (offload manager)
        // - wait for worker to send a plan - a simply map of {request_id, [(src_block_id, dst_block_id]}
        //   - count layers, when all layers are counted, enqueue block-wise transfers and hold per request the notification handles
        //   - i think we have oneshot return notificatoin channel that is optional offloads/writes, if so use t.
        //
        // - currently leader and worker are not connected
        //   - rework the zmq bits or just use nats to perform completion events on leadera
        //   - this will allow the leader to free cpu/disk blocks when partial xfers are complete.
        //   - once wired up, the host will keep a list of transfer ids associated with each action it puts in the metdata
        //   - trigger the worker -> leader notification when the transfer is complete using the transfer id
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

        Ok(())
    }

    pub fn clear_connector_metadata(&mut self) {
        tracing::debug!(iteration = self.iteration, "clearing connector metadata");
        debug_assert!(self.bound, "connector metadata not bound");
        self.bound = false;
        self.iteration = 0; // always reset; leader drives the counter
        self.layers_complete = 0;
    }

    pub fn save_kv_layer(&mut self, layer_name: String, _kv_layer: Py<PyAny>) -> PyResult<()> {
        self.layers_complete += 1;

        self.engine_tx
            .send(EngineMessage::UpdateLayersCompleted(
                layer_name,
                self.layers_complete,
            ))
            .map_err(to_pyerr)?;

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
        // under each scenario, we mark the `maybe_loading_finished` and `maybe_storing_finished` hashsets with
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
                !self.maybe_storing_finished.contains(&request_id),
                "request already in maybe storing finished"
            );

            // insert request into the maybe finished set
            self.maybe_storing_finished.insert(request_id.clone());
        }

        // visit each request slot in the maybe finished set
        for request_id in self.maybe_storing_finished.iter() {
            let slot = self.request_slots.get(request_id).unwrap();
            if slot.is_finished_storing() {
                tracing::debug!(request_id, "request slot is finished");
                is_finished_storing.insert(request_id.clone());
            } else {
                tracing::debug!(request_id, "request slot is not finished");
            }
        }

        // remove the finished requests from the maybe finished set
        // note: when storing is finished we also remove the request from the engine state
        for request_id in &is_finished_storing {
            self.maybe_storing_finished.remove(request_id);

            // currently chomping the error as the engine is closed and we are shutting down
            let _ = self
                .engine_tx
                .send(EngineMessage::RemoveEngineSlot(request_id.clone()));
        }

        // visit each request slot in the maybe loading finished set to see if it is finished
        for request_id in self.maybe_loading_finished.iter() {
            let slot = self.request_slots.get(request_id).unwrap();
            if slot.is_finished_loading() {
                tracing::debug!(request_id, "request slot is finished");
                is_finished_loading.insert(request_id.clone());
            } else {
                tracing::debug!(request_id, "request slot is not finished");
            }
        }

        // remove the finished requests from the maybe finished set
        for request_id in &is_finished_loading {
            self.maybe_loading_finished.remove(request_id);
        }

        (is_finished_storing, is_finished_loading)
    }
}

impl KvConnectorWorker {
    fn create_worker_slot(&mut self, request_id: String) -> PyResult<()> {
        // create a child token which will cancel on the parent but can be cancelled individually
        // with out effecting the parent
        let token = self.cancel_token.child_token();

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
            }
        }
        Ok(())
    }
}

async fn task_monitor() {}

#[derive(Default)]
struct EngineState {
    slots: HashMap<String, EngineSlot>,
    iteration: u64,
    layers_complete: u32,
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
        tracing::debug!(
            iteration = self.iteration,
            layers_completed,
            "layer {last_layer_name} is complete"
        );
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
