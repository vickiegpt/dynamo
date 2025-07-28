use std::collections::HashSet;
use std::sync::atomic::AtomicU64;

use super::protocol::*;
use super::*;

use tokio::sync::mpsc;

const DISCONNECTED_WARNING: &str =
    "runtime error: connections between components were lost; likely tearing down";

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("runtime error: connections between components were lost; likely tearing down")]
    Disconnected,
}

pub enum SchedulingDecision {
    Execute,
    Cancel,
}

/// A client for the scheduler. One-time use. Capture a clone per task.
#[derive(Clone)]
pub struct TransferSchedulerClient {
    scheduler_tx: mpsc::Sender<TransferScheduleRequest>,
}

impl TransferSchedulerClient {
    pub fn new(scheduler_tx: mpsc::Sender<TransferScheduleRequest>) -> Self {
        Self { scheduler_tx }
    }

    /// If the [SchedulingDecision::Execute] is returned, the caller receives a completion handle.
    /// The completion handle be marked as completed after the
    ///
    /// If the [SchedulingDecision::Cancel] is returned, the transfer is cancelled and the completion handle
    /// must not be dropped.
    #[tracing::instrument(level = "debug", skip_all, fields(request_id = %request.request_id, operation_id = %request.uuid))]
    pub async fn schedule_transfer(
        self,
        request: LeaderTransferRequest,
    ) -> Option<TransferCompletionHandle> {
        let (response_tx, response_rx) = oneshot::channel();
        let tx = self.scheduler_tx.clone();

        let request = TransferScheduleRequest {
            leader_request: request,
            response_tx,
        };

        if tx.send(request).await.is_err() {
            tracing::warn!(DISCONNECTED_WARNING);
            return None;
        }

        let handle = match response_rx.await {
            Ok(response) => response,
            Err(_) => {
                tracing::warn!(DISCONNECTED_WARNING);
                return None;
            }
        };

        tokio::select! {
            maybe_decision = handle.decision_rx =>  {
                match maybe_decision {
                    Ok(SchedulingDecision::Execute) => {
                        Some(handle.completion_handle)
                    }
                    Ok(SchedulingDecision::Cancel) => {
                        None
                    }
                    Err(_) => {
                        tracing::warn!(DISCONNECTED_WARNING);
                        handle.completion_handle.mark_as_error("connection to scheduler dropped".to_string());
                        None
                    }
                }
            }
            _ = handle.cancel_token.cancelled() => {
                tracing::debug!(
                    "transfer was explicitly cancelled via the cancel token"
                );
                handle.completion_handle.mark_as_cancelled();
                None
            }
        }
    }
}

pub struct WorkerSchedulerClient {
    slots: HashMap<String, WorkerSchedulerClientSlot>,
    scheduler_tx: mpsc::UnboundedSender<SchedulerMessage>,
    cancel_token: CancellationToken,
}

#[derive(Debug, Clone)]
pub struct WorkerSchedulerClientSlot {
    load_operations: HashSet<uuid::Uuid>,
    load_completed: Arc<AtomicU64>,
    store_operations: HashSet<uuid::Uuid>,
    store_completed: Arc<AtomicU64>,
    cancel_token: CancellationToken,
}

impl WorkerSchedulerClientSlot {
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self {
            load_operations: HashSet::new(),
            load_completed: Arc::new(AtomicU64::new(0)),
            store_operations: HashSet::new(),
            store_completed: Arc::new(AtomicU64::new(0)),
            cancel_token,
        }
    }

    fn make_scheduler_slot_request(&self, request_id: String) -> SchedulerCreateSlotDetails {
        SchedulerCreateSlotDetails {
            request_id,
            load_completed: self.load_completed.clone(),
            store_completed: self.store_completed.clone(),
            cancel_token: self.cancel_token.clone(),
        }
    }
}

impl WorkerSchedulerClient {
    pub fn create_slot(&mut self, request_id: String) -> Result<(), SchedulerError> {
        // create a child token which will cancel on the parent but can be cancelled individually
        // with out effecting the parent
        let token = self.cancel_token.child_token();

        // create a request slot with the child token
        // this will be the local worker slot
        let slot = WorkerSchedulerClientSlot::new(token);
        let request = slot.make_scheduler_slot_request(request_id.clone());

        // insert the slot into the local worker slots map
        self.slots.insert(request_id, slot);

        // send a request to insert the slot into the engine state
        self.scheduler_tx
            .send(SchedulerMessage::CreateSlot(request))
            .map_err(|_| SchedulerError::Disconnected)?;
        Ok(())
    }

    /// Enqueues a request to the scheduler.
    ///
    /// Both the worker client and the scheduler keep track of outstanding requests.
    /// The atomic counter to mark completion is shared, but only incremented by the scheduler.
    pub fn enqueue_request(&mut self, request: WorkerTransferRequest) {
        debug_assert!(
            self.slots.contains_key(&request.request_id),
            "slot does not exist"
        );

        let slot = self.slots.get_mut(&request.request_id).unwrap();

        match request.transfer_type {
            TransferType::Load => {
                slot.load_operations.insert(request.uuid);
            }
            TransferType::Store => {
                slot.store_operations.insert(request.uuid);
            }
        };

        if self
            .scheduler_tx
            .send(SchedulerMessage::EnqueueRequest(request))
            .is_err()
        {
            tracing::error!("connection to scheduler dropped; cancelling all transfers");
            slot.cancel_token.cancel();
        }
    }
}

pub type Iteration = u64;
pub type LayerName = String;
pub type LayerIndex = u32;

pub enum SchedulerMessage {
    /// Issued by worker to create a shared request state between worker and scheduler
    CreateSlot(SchedulerCreateSlotDetails),

    /// Enqueue a worker requested operation to the scheduler, this is one-half of the necessary
    /// bits to enqueu the operation. The other half is leader driven and propagated to the scheduler
    /// via the [TransferScheduleRequest]
    EnqueueRequest(WorkerTransferRequest),

    /// Issued at the start of a forward pass iteration
    StartIteration(Iteration),

    /// Issued at the end of a forward pass iteration, with the iteration number
    EndIteration(Iteration),

    /// Issued by the leader to update the number of layers completed
    UpdateLayersCompleted(LayerName, LayerIndex),

    /// Worker received a notification that the given request id has been completed.
    ///
    RequestFinished(String),
}

pub struct Scheduler {
    slots: HashMap<String, SchedulerSlot>,
    worker_rx: mpsc::UnboundedReceiver<SchedulerMessage>,
    transfer_rx: mpsc::UnboundedReceiver<TransferScheduleRequest>,
    iteration: u64,
    layers_complete: u32,
}

impl Scheduler {
    async fn step(&mut self) -> bool {
        tokio::select! {
            maybe_worker_msg = self.worker_rx.recv(), if !self.worker_rx.is_closed() => {
                match maybe_worker_msg {
                    Some(SchedulerMessage::StartIteration(new_iteration)) => {
                        self.start_iteration(new_iteration);
                    }
                    Some(SchedulerMessage::EndIteration(iteration)) => {
                        self.end_iteration(iteration);
                    }
                    Some(SchedulerMessage::CreateSlot(request)) => {
                        self.add_slot(request, self.iteration);
                    }
                    // Some(SchedulerMessage::RemoveRequestSlot(request_id)) => {
                    //     self.remove_slot(request_id);
                    // }
                    Some(SchedulerMessage::UpdateLayersCompleted(last_layer_name, layers_completed)) => {
                        self.update_layers_completed(last_layer_name, layers_completed);
                    }
                // Some(SchedulerMessage::EnqueueRequest(request)) => {
                //         self.enqueue_request(request);
                //     }
                //     Some(SchedulerMessage::RequestFinished(request_id)) => {
                //         self.request_finished(request_id);
                //     }
                    _ => {
                        panic!("received unexpected message from worker");
                    }
                    None => {
                        return false;
                    }
                }

             }
            maybe_transfer_msg = self.transfer_rx.recv() => { }
        }
        true
    }

    fn add_slot(&mut self, req: SchedulerCreateSlotDetails, created_at: u64) {
        let request_id = req.request_id.clone();
        debug_assert!(!self.slots.contains_key(&request_id), "slot already exists");
        tracing::debug!(
            request_id,
            iteration = self.iteration,
            "engine state adding slot"
        );
        self.slots
            .insert(request_id, SchedulerSlot::new(req, created_at));
    }

    fn remove_slot(&mut self, request_id: String) {
        debug_assert!(self.slots.contains_key(&request_id), "slot not found");
        self.slots.remove(&request_id);
        tracing::debug!(
            request_id,
            iteration = self.iteration,
            "engine state removing slot"
        );
    }

    fn start_iteration(&mut self, iteration: u64) {
        tracing::debug!(iteration, "engine state updating iteration");
        self.iteration = iteration;
        self.layers_complete = 0;
    }

    fn end_iteration(&mut self, iteration: u64) {
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

pub struct SchedulerCreateSlotDetails {
    pub request_id: String,
    pub load_completed: Arc<AtomicU64>,
    pub store_completed: Arc<AtomicU64>,
    pub cancel_token: CancellationToken,
}

pub struct SchedulerSlot {
    request_id: String,
    cancel_token: CancellationToken,
    load_completed: Arc<AtomicU64>,
    store_completed: Arc<AtomicU64>,
    created_at: u64,
}

impl SchedulerSlot {
    fn new(req: SchedulerCreateSlotDetails, created_at: u64) -> Self {
        Self {
            request_id: req.request_id,
            load_completed: req.load_completed,
            store_completed: req.store_completed,
            cancel_token: req.cancel_token,
            created_at,
        }
    }
}
