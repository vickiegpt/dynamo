// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Connector Protocol
//!
//! This module defines the messages used to communicate between the following components:
//! - Leader -> TransferEngine (block_manager::distributed)
//! - TransferEngine -> Scheduler
//! - Worker -> Scheduler
//!
//! ## Locality
//!
//! The TransferEngine, Scheduler and Worker are all guaranteed to be in the same process. `Scheduler`
//! is a per-worker scheduler and `TransferEngine` is also a per-worker component.
//!
//! ## Connector Operations
//!
//! There a two types of connector operations: load operations and store operations. The following must
//! be true:
//! - All loads must be initiated when the Slot is in the [`SlotState::Initialized`] state.
//! - While the slot is in the [`SlotState::OnboardStaged`] or the [`SlotState::Onboarding`] state,
//!   no active tokens can be scheduled, no stores can be issued.
//!   - Uknowns:
//!     - What happens on cancellation?
//! - To transition to the [`SlotState::Prefilling`] state, the slot must be in either the [`SlotState::Initialized`]
//!   [`SlotState::NotScheduled`], or [`SlotState::OnboardStaged`] state.
//!   - When in the [`SlotState::Prefilling`] state, store/save operations are allowed.
//!   - Store/Save operations are determined when processing the [`SchedulerOutput`].
//!   - If a store operation is issued, the following will happen:
//!     - Leader will trigger a message to the TransferEngine with the use StoreRequest and a ConnectorStoreRequest
//!     - The presence of the ConnectorStoreRequest will trigger the TransferEngine to request a SchedulerStoreRequest,
//!       this will block the transfer engine's store task from executing until released by the scheduler.
//!     - The Scheduler will not release the store task until the Worker has made sufficient progress, i.e. the data is
//!       to be stored has been computed and in device memory.
//!     - All leader slots are visited on each build metadata step, this allows for any leader initiated actions to be
//!       included in the metadata sent to the worker.
//!       - An operation must include: request_id, the iteration on which it was issued, the operation type, and a descriptor.
//!     - The Worker will pick up all operations from the leader's metadata and enqueue to the scheduler.
//!     - The Worker will issue notifications to the Scheduler at the start of each iteration and the completion of each
//!       layer in that iteration.
//!     - For an operation to be scheduled to run, the following must be true:
//!       - The TransferEngine must have registered the operation with the Scheduler.
//!       - The Worker must have registered the operation with the Scheduler.
//!       - Sufficient progress, either layer-wise or iteration-wise, must have been made.
//!     - For an operation to run, the following must be true:
//!       - The operation must be in the scheduled queue.
//!       - A concurrent token must be acquired.
//!     - A running operation will be monitored by a task awaiting a completion event.
//!       - When the completion event is received, the atomic completion counter will be incremented.
//!
//!
//! All transfer requests are triggered by the leader based on the details in the [`SchedulerOutput`].
//!
//! [`SchedulerOutput`] is transform

use std::sync::atomic::AtomicU64;

use super::scheduler::SchedulingDecision;
use super::*;

use tokio::sync::oneshot;

pub type LayerName = String;
pub type LayerIndex = u32;
pub type Iteration = u64;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum TransferType {
    Load,
    Store,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum SchedulerRequirement {
    IterationComplete(Iteration),

    /// The layer with the provided name and iteration counter must be complete.
    LayerNameComplete(LayerName, Iteration),

    /// The layer index and iteration counter must be complete.
    LayerComplete(LayerIndex, Iteration),
}

/// Issued by the leader, received by the TransferEngine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderTransferRequest {
    pub request_id: String,
    pub uuid: uuid::Uuid,
    pub requirement: SchedulerRequirement,
    pub transfer_type: TransferType,
}

/// Issued by the TransferEngine, received by the Scheduler.
/// Note: In order to be considered for scheduling, the [`TransferScheduleRequest`] and the [`WorkerTransferRequest`]
/// for the same operation (uuid) must be present on the scheduler.
pub struct TransferScheduleRequest {
    pub leader_request: LeaderTransferRequest,
    pub response_tx: oneshot::Sender<ScheduledTaskHandle>,
}

pub struct ScheduledTaskHandle {
    pub request_id: String,
    pub uuid: uuid::Uuid,
    pub transfer_type: TransferType,
    pub decision_rx: oneshot::Receiver<SchedulingDecision>,
    pub completion_handle: TransferCompletionHandle,
    pub cancel_token: CancellationToken,
}

/// Recived by the Worker, forward to the Scheduler.
/// In ordered to be considered for scheduling, both the [`TransferScheduleRequest`] and the [`WorkerTransferRequest`]
/// must be present on the scheduler.
///
/// Note: No response is required. The Worker holds an atomic counter for each oepration type. The expected count (local/non-atomic)
/// is incremented on receiving a request. The Worker knows all operations are complete when the shared atomic counter matches the
/// expected count.
///
/// Workers can not handle errors, they only deal with counters. All operations (which can be cancelled) must completed for a Worker
/// to mark the request_id as complete.
///
/// Scheduler requirements are only provided by the leader initiated transfer request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerTransferRequest {
    pub request_id: String,
    pub uuid: uuid::Uuid,
    pub transfer_type: TransferType,
}

/// Sent by Worker to Scheduler.
/// Combines [`WorkerTransferRequest`] and [`WorkerRequestState`] and issues a [`WorkerSchedulerRequest`]
///
/// This object has all the links to the worker to track completion and observe any cancellation signals.
pub struct WorkerSchedulerRequest {
    pub request_id: String,
    pub uuid: uuid::Uuid,
    pub transfer_type: TransferType,
    pub cancel_token: CancellationToken,
    pub completion_handle: TransferCompletionHandle,
}

// /// One-time use completion handle. Should only be triggered once after the operation is complete and the memory
// /// being targetting can be reused elsewhere.
// pub struct TransferCompletionHandle(Option<Arc<AtomicU64>>);

// impl TransferCompletionHandle {
//     pub(crate) fn new(counter: Arc<AtomicU64>) -> Self {
//         Self(Some(counter))
//     }

//     pub fn mark_as_complete(&mut self) {
//         if let Some(counter) = self.0.take() {
//             counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//         }
//     }
// }

// impl Drop for TransferCompletionHandle {
//     fn drop(&mut self) {
//         if let Some(counter) = self.0.take() {
//             tracing::error!("increment on drop dropped without being marked as complete - this could lead silent data corruption");
//             counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//         }
//     }
// }

pub enum CompletionStatus {
    Ok,
    Err(String),
    Cancelled,
}

pub struct TransferCompletionHandle(Option<oneshot::Sender<CompletionStatus>>);

impl TransferCompletionHandle {
    pub(crate) fn new(status_tx: oneshot::Sender<CompletionStatus>) -> Self {
        Self(Some(status_tx))
    }

    pub fn mark_as_success(mut self) {
        if let Some(status_tx) = self.0.take() {
            if status_tx.send(CompletionStatus::Ok).is_err() {
                tracing::error!(
                    "failed to send completion status; this could lead to silent data corruption"
                );
            }
        }
    }

    pub fn mark_as_error(mut self, error: String) {
        if let Some(status_tx) = self.0.take() {
            if status_tx.send(CompletionStatus::Err(error)).is_err() {
                tracing::error!(
                    "failed to send completion status; this could lead to silent data corruption"
                );
            }
        }
    }

    pub fn mark_as_cancelled(mut self) {
        if let Some(status_tx) = self.0.take() {
            if status_tx.send(CompletionStatus::Cancelled).is_err() {
                tracing::error!(
                    "failed to send completion status; this could lead to silent data corruption"
                );
            }
        }
    }
}

impl Drop for TransferCompletionHandle {
    fn drop(&mut self) {
        if let Some(status_tx) = self.0.take() {
            if status_tx
                .send(CompletionStatus::Err(
                    "transfer dropped without being explicitly marked as complete, error or cancelled".to_string(),
                ))
                .is_err()
            {
                tracing::error!(concat!(
                    "logic error: implementation failed to respect the [TransferCompletionHandle] policy; ",
                    "handle dropped with being explicitly marked; this may lead to data corruption of the ",
                    "handle was dropped while a transfer was still in progress; please report immediately."
                ));
            }
        }
    }
}
