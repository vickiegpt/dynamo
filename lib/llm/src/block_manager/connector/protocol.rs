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

use super::*;
