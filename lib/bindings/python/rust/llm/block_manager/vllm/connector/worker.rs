// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use super::*;

#[derive(Debug, Clone)]
pub struct WorkerAction {
    pub action_type: WorkerActionType,
    // Add other fields as needed
}

impl WorkerAction {
    pub fn action_type(&self) -> WorkerActionType {
        self.action_type
    }
}

pub struct KvConnectorWorker {
    slots: HashMap<String, WorkerSlot>,

    /// Map of actions per layer, per slot
    forward_pass_actions: HashMap<String, HashMap<String, WorkerAction>>,
}

impl KvConnectorWorker {
    /// Loads the metadata from the leader.
    /// This action translates the metadata into a set of actions that the worker will perform.
    /// All actions much be assigned to a slot before [`KvConnectorWorker::clear_metadata`] is called.
    pub fn bind_metadata(&mut self, metadata: KvConnectorMetadata) {
        // build forward pass actions
        unimplemented!()
    }

    pub fn clear_metadata(&mut self) {
        assert!(
            self.forward_pass_actions.is_empty(),
            "All actions must be assigned to a slot before clearing metadata"
        );
    }

    /// This function serves two purposes:
    /// 1. To mark the slots as leader finished.
    /// 2. To report which slots have fully completed all their outstanding actions.
    ///
    /// The leader finish event can potentially trigger cancellation of best effort actions; however,
    /// all outstanding actions must be completed before the slot can report it has finished.
    ///
    /// If the implementation chooses to do so, it can trigger a cancellation token on a when provided
    /// a finished event, but this is not required.
    ///
    /// However, failure to cancel increases memory pressure on the GPU pool as there are no coarse grain
    /// API calls to release specific GPU blocks. Currently it appears it's an all or nothing approach
    /// with respect to GPU block ownership by the slot.
    pub fn get_finished(&mut self, finished_requests: &mut HashSet<String>) -> CompletedSlots {
        let mut completed_slots = CompletedSlots::default();
        let mut slots_to_remove = Vec::new();

        finished_requests.iter().for_each(|request| {
            if let Some(slot) = self.slots.get_mut(request) {
                slot.leader_finished = true;
            } else {
                panic!("Request not found in slots: {}", request);
            }
        });

        for (slot_id, slot) in self.slots.iter_mut() {
            if let Some(action_type) = slot.is_finished() {
                match action_type {
                    WorkerActionType::Loading => {
                        completed_slots
                            .recv_loading_requests
                            .insert(slot_id.clone());
                    }
                    WorkerActionType::Saving | WorkerActionType::Idle => {
                        // The leader must have already informed the worker that it is finished
                        // per the vllm protocol/policies.
                        assert!(slot.leader_finished);
                        completed_slots.send_saving_requests.insert(slot_id.clone());
                        slots_to_remove.push(slot_id.clone());
                    }
                }
            }
        }

        // Remove completed slots
        for slot_id in slots_to_remove {
            self.slots.remove(&slot_id);
        }

        completed_slots
    }
}

/// Workers can only be in one of these states.
///
/// If the worker is active loading, it can only accept [`WorkerActionType::Loading`] actions,
/// and if the worker is active saving, it can only accept [`WorkerActionType::Saving`] actions.
///
/// After all actions of a given type are completed the slot should transition to [`WorkerActionType::Idle`]
/// on during the next [`KvConnectorWorker::get_finished`] during a visit to the [`WorkerSlot::is_finished`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerActionType {
    Idle,
    Loading,
    Saving,
}

impl Default for WorkerActionType {
    fn default() -> Self {
        Self::Idle
    }
}

#[derive(Debug, Clone, Default)]
pub struct WorkerSlot {
    state: WorkerActionType,
    inflight_actions: Vec<WorkerAction>,
    pending_actions: HashMap<String, Vec<WorkerAction>>,
    leader_finished: bool,
}

impl WorkerSlot {
    pub fn is_finished(&self) -> Option<WorkerActionType> {
        // all pending actions must be triggered
        unimplemented!()
    }
}

#[derive(Debug, Clone, Default)]
pub struct CompletedSlots {
    send_saving_requests: HashSet<String>,
    recv_loading_requests: HashSet<String>,
}
