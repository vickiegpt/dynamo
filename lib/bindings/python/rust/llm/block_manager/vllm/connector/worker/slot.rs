// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::block_manager::distributed::BlockTransferPool;

use super::*;

pub struct CreateEngineSlotRequest {
    pub request_id: String,
    pub completed: Arc<AtomicU64>,
    pub cancel_token: CancellationToken,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerSlotState {
    Initialized,
    Onboarding,
    Offloading,
}

#[derive(Debug, Clone)]
pub struct WorkerSlot {
    operations: Vec<uuid::Uuid>,
    completed: Arc<AtomicU64>,
    cancel_token: CancellationToken,
    state: WorkerSlotState,
}

impl WorkerSlot {
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self {
            operations: vec![],
            completed: Arc::new(AtomicU64::new(0)),
            cancel_token,
            state: WorkerSlotState::Initialized,
        }
    }

    pub fn get_state(&self) -> &WorkerSlotState {
        &self.state
    }

    pub fn set_state(&mut self, state: WorkerSlotState) {
        self.state = state;
    }

    pub fn add_operation(&mut self, operation: &ConnectorOperation) {
        // if operation.xfer_req.to_pool == BlockTransferPool::Device {
        //     debug_assert!(self.state == WorkerSlotState::Onboarding);
        // } else if operation.xfer_req.from_pool == BlockTransferPool::Device {
        //     debug_assert!(self.state == WorkerSlotState::Offloading);
        // }
        self.operations.push(operation.uuid);
        tracing::debug!(
            request_id = operation.req_id,
            operation_id = %operation.uuid,
            "adding operation to slot: total operations: {}; completed: {}",
            self.operations.len(),
            self.num_completed()
        );
    }

    pub fn num_inflight(&self) -> usize {
        self.operations.len()
    }

    pub fn num_completed(&self) -> usize {
        self.completed.load(std::sync::atomic::Ordering::Relaxed) as usize
    }

    pub fn all_tasks_completed(&self) -> bool {
        self.num_inflight() == self.num_completed()
    }

    pub fn make_engine_slot_request(&self, request_id: String) -> CreateEngineSlotRequest {
        CreateEngineSlotRequest {
            request_id,
            completed: self.completed.clone(),
            cancel_token: self.cancel_token.clone(),
        }
    }
}

pub struct EngineSlot {
    request_id: String,
    pub cancel_token: CancellationToken,
    pub completed: Arc<AtomicU64>,
    created_at: u64,
}

impl EngineSlot {
    pub fn new(req: CreateEngineSlotRequest, created_at: u64) -> Self {
        tracing::debug!(
            request_id = req.request_id,
            iteration = created_at,
            "creating engine slot"
        );
        Self {
            request_id: req.request_id,
            cancel_token: req.cancel_token,
            completed: req.completed,
            created_at,
        }
    }
}
