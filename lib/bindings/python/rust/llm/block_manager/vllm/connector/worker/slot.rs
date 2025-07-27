// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub struct CreateEngineSlotRequest {
    pub request_id: String,
    pub load_completed: Arc<AtomicU64>,
    pub store_completed: Arc<AtomicU64>,
    pub cancel_token: CancellationToken,
}

#[derive(Debug, Clone)]
pub struct WorkerSlot {
    load_operations: Vec<String>,
    load_completed: Arc<AtomicU64>,
    store_operations: Vec<String>,
    store_completed: Arc<AtomicU64>,
    cancel_token: CancellationToken,
}

impl WorkerSlot {
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self {
            load_operations: vec![],
            load_completed: Arc::new(AtomicU64::new(0)),
            store_operations: vec![],
            store_completed: Arc::new(AtomicU64::new(0)),
            cancel_token,
        }
    }

    pub fn add_load_operation(&mut self, operation: String) {
        self.load_operations.push(operation);
    }

    pub fn add_store_operation(&mut self, operation: String) {
        self.store_operations.push(operation);
    }

    pub fn num_inflight_loads(&self) -> usize {
        self.load_operations.len()
    }

    pub fn num_inflight_stores(&self) -> usize {
        self.store_operations.len()
    }

    pub fn num_completed_loads(&self) -> usize {
        self.load_completed
            .load(std::sync::atomic::Ordering::Relaxed) as usize
    }

    pub fn num_completed_stores(&self) -> usize {
        self.store_completed
            .load(std::sync::atomic::Ordering::Relaxed) as usize
    }

    pub fn is_finished_storing(&self) -> bool {
        self.num_inflight_stores() == self.num_completed_stores()
    }

    pub fn is_finished_loading(&self) -> bool {
        self.num_inflight_loads() == self.num_completed_loads()
    }

    pub fn make_engine_slot_request(&self, request_id: String) -> CreateEngineSlotRequest {
        CreateEngineSlotRequest {
            request_id,
            load_completed: self.load_completed.clone(),
            store_completed: self.store_completed.clone(),
            cancel_token: self.cancel_token.clone(),
        }
    }
}

pub struct EngineSlot {
    request_id: String,
    cancel_token: CancellationToken,
    load_completed: Arc<AtomicU64>,
    store_completed: Arc<AtomicU64>,
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
            load_completed: req.load_completed,
            store_completed: req.store_completed,
            created_at,
        }
    }
}
