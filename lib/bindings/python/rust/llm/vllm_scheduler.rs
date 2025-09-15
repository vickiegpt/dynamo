// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the Rust vLLM scheduler integration.
//!
//! This module provides a thin PyO3 wrapper around the core scheduler
//! implementation in dynamo_llm::integrations::vllm::scheduler.

use pyo3::prelude::*;
use dynamo_llm::integrations::vllm::scheduler;

/// PyO3 wrapper around the core RustSchedulerState.
/// This is a thin wrapper that forwards all calls to the core implementation.
#[pyclass]
pub struct RustSchedulerState {
    /// The actual scheduler state from the core crate
    inner: scheduler::RustSchedulerState,
}

#[pymethods]
impl RustSchedulerState {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: scheduler::RustSchedulerState::new(),
        }
    }

    /// Add a new request to the Rust scheduler state.
    /// Called from Python when DynamoScheduler.add_request() is invoked.
    /// Note: cache_salt is now passed as a string and converted to hash in Rust.
    #[pyo3(signature = (request_id, prompt_token_ids, cache_salt=None, lora_int_id=None, priority=0, arrival_time=0.0))]
    pub fn add_request(
        &self,
        request_id: String,
        prompt_token_ids: Vec<i32>,
        cache_salt: Option<String>,
        lora_int_id: Option<i64>,
        priority: i32,
        arrival_time: f64,
    ) -> PyResult<()> {
        self.inner
            .add_request(
                request_id,
                prompt_token_ids,
                cache_salt,
                lora_int_id,
                priority,
                arrival_time,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Mark requests as finished without removing them.
    /// Called from Python when finish_requests is invoked externally.
    pub fn mark_as_finished(&self, request_ids: Vec<String>) -> PyResult<()> {
        self.inner
            .mark_as_finished(request_ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Remove finished requests from the Rust scheduler state.
    /// Called from Python when processing finished_req_ids in update_from_output.
    pub fn remove_finished_requests(&self, request_ids: Vec<String>) -> PyResult<()> {
        self.inner
            .remove_finished_requests(request_ids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Get the current number of tracked requests.
    pub fn num_requests(&self) -> usize {
        self.inner.num_requests()
    }

    /// Check if a request is being tracked.
    pub fn has_request(&self, request_id: &str) -> bool {
        self.inner.has_request(request_id)
    }

    /// Get all currently tracked request IDs (for debugging).
    pub fn get_request_ids(&self) -> Vec<String> {
        self.inner.get_request_ids()
    }
}