// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the SchedulerRecorder

use dynamo_llm::integrations::vllm::recorder::SchedulerRecorder as RustRecorder;
use dynamo_llm::integrations::vllm::types::*;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Python-accessible SchedulerRecorder
#[pyclass(name = "SchedulerRecorder")]
pub struct PySchedulerRecorder {
    inner: RustRecorder,
}

#[pymethods]
impl PySchedulerRecorder {
    /// Create a new SchedulerRecorder
    #[new]
    #[pyo3(signature = (model, vllm_version))]
    fn new(model: String, vllm_version: String) -> Self {
        Self {
            inner: RustRecorder::new(model, vllm_version),
        }
    }

    /// Record a scheduler output (already converted to Rust)
    fn record_schedule_output(&mut self, output: SchedulerOutput) -> PyResult<()> {
        self.inner.record_schedule_output(output);
        Ok(())
    }

    /// Record a model runner output (already converted to Rust)
    fn record_model_runner_output(&mut self, output: ModelRunnerOutput) -> PyResult<()> {
        self.inner.record_model_runner_output(output);
        Ok(())
    }

    /// Record engine core outputs (already converted to Rust)
    fn record_engine_core_outputs(&mut self, outputs: EngineCoreOutputs) -> PyResult<()> {
        self.inner.record_engine_core_outputs(outputs);
        Ok(())
    }

    /// Move to the next iteration
    fn next_iteration(&mut self) -> PyResult<()> {
        self.inner.next_iteration();
        Ok(())
    }

    /// Get the current iteration number
    fn current_iteration(&self) -> u64 {
        self.inner.current_iteration()
    }

    /// Save the recording to a JSON file
    fn save_to_file(&mut self, path: String) -> PyResult<()> {
        let path = PathBuf::from(path);
        self.inner
            .save_to_file(&path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))
    }

    /// Clear all recordings
    fn clear(&mut self) -> PyResult<()> {
        self.inner.clear();
        Ok(())
    }

    /// Get the number of recorded iterations
    fn num_iterations(&self) -> usize {
        self.inner.get_trace().iterations.len()
    }
}

/// Load a recording from a JSON file
#[pyfunction]
pub fn load_scheduler_trace(path: String) -> PyResult<SchedulerTrace> {
    let path = PathBuf::from(path);
    RustRecorder::load_from_file(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))
}