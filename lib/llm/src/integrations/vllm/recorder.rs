// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler recorder for capturing vLLM scheduler behavior
//!
//! Records scheduler outputs and model runner outputs for replay and testing.

use super::types::*;
use chrono::Utc;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Records scheduler interactions for later replay
pub struct SchedulerRecorder {
    /// Current iteration counter
    iteration: u64,

    /// All recorded iterations
    recordings: Vec<IterationRecord>,

    /// Partial record being built for current iteration
    current_record: Option<PartialIterationRecord>,

    /// Metadata for the recording
    metadata: TraceMetadata,
}

/// Partial record while building an iteration
struct PartialIterationRecord {
    iteration: u64,
    schedule_output: Option<SchedulerOutput>,
    model_runner_output: Option<ModelRunnerOutput>,
    engine_core_outputs: Option<EngineCoreOutputs>,
    timestamp: f64,
}

impl SchedulerRecorder {
    /// Create a new recorder with metadata
    pub fn new(model: String, vllm_version: String) -> Self {
        Self {
            iteration: 0,
            recordings: Vec::new(),
            current_record: None,
            metadata: TraceMetadata {
                vllm_version,
                model,
                timestamp: Utc::now().to_rfc3339(),
                total_iterations: 0,
            },
        }
    }

    /// Record a scheduler output
    pub fn record_schedule_output(&mut self, output: SchedulerOutput) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        match &mut self.current_record {
            Some(record) => {
                record.schedule_output = Some(output);
            }
            None => {
                self.current_record = Some(PartialIterationRecord {
                    iteration: self.iteration,
                    schedule_output: Some(output),
                    model_runner_output: None,
                    engine_core_outputs: None,
                    timestamp,
                });
            }
        }
    }

    /// Record a model runner output
    pub fn record_model_runner_output(&mut self, output: ModelRunnerOutput) {
        match &mut self.current_record {
            Some(record) => {
                record.model_runner_output = Some(output);
            }
            None => {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();

                self.current_record = Some(PartialIterationRecord {
                    iteration: self.iteration,
                    schedule_output: None,
                    model_runner_output: Some(output),
                    engine_core_outputs: None,
                    timestamp,
                });
            }
        }
    }

    /// Record engine core outputs
    pub fn record_engine_core_outputs(&mut self, outputs: EngineCoreOutputs) {
        match &mut self.current_record {
            Some(record) => {
                record.engine_core_outputs = Some(outputs);
            }
            None => {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();

                self.current_record = Some(PartialIterationRecord {
                    iteration: self.iteration,
                    schedule_output: None,
                    model_runner_output: None,
                    engine_core_outputs: Some(outputs),
                    timestamp,
                });
            }
        }
    }

    /// Move to the next iteration
    pub fn next_iteration(&mut self) {
        // Finalize current record if complete
        if let Some(record) = self.current_record.take() {
            if record.schedule_output.is_some()
                && record.model_runner_output.is_some()
                && record.engine_core_outputs.is_some()
            {
                let complete_record = IterationRecord {
                    iteration: record.iteration,
                    schedule_output: record.schedule_output.unwrap(),
                    model_runner_output: record.model_runner_output.unwrap(),
                    engine_core_outputs: record.engine_core_outputs.unwrap(),
                    timestamp: record.timestamp,
                };
                self.recordings.push(complete_record);
            } else {
                eprintln!(
                    "Warning: Incomplete iteration {} - schedule: {}, model: {}, engine: {}",
                    record.iteration,
                    record.schedule_output.is_some(),
                    record.model_runner_output.is_some(),
                    record.engine_core_outputs.is_some()
                );
                // Still save partial record if needed
                if record.schedule_output.is_some() || record.model_runner_output.is_some() {
                    // Create a minimal complete record with defaults
                    let complete_record = IterationRecord {
                        iteration: record.iteration,
                        schedule_output: record.schedule_output.unwrap_or_else(|| {
                            SchedulerOutput {
                                scheduled_new_reqs: Vec::new(),
                                scheduled_cached_reqs: CachedRequestData {
                                    req_ids: Vec::new(),
                                    resumed_from_preemption: Vec::new(),
                                    new_token_ids: Vec::new(),
                                    new_block_ids: Vec::new(),
                                    num_computed_tokens: Vec::new(),
                                },
                                num_scheduled_tokens: HashMap::new(),
                                total_num_scheduled_tokens: 0,
                                scheduled_spec_decode_tokens: HashMap::new(),
                                scheduled_encoder_inputs: HashMap::new(),
                                num_common_prefix_blocks: Vec::new(),
                                finished_req_ids: Vec::new(),
                                free_encoder_mm_hashes: Vec::new(),
                            }
                        }),
                        model_runner_output: record.model_runner_output.unwrap_or_else(|| {
                            ModelRunnerOutput {
                                req_ids: Vec::new(),
                                req_id_to_index: HashMap::new(),
                                sampled_token_ids: Vec::new(),
                                logprobs: None,
                                prompt_logprobs_dict: HashMap::new(),
                                num_nans_in_logits: None,
                            }
                        }),
                        engine_core_outputs: record.engine_core_outputs.unwrap_or_else(|| {
                            EngineCoreOutputs {
                                engine_index: 0,
                                outputs: Vec::new(),
                                timestamp: record.timestamp,
                            }
                        }),
                        timestamp: record.timestamp,
                    };
                    self.recordings.push(complete_record);
                }
            }
        }

        // Increment iteration counter
        self.iteration += 1;
        self.current_record = None;
    }

    /// Get current iteration number
    pub fn current_iteration(&self) -> u64 {
        self.iteration
    }

    /// Save recordings to a JSON file
    pub fn save_to_file(&mut self, path: &Path) -> std::io::Result<()> {
        // Finalize any pending record
        if self.current_record.is_some() {
            self.next_iteration();
        }

        // Update metadata
        self.metadata.total_iterations = self.recordings.len();

        let trace = SchedulerTrace {
            metadata: self.metadata.clone(),
            iterations: self.recordings.clone(),
        };

        let json = serde_json::to_string_pretty(&trace)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;

        println!("Saved {} iterations to {:?}", self.recordings.len(), path);
        Ok(())
    }

    /// Load recordings from a JSON file
    pub fn load_from_file(path: &Path) -> std::io::Result<SchedulerTrace> {
        let file = File::open(path)?;
        let trace: SchedulerTrace = serde_json::from_reader(file)?;
        Ok(trace)
    }

    /// Get the recorded trace
    pub fn get_trace(&self) -> SchedulerTrace {
        SchedulerTrace {
            metadata: self.metadata.clone(),
            iterations: self.recordings.clone(),
        }
    }

    /// Clear all recordings
    pub fn clear(&mut self) {
        self.recordings.clear();
        self.current_record = None;
        self.iteration = 0;
    }
}
