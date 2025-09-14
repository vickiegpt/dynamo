// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust data structures for vLLM scheduler types
//!
//! These structures mirror the essential fields from vLLM's Python objects
//! and are designed to be serializable for recording and replay.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a new request being scheduled for the first time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRequestData {
    pub req_id: String,
    pub prompt_token_ids: Vec<i32>,
    pub block_ids: Vec<Vec<i32>>,  // tuple[list[int], ...] in Python
    pub num_computed_tokens: usize,
    // Additional fields we might need
    pub mm_hashes: Vec<String>,
    pub mm_positions: Vec<PlaceholderRange>,
}

/// Placeholder range for multimodal inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceholderRange {
    pub start: usize,
    pub end: usize,
}

/// Represents cached requests that have been scheduled before
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRequestData {
    pub req_ids: Vec<String>,
    pub resumed_from_preemption: Vec<bool>,
    pub new_token_ids: Vec<Vec<i32>>,  // For pipeline parallelism
    pub new_block_ids: Vec<Option<Vec<Vec<i32>>>>,
    pub num_computed_tokens: Vec<usize>,
}

/// Main scheduler output structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerOutput {
    /// Requests scheduled for the first time
    pub scheduled_new_reqs: Vec<NewRequestData>,

    /// Previously scheduled requests (cached)
    pub scheduled_cached_reqs: CachedRequestData,

    /// Number of tokens scheduled for each request
    pub num_scheduled_tokens: HashMap<String, usize>,

    /// Total number of scheduled tokens
    pub total_num_scheduled_tokens: usize,

    /// Speculative decode tokens per request
    pub scheduled_spec_decode_tokens: HashMap<String, Vec<i32>>,

    /// Encoder inputs that need processing
    pub scheduled_encoder_inputs: HashMap<String, Vec<usize>>,

    /// Number of common prefix blocks for cascade attention
    pub num_common_prefix_blocks: Vec<usize>,

    /// Finished request IDs
    pub finished_req_ids: Vec<String>,

    /// MM hashes to free from encoder cache
    pub free_encoder_mm_hashes: Vec<String>,
}

/// Logprobs data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobsLists {
    pub logprob_token_ids: Vec<Vec<i32>>,
    pub logprobs: Vec<Vec<f32>>,
    pub sampled_token_ranks: Vec<i32>,
}

/// Model runner output structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRunnerOutput {
    /// Request IDs in order
    pub req_ids: Vec<String>,

    /// Map from request ID to index
    pub req_id_to_index: HashMap<String, usize>,

    /// Sampled token IDs for each request
    pub sampled_token_ids: Vec<Vec<i32>>,

    /// Optional logprobs
    pub logprobs: Option<LogprobsLists>,

    /// Prompt logprobs per request
    pub prompt_logprobs_dict: HashMap<String, Option<LogprobsLists>>,

    /// Number of NaNs in logits (for debugging)
    pub num_nans_in_logits: Option<HashMap<String, usize>>,
}

/// Finish reason for a request
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FinishReason {
    Stop = 0,
    Length = 1,
    Abort = 2,
}

/// Engine core event type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EngineCoreEventType {
    Queued = 1,
    Scheduled = 2,
    Preempted = 3,
}

/// Engine core event with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreEvent {
    pub event_type: EngineCoreEventType,
    pub timestamp: f64,
}

/// Output for a single request from the engine core
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<i32>,
    pub new_logprobs: Option<LogprobsLists>,
    pub finish_reason: Option<FinishReason>,
    pub stop_reason: Option<StopReason>,
    pub events: Option<Vec<EngineCoreEvent>>,
    pub num_cached_tokens: usize,
}

/// Stop reason (can be string or int)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopReason {
    String(String),
    Int(i32),
}

/// Collection of engine core outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreOutputs {
    pub engine_index: usize,
    pub outputs: Vec<EngineCoreOutput>,
    pub timestamp: f64,
}

/// Complete iteration record containing all scheduler data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationRecord {
    pub iteration: u64,
    pub schedule_output: SchedulerOutput,
    pub model_runner_output: ModelRunnerOutput,
    pub engine_core_outputs: EngineCoreOutputs,
    pub timestamp: f64,
}

/// Complete recording trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerTrace {
    pub metadata: TraceMetadata,
    pub iterations: Vec<IterationRecord>,
}

/// Metadata about the recording
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    pub vllm_version: String,
    pub model: String,
    pub timestamp: String,
    pub total_iterations: usize,
}