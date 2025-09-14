// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-to-Rust converters for vLLM scheduler types

use dynamo_llm::integrations::vllm::types::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Convert Python SchedulerOutput to Rust
#[pyfunction]
pub fn convert_scheduler_output(py: Python, obj: &Bound<'_, PyAny>) -> PyResult<SchedulerOutput> {
    // Extract scheduled_new_reqs
    let new_reqs_py = obj.getattr("scheduled_new_reqs")?;
    let mut scheduled_new_reqs = Vec::new();

    for item in new_reqs_py.iter()? {
        let item = item?;
        let req_id = item.getattr("req_id")?.extract::<String>()?;
        let prompt_token_ids = item.getattr("prompt_token_ids")?.extract::<Vec<i32>>()?;

        // Extract block_ids (tuple of lists)
        let block_ids_tuple = item.getattr("block_ids")?;
        let mut block_ids = Vec::new();
        for block_list in block_ids_tuple.iter()? {
            let block_list = block_list?;
            let blocks = block_list.extract::<Vec<i32>>()?;
            block_ids.push(blocks);
        }

        let num_computed_tokens = item.getattr("num_computed_tokens")?.extract::<usize>()?;

        // Extract mm_hashes and mm_positions
        let mm_hashes = if let Ok(hashes) = item.getattr("mm_hashes") {
            hashes.extract::<Vec<String>>().unwrap_or_default()
        } else {
            Vec::new()
        };

        let mm_positions = if let Ok(positions) = item.getattr("mm_positions") {
            let mut ranges = Vec::new();
            for pos in positions.iter()? {
                let pos = pos?;
                if let Ok(start) = pos.getattr("start") {
                    let start = start.extract::<usize>()?;
                    let end = pos.getattr("end")?.extract::<usize>()?;
                    ranges.push(PlaceholderRange { start, end });
                }
            }
            ranges
        } else {
            Vec::new()
        };

        scheduled_new_reqs.push(NewRequestData {
            req_id,
            prompt_token_ids,
            block_ids,
            num_computed_tokens,
            mm_hashes,
            mm_positions,
        });
    }

    // Extract scheduled_cached_reqs
    let cached_reqs_py = obj.getattr("scheduled_cached_reqs")?;
    let scheduled_cached_reqs = CachedRequestData {
        req_ids: cached_reqs_py.getattr("req_ids")?.extract::<Vec<String>>()?,
        resumed_from_preemption: cached_reqs_py
            .getattr("resumed_from_preemption")?
            .extract::<Vec<bool>>()?,
        new_token_ids: cached_reqs_py
            .getattr("new_token_ids")?
            .extract::<Vec<Vec<i32>>>()?,
        new_block_ids: {
            let new_blocks = cached_reqs_py.getattr("new_block_ids")?;
            let mut result = Vec::new();
            for item in new_blocks.iter()? {
                let item = item?;
                if item.is_none() {
                    result.push(None);
                } else {
                    let mut block_ids = Vec::new();
                    for block_list in item.iter()? {
                        let block_list = block_list?;
                        let blocks = block_list.extract::<Vec<i32>>()?;
                        block_ids.push(blocks);
                    }
                    result.push(Some(block_ids));
                }
            }
            result
        },
        num_computed_tokens: cached_reqs_py
            .getattr("num_computed_tokens")?
            .extract::<Vec<usize>>()?,
    };

    // Extract num_scheduled_tokens
    let num_scheduled_tokens_py = obj.getattr("num_scheduled_tokens")?;
    let num_scheduled_tokens = if let Ok(dict) = num_scheduled_tokens_py.downcast::<PyDict>() {
        let mut map = HashMap::new();
        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            let value = value.extract::<usize>()?;
            map.insert(key, value);
        }
        map
    } else {
        HashMap::new()
    };

    // Extract other fields
    let total_num_scheduled_tokens = obj
        .getattr("total_num_scheduled_tokens")?
        .extract::<usize>()?;

    let scheduled_spec_decode_tokens = if let Ok(spec_tokens) = obj.getattr("scheduled_spec_decode_tokens") {
        if let Ok(dict) = spec_tokens.downcast::<PyDict>() {
            let mut map = HashMap::new();
            for (key, value) in dict.iter() {
                let key = key.extract::<String>()?;
                let value = value.extract::<Vec<i32>>()?;
                map.insert(key, value);
            }
            map
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    let scheduled_encoder_inputs = if let Ok(encoder_inputs) = obj.getattr("scheduled_encoder_inputs") {
        if let Ok(dict) = encoder_inputs.downcast::<PyDict>() {
            let mut map = HashMap::new();
            for (key, value) in dict.iter() {
                let key = key.extract::<String>()?;
                let value = value.extract::<Vec<usize>>()?;
                map.insert(key, value);
            }
            map
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    let num_common_prefix_blocks = obj
        .getattr("num_common_prefix_blocks")?
        .extract::<Vec<usize>>()
        .unwrap_or_default();

    let finished_req_ids = if let Ok(finished) = obj.getattr("finished_req_ids") {
        // Convert set to vec
        let mut ids = Vec::new();
        for item in finished.iter()? {
            let item = item?;
            ids.push(item.extract::<String>()?);
        }
        ids
    } else {
        Vec::new()
    };

    let free_encoder_mm_hashes = obj
        .getattr("free_encoder_mm_hashes")?
        .extract::<Vec<String>>()
        .unwrap_or_default();

    Ok(SchedulerOutput {
        scheduled_new_reqs,
        scheduled_cached_reqs,
        num_scheduled_tokens,
        total_num_scheduled_tokens,
        scheduled_spec_decode_tokens,
        scheduled_encoder_inputs,
        num_common_prefix_blocks,
        finished_req_ids,
        free_encoder_mm_hashes,
    })
}

/// Convert Python ModelRunnerOutput to Rust
#[pyfunction]
pub fn convert_model_runner_output(py: Python, obj: &Bound<'_, PyAny>) -> PyResult<ModelRunnerOutput> {
    let req_ids = obj.getattr("req_ids")?.extract::<Vec<String>>()?;

    let req_id_to_index_py = obj.getattr("req_id_to_index")?;
    let req_id_to_index = if let Ok(dict) = req_id_to_index_py.downcast::<PyDict>() {
        let mut map = HashMap::new();
        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            let value = value.extract::<usize>()?;
            map.insert(key, value);
        }
        map
    } else {
        HashMap::new()
    };

    let sampled_token_ids = obj
        .getattr("sampled_token_ids")?
        .extract::<Vec<Vec<i32>>>()?;

    let logprobs = if let Ok(logprobs_py) = obj.getattr("logprobs") {
        if !logprobs_py.is_none() {
            Some(LogprobsLists {
                logprob_token_ids: logprobs_py
                    .getattr("logprob_token_ids")?
                    .extract::<Vec<Vec<i32>>>()?,
                logprobs: logprobs_py
                    .getattr("logprobs")?
                    .extract::<Vec<Vec<f32>>>()?,
                sampled_token_ranks: logprobs_py
                    .getattr("sampled_token_ranks")?
                    .extract::<Vec<i32>>()?,
            })
        } else {
            None
        }
    } else {
        None
    };

    let prompt_logprobs_dict = if let Ok(prompt_dict) = obj.getattr("prompt_logprobs_dict") {
        if let Ok(dict) = prompt_dict.downcast::<PyDict>() {
            let mut map = HashMap::new();
            for (key, value) in dict.iter() {
                let key = key.extract::<String>()?;
                if !value.is_none() {
                    let logprobs = Some(LogprobsLists {
                        logprob_token_ids: value
                            .getattr("logprob_token_ids")?
                            .extract::<Vec<Vec<i32>>>()?,
                        logprobs: value.getattr("logprobs")?.extract::<Vec<Vec<f32>>>()?,
                        sampled_token_ranks: value
                            .getattr("selected_token_ranks")?
                            .extract::<Vec<i32>>()?,
                    });
                    map.insert(key, logprobs);
                } else {
                    map.insert(key, None);
                }
            }
            map
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    let num_nans_in_logits = if let Ok(nans) = obj.getattr("num_nans_in_logits") {
        if !nans.is_none() {
            if let Ok(dict) = nans.downcast::<PyDict>() {
                let mut map = HashMap::new();
                for (key, value) in dict.iter() {
                    let key = key.extract::<String>()?;
                    let value = value.extract::<usize>()?;
                    map.insert(key, value);
                }
                Some(map)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    Ok(ModelRunnerOutput {
        req_ids,
        req_id_to_index,
        sampled_token_ids,
        logprobs,
        prompt_logprobs_dict,
        num_nans_in_logits,
    })
}

/// Convert Python EngineCoreOutputs to Rust
#[pyfunction]
pub fn convert_engine_core_outputs(py: Python, outputs_dict: &Bound<'_, PyDict>) -> PyResult<EngineCoreOutputs> {
    let mut all_outputs = Vec::new();

    // The dict is keyed by engine_index
    for (engine_idx, outputs_list) in outputs_dict.iter() {
        let engine_index = engine_idx.extract::<usize>()?;

        for output in outputs_list.iter()? {
            let output = output?;

            let request_id = output.getattr("request_id")?.extract::<String>()?;
            let new_token_ids = output.getattr("new_token_ids")?.extract::<Vec<i32>>()?;

            let new_logprobs = if let Ok(logprobs_py) = output.getattr("new_logprobs") {
                if !logprobs_py.is_none() {
                    Some(LogprobsLists {
                        logprob_token_ids: logprobs_py
                            .getattr("logprob_token_ids")?
                            .extract::<Vec<Vec<i32>>>()?,
                        logprobs: logprobs_py
                            .getattr("logprobs")?
                            .extract::<Vec<Vec<f32>>>()?,
                        sampled_token_ranks: logprobs_py
                            .getattr("sampled_token_ranks")?
                            .extract::<Vec<i32>>()?,
                    })
                } else {
                    None
                }
            } else {
                None
            };

            let finish_reason = if let Ok(reason) = output.getattr("finish_reason") {
                if !reason.is_none() {
                    let val = reason.extract::<i32>()?;
                    Some(match val {
                        0 => FinishReason::Stop,
                        1 => FinishReason::Length,
                        2 => FinishReason::Abort,
                        _ => FinishReason::Abort,
                    })
                } else {
                    None
                }
            } else {
                None
            };

            let stop_reason = if let Ok(reason) = output.getattr("stop_reason") {
                if !reason.is_none() {
                    if let Ok(s) = reason.extract::<String>() {
                        Some(StopReason::String(s))
                    } else if let Ok(i) = reason.extract::<i32>() {
                        Some(StopReason::Int(i))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let events = if let Ok(events_py) = output.getattr("events") {
                if !events_py.is_none() {
                    let mut events = Vec::new();
                    for event in events_py.iter()? {
                        let event = event?;
                        let event_type = event.getattr("type")?.extract::<i32>()?;
                        let timestamp = event.getattr("timestamp")?.extract::<f64>()?;

                        let event_type = match event_type {
                            1 => EngineCoreEventType::Queued,
                            2 => EngineCoreEventType::Scheduled,
                            3 => EngineCoreEventType::Preempted,
                            _ => EngineCoreEventType::Queued,
                        };

                        events.push(EngineCoreEvent {
                            event_type,
                            timestamp,
                        });
                    }
                    Some(events)
                } else {
                    None
                }
            } else {
                None
            };

            let num_cached_tokens = output
                .getattr("num_cached_tokens")?
                .extract::<usize>()
                .unwrap_or(0);

            all_outputs.push(EngineCoreOutput {
                request_id,
                new_token_ids,
                new_logprobs,
                finish_reason,
                stop_reason,
                events,
                num_cached_tokens,
            });
        }

        // Use first engine index (typically 0)
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        return Ok(EngineCoreOutputs {
            engine_index,
            outputs: all_outputs,
            timestamp,
        });
    }

    // Return empty if no outputs
    Ok(EngineCoreOutputs {
        engine_index: 0,
        outputs: Vec::new(),
        timestamp: 0.0,
    })
}