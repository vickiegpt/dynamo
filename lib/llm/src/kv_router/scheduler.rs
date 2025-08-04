// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use dashmap::DashMap;
use dynamo_runtime::component::{Component, Instance};
use dynamo_runtime::traits::events::{EventPublisher, EventSubscriber};
use futures::StreamExt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use super::indexer::OverlapScores;
use super::prefill_counter::{get_snapshot, PrefillCountersMultiWorker};
use super::protocols::WorkerSelectionResult;
use super::scoring::LoadEvent;
use super::KvRouterConfig;
use super::WorkerSelector;
use super::{KV_HIT_RATE_SUBJECT, KV_METRICS_SUBJECT};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVHitRateEvent {
    pub worker_id: i64,
    pub isl_blocks: usize,
    pub overlap_blocks: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints aviailable to route work")]
    NoEndpoints,

    #[error("all workers busy")]
    AllWorkersBusy,

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,
}

#[derive(Debug)]
pub struct SchedulingResponse {
    pub best_worker_id: i64,
    pub new_tokens: usize,
}

pub struct SchedulingRequest {
    pub isl_tokens: usize,
    pub overlaps: OverlapScores,
    pub decode_blocks: HashMap<i64, usize>,
    pub prefill_tokens: HashMap<i64, usize>,
    resp_tx: tokio::sync::oneshot::Sender<SchedulingResponse>,
}

impl SchedulingRequest {
    pub fn respond(self, response: SchedulingResponse) {
        if self.resp_tx.send(response).is_err() {
            tracing::error!("failed to send response to requestor");
        }
    }
}

pub struct KvScheduler {
    request_tx: tokio::sync::mpsc::Sender<SchedulingRequest>,
    prefill_tokens: Arc<PrefillCountersMultiWorker>,
    barrier: Mutex<()>,
}

impl KvScheduler {
    pub async fn start(
        component: Component,
        block_size: u32,
        mut instances_rx: tokio::sync::watch::Receiver<Vec<Instance>>, // Changed from ProcessedEndpoints
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
    ) -> Result<Self, KvSchedulerError> {
        let selector = selector.unwrap_or(Box::new(DefaultWorkerSelector::default()));
        let mut instances: Vec<Instance> = instances_rx.borrow_and_update().clone();

        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<KVHitRateEvent>();
        let ns_clone = component.namespace().clone();
        tokio::spawn(async move {
            let mut event_rx = event_rx;
            while let Some(event) = event_rx.recv().await {
                if let Err(e) = ns_clone.publish(KV_HIT_RATE_SUBJECT, &event).await {
                    tracing::warn!("Failed to publish KV hit rate event: {:?}", e);
                }
            }
        });

        // token and block states
        let active_blocks = Arc::new(DashMap::<i64, usize>::new()); // Create dashmap for worker_id -> active_blocks
        let prefill_tokens = Arc::new(PrefillCountersMultiWorker::new(component.clone()));

        // listen on load metrics from backends to update active block states
        let active_blocks_clone = active_blocks.clone();
        let ns_clone = component.namespace().clone();
        tokio::spawn(async move {
            let mut load_subscriber = ns_clone
                .subscribe_with_type::<LoadEvent>(KV_METRICS_SUBJECT)
                .await
                .expect("Cannot launch load subscriber");

            while let Some(event_result) = load_subscriber.next().await {
                let Ok(load_event) = event_result else {
                    tracing::warn!("Error receiving load event: {}", event_result.unwrap_err());
                    continue;
                };

                active_blocks_clone.insert(
                    load_event.worker_id,
                    load_event.data.kv_stats.kv_active_blocks as usize,
                );
            }
        });

        let prefill_tokens_clone = prefill_tokens.clone();
        let (request_tx, request_rx) = tokio::sync::mpsc::channel::<SchedulingRequest>(1024);
        // Background task to handle scheduling requests
        tokio::spawn(async move {
            let mut request_rx = request_rx;
            tracing::trace!("scheduler background task started");

            loop {
                // First, check for instance updates (non-blocking)
                if instances_rx.has_changed().unwrap_or(false) {
                    instances = instances_rx.borrow_and_update().clone();
                }

                // Then, wait for a new request
                let Some(mut request) = request_rx.recv().await else {
                    tracing::warn!("scheduler shutdown");
                    break;
                };
                tracing::trace!("received request to be scheduled");

                request.prefill_tokens = prefill_tokens_clone.running_sums().await;
                request.decode_blocks = get_snapshot(active_blocks.as_ref());

                match selector.select_worker(&instances, &request, block_size) {
                    Ok(selection) => {
                        if let Err(e) = event_tx.send(KVHitRateEvent {
                            worker_id: selection.worker_id,
                            isl_blocks: selection.required_blocks as usize,
                            overlap_blocks: selection.overlap_blocks,
                        }) {
                            tracing::warn!("Failed to send KV hit rate event: {:?}", e);
                        }

                        let response = SchedulingResponse {
                            best_worker_id: selection.worker_id,
                            new_tokens: request.isl_tokens
                                - (selection.overlap_blocks * block_size) as usize,
                        };
                        request.respond(response);
                        continue;
                    }
                    Err(KvSchedulerError::NoEndpoints) => {
                        tracing::trace!("no endpoints available; waiting for endpoints update");
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        continue;
                    }
                    // TODO: this is not actually hooked up
                    Err(KvSchedulerError::AllWorkersBusy) => {
                        tracing::trace!("all workers busy; waiting for more capacity");
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        continue;
                    }
                    Err(e) => {
                        tracing::error!("error scheduling request: {:?}", e);
                        break;
                    }
                }
            }

            tracing::trace!("background endpoint subscriber shutting down");
        });

        Ok(KvScheduler {
            request_tx,
            prefill_tokens,
            barrier: Mutex::new(()),
        })
    }

    pub async fn schedule(
        &self,
        request_id: String,
        isl_tokens: usize,
        overlaps: OverlapScores,
    ) -> Result<i64, KvSchedulerError> {
        // TODO: this is temporary needed for now to ensure blocking of scheduling and updating
        // Need to remove in future if we have better algo to truly enable async processing
        let _guard = self.barrier.lock().await;

        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            isl_tokens,
            overlaps,
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            resp_tx,
        };

        self.request_tx
            .send(request)
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;
        let response = resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;

        let best_worker_id = response.best_worker_id;
        let _ = self
            .prefill_tokens
            .add_prefill(best_worker_id, request_id, response.new_tokens)
            .await;
        Ok(best_worker_id)
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) {
        let _ = self.prefill_tokens.remove_prefill(request_id).await;
    }
}

// Helper function for softmax sampling
fn softmax_sample(logits: &HashMap<i64, f64>, temperature: f64) -> i64 {
    if logits.is_empty() {
        panic!("Empty logits for softmax sampling");
    }

    // Guard: if temperature is 0, return the key with the smallest logit value
    if temperature == 0.0 {
        // Find the minimum logit value
        let min_logit = logits.values().fold(f64::INFINITY, |a, &b| a.min(b));

        // Collect all keys with the minimum logit value (to handle ties)
        let min_keys: Vec<_> = logits
            .iter()
            .filter(|(_, &v)| v == min_logit)
            .map(|(k, _)| *k)
            .collect();

        // Randomly select from the minimum keys (handles single key case naturally)
        let mut rng = rand::rng();
        let index = rng.random_range(0..min_keys.len());
        return min_keys[index];
    }

    let keys: Vec<_> = logits.keys().copied().collect();
    let values: Vec<_> = logits.values().copied().collect();

    // Find min and max for normalization
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let probabilities = if min_val == max_val {
        // All values are the same, uniform probability
        vec![1.0 / keys.len() as f64; keys.len()]
    } else {
        // Normalize values
        let normalized: Vec<_> = values
            .iter()
            .map(|&v| {
                // Lower is better, so negate
                // Note we don't need to do actual min-max norm here, just off by an offset
                let norm = v / (max_val - min_val);
                -norm
            })
            .collect();

        // Apply temperature and softmax
        let scaled: Vec<_> = normalized.iter().map(|&v| v / temperature).collect();

        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<_> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();

        let sum_exp: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&v| v / sum_exp).collect()
    };

    // Sample from the probability distribution
    let mut rng = rand::rng();
    let sample: f64 = rng.random();

    let mut cumsum = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return keys[i];
        }
    }

    // Fallback to last key (shouldn't normally reach here)
    keys[keys.len() - 1]
}

// Default implementation matching the Python _cost_function
#[derive(Debug, Clone, Default)]
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
        }
    }
}

impl WorkerSelector for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &[Instance],
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);

        if workers.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let isl = request.isl_tokens;
        let request_blocks = isl.div_ceil(block_size as usize);
        let overlaps = &request.overlaps.scores;

        let decode_blocks = &request.decode_blocks;
        let prefill_tokens = &request.prefill_tokens;

        let mut worker_logits = HashMap::new();
        let mut max_logit = f64::NEG_INFINITY;

        // Calculate logits for each worker
        for instance in workers.iter() {
            let worker_id = instance.instance_id;

            // this is the number of prefill tokens the worker would have if the request were scheduled there
            let prefill_token = *prefill_tokens.get(&worker_id).unwrap_or(&0);
            let overlap = *overlaps.get(&worker_id).unwrap_or(&0);
            let new_token = isl - (overlap * block_size) as usize;
            let potential_prefill_token = prefill_token + new_token;
            let potential_prefill_block = (potential_prefill_token as f64) / (block_size as f64);

            // this is the number of decode blocks currently on the worker
            let decode_block = *decode_blocks.get(&worker_id).unwrap_or(&0) as f64;

            // Calculate logit (lower is better)
            let logit =
                self.kv_router_config.overlap_score_weight * potential_prefill_block + decode_block;
            max_logit = max_logit.max(logit);

            worker_logits.insert(worker_id, logit);

            tracing::info!(
                "Formula for {worker_id}: {logit:.3} = {:.1} * {potential_prefill_block:.3} + {decode_block:.3}  (cached_blocks: {})",
                self.kv_router_config.overlap_score_weight,
                overlap,
            );
        }

        // Use softmax sampling to select worker
        let temperature = self.kv_router_config.router_temperature;
        let best_worker_id = softmax_sample(&worker_logits, temperature);
        let best_logit = worker_logits[&best_worker_id];

        tracing::info!(
            "Selected worker: {}, logit: {:.3}",
            best_worker_id,
            best_logit
        );

        Ok(WorkerSelectionResult {
            worker_id: best_worker_id,
            required_blocks: request_blocks as u64,
            overlap_blocks: overlaps.get(&best_worker_id).copied().unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sample_single_key() {
        // Test that with a single key, softmax_sample always returns that key
        let mut logits = HashMap::new();
        let worker_id = 42;
        logits.insert(worker_id, 0.5); // The value doesn't matter

        // Test with different temperatures
        for temperature in &[0.1, 1.0, 10.0] {
            let result = softmax_sample(&logits, *temperature);
            assert_eq!(result, worker_id, "Should return the only available worker");
        }

        // Test with different logit values
        logits.clear();
        logits.insert(worker_id, -100.0); // Very negative value
        assert_eq!(softmax_sample(&logits, 1.0), worker_id);

        logits.clear();
        logits.insert(worker_id, 100.0); // Very positive value
        assert_eq!(softmax_sample(&logits, 1.0), worker_id);

        logits.clear();
        logits.insert(worker_id, 0.0); // Zero value
        assert_eq!(softmax_sample(&logits, 1.0), worker_id);
    }

    #[test]
    fn test_softmax_sample_zero_temperature() {
        // Test that with temperature 0, softmax_sample returns the key with smallest logit
        let mut logits = HashMap::new();
        logits.insert(1, 5.0);
        logits.insert(2, 3.0); // This has the smallest logit
        logits.insert(3, 7.0);
        logits.insert(4, 3.5);

        // With temperature 0, should always return worker 2 (smallest logit)
        for _ in 0..10 {
            let result = softmax_sample(&logits, 0.0);
            assert_eq!(
                result, 2,
                "Should return worker with smallest logit when temperature is 0"
            );
        }

        // Test with negative values
        logits.clear();
        logits.insert(10, -1.0);
        logits.insert(20, -5.0); // This has the smallest logit
        logits.insert(30, 0.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(result, 20, "Should handle negative logits correctly");
    }
}
