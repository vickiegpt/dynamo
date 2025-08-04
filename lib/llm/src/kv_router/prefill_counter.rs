// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::events::{EventPublisher, EventSubscriber};
use futures::StreamExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;
// Remove the Mutex import since we're using DashMap

use super::protocols::{PrefillEvent, PrefillEventData};
use crate::kv_router::PREFILL_SUBJECT;
use dashmap::DashMap;
use std::collections::HashMap;
use std::hash::Hash;

pub fn get_snapshot<K, V>(state: &DashMap<K, V>) -> HashMap<K, V>
where
    K: Clone + Hash + Eq,
    V: Copy,
{
    state
        .iter()
        .map(|entry| (entry.key().clone(), *entry.value()))
        .collect()
}

#[derive(Default)]
struct PrefillCounterState {
    tokens_map: DashMap<String, usize>,
    running_sum: AtomicUsize,
}

impl PrefillCounterState {
    fn insert(&self, key: String, value: usize) -> Option<usize> {
        let old_value = self.tokens_map.insert(key, value);

        if let Some(old) = old_value {
            self.running_sum.fetch_sub(old, Ordering::SeqCst);
            self.running_sum.fetch_add(value, Ordering::SeqCst);
        } else {
            self.running_sum.fetch_add(value, Ordering::SeqCst);
        }

        old_value
    }

    fn remove(&self, key: &str) -> Option<usize> {
        let removed = self.tokens_map.remove(key).map(|(_, v)| v);

        if let Some(value) = removed {
            self.running_sum.fetch_sub(value, Ordering::SeqCst);
        }

        removed
    }

    fn running_sum(&self) -> usize {
        self.running_sum.load(Ordering::SeqCst)
    }
}

/// A counter that tracks pending prefill tokens for each request.
///
/// This struct maintains a local hashmap of request_id to token count,
/// and a running sum of all tokens. It no longer handles its own subscriptions.
#[derive(Clone, Default)]
pub struct PrefillCounter {
    state: Arc<PrefillCounterState>,
}

impl PrefillCounter {
    // Internal methods for direct state manipulation (no publishing)
    fn insert_direct(&self, request_id: String, tokens: usize) -> Option<usize> {
        self.state.insert(request_id, tokens)
    }

    fn remove_direct(&self, request_id: &str) -> Option<usize> {
        self.state.remove(request_id)
    }

    fn update_direct(&self, request_id: String, new_tokens: usize) {
        if let Some(old_tokens_ref) = self.state.tokens_map.get(&request_id) {
            let old_tokens = *old_tokens_ref;
            let delta = new_tokens as isize - old_tokens as isize;
            self.state
                .running_sum
                .fetch_add(delta as usize, Ordering::SeqCst);
            self.state.tokens_map.insert(request_id, new_tokens);
        }
    }

    pub fn get(&self, request_id: &str) -> Option<usize> {
        self.state.tokens_map.get(request_id).map(|entry| *entry)
    }

    pub fn running_sum(&self) -> usize {
        self.state.running_sum()
    }

    pub fn len(&self) -> usize {
        self.state.tokens_map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.tokens_map.is_empty()
    }

    /// Returns a snapshot of the current state as a HashMap
    pub fn snapshot(&self) -> HashMap<String, usize> {
        get_snapshot(&self.state.tokens_map)
    }
}

/// A collection of PrefillCounters for multiple workers with centralized event handling
pub struct PrefillCountersMultiWorker {
    pub counters: Arc<DashMap<i64, PrefillCounter>>,
    pub request_to_workers: Arc<DashMap<String, i64>>,
    component: Component,
    router_id: Uuid,
}

impl PrefillCountersMultiWorker {
    pub fn new(component: Component) -> Self {
        let counters = Arc::new(DashMap::new());
        let request_to_workers = Arc::new(DashMap::new());
        let router_id = Uuid::new_v4();

        let multi_worker = Self {
            counters: counters.clone(),
            request_to_workers: request_to_workers.clone(),
            component: component.clone(),
            router_id,
        };

        // Start the subscription loop
        let counters_clone = counters.clone();
        let request_to_workers_clone = request_to_workers.clone();
        let component_clone = component.clone();
        let router_id_clone = router_id;

        tokio::spawn(async move {
            if let Err(e) = Self::subscribe_to_events(
                counters_clone,
                request_to_workers_clone,
                component_clone,
                router_id_clone,
            )
            .await
            {
                tracing::error!("Error in prefill events subscription: {}", e);
            }
        });

        multi_worker
    }

    /// Background task to subscribe to prefill events and update all counters
    async fn subscribe_to_events(
        counters: Arc<DashMap<i64, PrefillCounter>>,
        request_to_workers: Arc<DashMap<String, i64>>,
        component: Component,
        router_id: Uuid,
    ) -> Result<()> {
        let mut subscriber = component
            .subscribe_with_type::<PrefillEvent>(PREFILL_SUBJECT)
            .await?;

        while let Some(result) = subscriber.next().await {
            let Ok(event) = result else {
                tracing::error!("Error receiving prefill event: {}", result.unwrap_err());
                continue;
            };

            // Skip events emitted by itself
            if event.router_id == router_id {
                continue;
            }

            match event.data {
                PrefillEventData::NewPrefill(tokens) => {
                    let Some(worker_id_ref) = request_to_workers.get(&event.request_id) else {
                        continue;
                    };

                    let worker_id = *worker_id_ref;
                    let Some(counter) = counters.get(&worker_id) else {
                        tracing::warn!(
                            "No counter found for worker {} when handling NewPrefill for request {}",
                            worker_id,
                            event.request_id
                        );
                        continue;
                    };

                    counter.insert_direct(event.request_id.clone(), tokens);
                }
                PrefillEventData::UpdatePrefill(new_tokens) => {
                    let Some(worker_id_ref) = request_to_workers.get(&event.request_id) else {
                        continue;
                    };

                    let worker_id = *worker_id_ref;
                    let Some(counter) = counters.get(&worker_id) else {
                        tracing::warn!(
                            "No counter found for worker {} when handling UpdatePrefill for request {}",
                            worker_id,
                            event.request_id
                        );
                        continue;
                    };

                    counter.update_direct(event.request_id.clone(), new_tokens);
                }
                PrefillEventData::CompletePrefill => {
                    let Some((_, worker_id)) = request_to_workers.remove(&event.request_id) else {
                        continue;
                    };

                    let Some(counter) = counters.get(&worker_id) else {
                        tracing::warn!(
                            "No counter found for worker {} when handling CompletePrefill for request {}",
                            worker_id,
                            event.request_id
                        );
                        continue;
                    };

                    if counter.remove_direct(&event.request_id).is_none() {
                        tracing::warn!(
                            "Attempted to remove non-existent request: {}",
                            event.request_id
                        );
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn add_prefill(
        &self,
        worker_id: i64,
        request_id: String,
        new_tokens: usize,
    ) -> Result<()> {
        if let Some(existing_worker_id) = self.request_to_workers.get(&request_id) {
            tracing::warn!(
                "Request {} already exists for worker {}, but trying to add to worker {}",
                request_id,
                *existing_worker_id,
                worker_id
            );
        }
        self.request_to_workers
            .insert(request_id.clone(), worker_id);

        let counter = if let Some(counter) = self.counters.get(&worker_id) {
            counter.clone()
        } else {
            tracing::warn!(
                "Worker {} does not exist, creating new PrefillCounter",
                worker_id
            );
            let new_counter = PrefillCounter::default();
            self.counters.insert(worker_id, new_counter.clone());
            new_counter
        };

        let old_value = counter.insert_direct(request_id.clone(), new_tokens);

        // Publish the event
        let event = PrefillEvent {
            request_id,
            data: if old_value.is_some() {
                PrefillEventData::UpdatePrefill(new_tokens)
            } else {
                PrefillEventData::NewPrefill(new_tokens)
            },
            router_id: self.router_id,
        };
        self.component.publish(PREFILL_SUBJECT, &event).await?;

        Ok(())
    }

    pub async fn remove_prefill(&self, request_id: &str) -> Result<Option<usize>> {
        let Some((_request_id, worker_id)) = self.request_to_workers.remove(request_id) else {
            tracing::warn!("Request {} not found", request_id);
            return Ok(None);
        };

        if let Some(counter) = self.counters.get(&worker_id) {
            let removed_tokens = counter.remove_direct(request_id);

            if removed_tokens.is_some() {
                let event = PrefillEvent {
                    request_id: request_id.to_string(),
                    data: PrefillEventData::CompletePrefill,
                    router_id: self.router_id,
                };
                self.component.publish(PREFILL_SUBJECT, &event).await?;
            }

            Ok(removed_tokens)
        } else {
            tracing::warn!(
                "Worker {} not found in counters for request {}",
                worker_id,
                request_id
            );
            Ok(None)
        }
    }

    /// Get the running sums for all workers as a HashMap<i64, usize>
    pub async fn running_sums(&self) -> HashMap<i64, usize> {
        self.counters
            .iter()
            .map(|entry| (*entry.key(), entry.value().running_sum()))
            .collect()
    }

    /// Get a specific counter's running sum
    pub async fn get_worker_sum(&self, worker_id: i64) -> Option<usize> {
        self.counters.get(&worker_id).map(|c| c.running_sum())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use tokio::time::Duration;

    #[tokio::test]
    #[ignore]
    async fn test_prefill_counter_multiworker_synchronization() -> Result<()> {
        // Initialize logging
        dynamo_runtime::logging::init();

        // Create runtime and distributed runtime
        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        // Create namespace and components
        let namespace = distributed.namespace("test_prefill_multiworker")?;
        let component = namespace
            .component("counters")?
            .service_builder()
            .create()
            .await?;

        // Create two PrefillCountersMultiWorker instances
        let multi_worker1 = PrefillCountersMultiWorker::new(component.clone());
        let multi_worker2 = PrefillCountersMultiWorker::new(component.clone());

        // Give some time for subscribers to initialize
        tokio::time::sleep(Duration::from_millis(2000)).await;

        let worker_id_1 = 1;
        let worker_id_2 = 2;
        let tokens_per_request = 100;
        let requests_per_worker = 10;

        // Send requests to multi_worker1's worker
        for i in 0..requests_per_worker {
            let request_id = format!("mw1_request_{}", i);
            multi_worker1
                .add_prefill(worker_id_1, request_id, tokens_per_request)
                .await?;
        }

        // Send requests to multi_worker2's worker
        for i in 0..requests_per_worker {
            let request_id = format!("mw2_request_{}", i);
            multi_worker2
                .add_prefill(worker_id_2, request_id, tokens_per_request)
                .await?;
        }

        // Wait for synchronization
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify both multi-workers see all requests
        let sums1 = multi_worker1.running_sums().await;
        let sums2 = multi_worker2.running_sums().await;

        // Each multi-worker should see both workers
        assert_eq!(
            sums1.get(&worker_id_1),
            Some(&(requests_per_worker * tokens_per_request)),
            "MultiWorker1 should see worker 1's requests"
        );
        assert_eq!(
            sums1.get(&worker_id_2),
            Some(&(requests_per_worker * tokens_per_request)),
            "MultiWorker1 should see worker 2's requests"
        );
        assert_eq!(
            sums2.get(&worker_id_1),
            Some(&(requests_per_worker * tokens_per_request)),
            "MultiWorker2 should see worker 1's requests"
        );
        assert_eq!(
            sums2.get(&worker_id_2),
            Some(&(requests_per_worker * tokens_per_request)),
            "MultiWorker2 should see worker 2's requests"
        );

        // Remove all requests from multi_worker1
        for i in 0..requests_per_worker {
            let request_id = format!("mw1_request_{}", i);
            multi_worker1.remove_prefill(&request_id).await?;
        }

        // Remove all requests from multi_worker2
        for i in 0..requests_per_worker {
            let request_id = format!("mw2_request_{}", i);
            multi_worker2.remove_prefill(&request_id).await?;
        }

        // Wait for removal synchronization
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify both multi-workers show zero sums
        let final_sums1 = multi_worker1.running_sums().await;
        let final_sums2 = multi_worker2.running_sums().await;

        assert_eq!(
            final_sums1.get(&worker_id_1).copied().unwrap_or(0),
            0,
            "MultiWorker1 should show zero for worker 1"
        );
        assert_eq!(
            final_sums1.get(&worker_id_2).copied().unwrap_or(0),
            0,
            "MultiWorker1 should show zero for worker 2"
        );
        assert_eq!(
            final_sums2.get(&worker_id_1).copied().unwrap_or(0),
            0,
            "MultiWorker2 should show zero for worker 1"
        );
        assert_eq!(
            final_sums2.get(&worker_id_2).copied().unwrap_or(0),
            0,
            "MultiWorker2 should show zero for worker 2"
        );

        // Shutdown runtime
        runtime.shutdown();

        Ok(())
    }
}
