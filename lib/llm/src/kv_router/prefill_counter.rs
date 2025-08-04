// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::events::{EventPublisher, EventSubscriber};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
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

/// A counter that tracks pending prefill tokens for each request.
///
/// This struct maintains a local hashmap of request_id to token count,
/// a running sum of all tokens, and subscribes to prefill events over NATS
/// to keep the counts synchronized across components.
#[derive(Clone)]
pub struct PrefillCounter {
    state: Arc<RwLock<PrefillCounterState>>,
    component: Component,
    router_id: Uuid,
}

struct PrefillCounterState {
    tokens_map: DashMap<String, usize>,
    running_sum: AtomicUsize,
}

impl PrefillCounterState {
    fn new() -> Self {
        Self {
            tokens_map: DashMap::new(),
            running_sum: AtomicUsize::new(0),
        }
    }

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

impl PrefillCounter {
    /// Create a new PrefillCounter with the given component.
    ///
    /// This will start a background task that subscribes to PREFILL_SUBJECT
    /// and updates the internal state based on received events.
    pub fn new(component: Component) -> Self {
        let state = Arc::new(RwLock::new(PrefillCounterState::new()));
        let router_id = Uuid::new_v4();

        let counter = Self {
            state: state.clone(),
            component: component.clone(),
            router_id,
        };

        let state_clone = state.clone();
        let component_clone = component.clone();
        let router_id_clone = router_id;

        tokio::spawn(async move {
            if let Err(e) =
                Self::subscribe_to_events(state_clone, component_clone, router_id_clone).await
            {
                tracing::error!("Error in prefill events subscription: {}", e);
            }
        });

        counter
    }

    /// Background task to subscribe to prefill events and update internal state
    async fn subscribe_to_events(
        state: Arc<RwLock<PrefillCounterState>>,
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
                    let state_write = state.write().await;
                    state_write.insert(event.request_id.clone(), tokens);
                }
                PrefillEventData::UpdatePrefill(new_tokens) => {
                    let state_write = state.write().await;
                    let Some(old_tokens_ref) = state_write.tokens_map.get(&event.request_id) else {
                        continue;
                    };
                    let old_tokens = *old_tokens_ref;

                    let delta = new_tokens as isize - old_tokens as isize;
                    state_write
                        .running_sum
                        .fetch_add(delta as usize, Ordering::SeqCst);
                    state_write
                        .tokens_map
                        .insert(event.request_id.clone(), new_tokens);
                }
                PrefillEventData::CompletePrefill => {
                    let state_write = state.write().await;
                    if state_write.remove(&event.request_id).is_none() {
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

    pub async fn insert(&self, request_id: String, tokens: usize) -> Result<Option<usize>> {
        let state = self.state.write().await;
        let old_value = state.insert(request_id.clone(), tokens);

        // Send appropriate event based on whether this is a new prefill or an update
        let event = PrefillEvent {
            request_id,
            data: if old_value.is_some() {
                PrefillEventData::UpdatePrefill(tokens)
            } else {
                PrefillEventData::NewPrefill(tokens)
            },
            router_id: self.router_id,
        };
        self.component.publish(PREFILL_SUBJECT, &event).await?;

        Ok(old_value)
    }

    pub async fn remove(&self, request_id: &str) -> Result<Option<usize>> {
        let state = self.state.write().await;
        let removed_tokens = state.remove(request_id);

        if removed_tokens.is_some() {
            let event = PrefillEvent {
                request_id: request_id.to_string(),
                data: PrefillEventData::CompletePrefill,
                router_id: self.router_id,
            };
            self.component.publish(PREFILL_SUBJECT, &event).await?;
        }

        Ok(removed_tokens)
    }

    pub async fn get(&self, request_id: &str) -> Option<usize> {
        let state = self.state.read().await;
        state.tokens_map.get(request_id).map(|entry| *entry)
    }

    pub async fn running_sum(&self) -> usize {
        let state = self.state.read().await;
        state.running_sum()
    }

    pub async fn len(&self) -> usize {
        let state = self.state.read().await;
        state.tokens_map.len()
    }

    pub async fn is_empty(&self) -> bool {
        let state = self.state.read().await;
        state.tokens_map.is_empty()
    }

    /// Returns a snapshot of the current state as a HashMap
    pub async fn snapshot(&self) -> HashMap<String, usize> {
        let state = self.state.read().await;
        get_snapshot(&state.tokens_map)
    }
}

/// A collection of PrefillCounters for multiple workers
pub struct PrefillCountersMultiWorker {
    pub counters: DashMap<i64, PrefillCounter>,
    pub request_to_workers: DashMap<String, i64>,
    component: Component,
}

impl PrefillCountersMultiWorker {
    pub fn new(component: Component) -> Self {
        Self {
            counters: DashMap::new(),
            request_to_workers: DashMap::new(),
            component,
        }
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

        if let Some(counter) = self.counters.get(&worker_id) {
            counter.insert(request_id, new_tokens).await?;
        } else {
            tracing::warn!(
                "Worker {} does not exist, creating new PrefillCounter",
                worker_id
            );
            let new_counter = PrefillCounter::new(self.component.clone());
            new_counter.insert(request_id, new_tokens).await?;
            self.counters.insert(worker_id, new_counter);
        }

        Ok(())
    }

    pub async fn remove_prefill(&self, request_id: &str) -> Result<Option<usize>> {
        let Some((_request_id, worker_id)) = self.request_to_workers.remove(request_id) else {
            tracing::warn!("Request {} not found", request_id);
            return Ok(None);
        };

        if let Some(counter) = self.counters.get(&worker_id) {
            counter.remove(request_id).await
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
        let futures = FuturesUnordered::new();

        for entry in self.counters.iter() {
            let worker_id = *entry.key();
            let counter = entry.value().clone();
            futures.push(async move { (worker_id, counter.running_sum().await) });
        }

        futures.collect::<HashMap<_, _>>().await
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use std::collections::HashMap;
    use tokio::time::Duration;

    #[tokio::test]
    #[ignore]
    async fn test_prefill_counter_synchronization() -> Result<()> {
        // Initialize logging
        dynamo_runtime::logging::init();

        // Create runtime and distributed runtime
        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        // Create namespace and a single component
        let namespace = distributed.namespace("test_prefill_counter")?;
        let component = namespace
            .component("shared_counter")?
            .service_builder()
            .create()
            .await?;

        // Create two PrefillCounter instances using the same component (cloned)
        let counter1 = PrefillCounter::new(component.clone());
        let counter2 = PrefillCounter::new(component.clone());

        // Give some time for subscribers to initialize
        tokio::time::sleep(Duration::from_millis(2000)).await;

        // Track all request_ids and their token counts for verification
        let mut expected_tokens = HashMap::new();
        let tokens_per_request = 100;
        let requests_per_counter = 50;

        // Send 50 requests to counter1
        for i in 0..requests_per_counter {
            let request_id = format!("counter1_request_{}", i);
            counter1
                .insert(request_id.clone(), tokens_per_request)
                .await?;
            expected_tokens.insert(request_id, tokens_per_request);
        }

        // Send 50 requests to counter2
        for i in 0..requests_per_counter {
            let request_id = format!("counter2_request_{}", i);
            counter2
                .insert(request_id.clone(), tokens_per_request)
                .await?;
            expected_tokens.insert(request_id, tokens_per_request);
        }

        // Wait for synchronization
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Verify both counters have the same running sum
        let expected_sum = (requests_per_counter * 2) * tokens_per_request;
        let sum1 = counter1.running_sum().await;
        let sum2 = counter2.running_sum().await;

        assert_eq!(
            sum1, expected_sum,
            "Counter1 running sum mismatch. Expected: {}, Got: {}",
            expected_sum, sum1
        );
        assert_eq!(
            sum2, expected_sum,
            "Counter2 running sum mismatch. Expected: {}, Got: {}",
            expected_sum, sum2
        );

        // Verify both counters have all 100 requests
        let len1 = counter1.len().await;
        let len2 = counter2.len().await;
        assert_eq!(
            len1,
            requests_per_counter * 2,
            "Counter1 should have {} requests",
            requests_per_counter * 2
        );
        assert_eq!(
            len2,
            requests_per_counter * 2,
            "Counter2 should have {} requests",
            requests_per_counter * 2
        );

        // Spot check some individual requests on both counters
        for i in 0..5 {
            let request_id = format!("counter1_request_{}", i);
            let tokens1 = counter1.get(&request_id).await;
            let tokens2 = counter2.get(&request_id).await;
            assert_eq!(
                tokens1,
                Some(tokens_per_request),
                "Counter1 missing request {}",
                request_id
            );
            assert_eq!(
                tokens2,
                Some(tokens_per_request),
                "Counter2 missing request {}",
                request_id
            );
        }

        // Now remove all requests from both counters
        for i in 0..requests_per_counter {
            let request_id = format!("counter1_request_{}", i);
            counter1.remove(&request_id).await?;
        }

        for i in 0..requests_per_counter {
            let request_id = format!("counter2_request_{}", i);
            counter2.remove(&request_id).await?;
        }

        // Wait for removal synchronization
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Verify both counters have zero running sum
        let final_sum1 = counter1.running_sum().await;
        let final_sum2 = counter2.running_sum().await;
        assert_eq!(
            final_sum1, 0,
            "Counter1 should have zero running sum after removal"
        );
        assert_eq!(
            final_sum2, 0,
            "Counter2 should have zero running sum after removal"
        );

        // Verify both counters are empty
        assert!(counter1.is_empty().await, "Counter1 should be empty");
        assert!(counter2.is_empty().await, "Counter2 should be empty");

        // Shutdown runtime
        runtime.shutdown();

        Ok(())
    }
}
