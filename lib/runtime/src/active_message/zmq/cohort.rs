// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};
use tokio::sync::RwLock;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::InstanceId,
};

use super::client::ZmqActiveMessageClient;

#[derive(Debug)]
struct CohortState {
    leader_instance: InstanceId,
    worker_instances: Vec<InstanceId>,
    client: Arc<ZmqActiveMessageClient>,
}

#[derive(Debug, Clone)]
pub struct LeaderWorkerCohort {
    state: Arc<RwLock<CohortState>>,
}

impl LeaderWorkerCohort {
    pub fn new(
        leader_instance: InstanceId,
        worker_instances: Vec<InstanceId>,
        client: Arc<ZmqActiveMessageClient>,
    ) -> Self {
        let state = CohortState {
            leader_instance,
            worker_instances,
            client,
        };

        Self {
            state: Arc::new(RwLock::new(state)),
        }
    }

    pub async fn await_handler_on_all_workers(
        &self,
        handler: &str,
        timeout: Option<Duration>,
    ) -> Result<()> {
        let state = self.state.read().await;
        let workers = state.worker_instances.clone();
        let client = state.client.clone();
        drop(state);

        let mut tasks = Vec::new();

        for worker_id in workers {
            let client = client.clone();
            let handler = handler.to_string();

            tasks.push(tokio::spawn(async move {
                client.await_handler(worker_id, &handler, timeout).await
            }));
        }

        for (idx, task) in tasks.into_iter().enumerate() {
            match task.await {
                Ok(Ok(())) => {
                    debug!("Worker {} has handler '{}'", idx, handler);
                }
                Ok(Err(e)) => {
                    warn!(
                        "Worker {} failed to confirm handler '{}': {}",
                        idx, handler, e
                    );
                    return Err(e);
                }
                Err(e) => {
                    warn!("Worker {} task panicked: {}", idx, e);
                    return Err(anyhow::anyhow!("Worker task panicked: {}", e));
                }
            }
        }

        Ok(())
    }

    pub async fn broadcast_to_workers(&self, handler: &str, payload: Bytes) -> Result<()> {
        let state = self.state.read().await;
        let workers = state.worker_instances.clone();
        let client = state.client.clone();
        drop(state);

        for worker_id in workers {
            if let Err(e) = client
                .send_message(worker_id, handler, payload.clone())
                .await
            {
                warn!("Failed to send to worker {}: {}", worker_id, e);
            }
        }

        Ok(())
    }

    pub async fn broadcast_to_workers_with_acks(
        &self,
        handler: &str,
        payload: Bytes,
        timeout: Duration,
    ) -> Result<()> {
        let state = self.state.read().await;
        let workers = state.worker_instances.clone();
        let client = state.client.clone();
        drop(state);

        let mut ack_receivers = Vec::new();

        for worker_id in workers {
            let ack_id = Uuid::new_v4();
            let ack_rx = client.register_ack(ack_id, timeout).await?;
            ack_receivers.push((worker_id, ack_rx));

            let mut payload_with_ack = serde_json::from_slice::<serde_json::Value>(&payload)?;
            if let serde_json::Value::Object(ref mut map) = payload_with_ack {
                map.insert(
                    "_ack_id".to_string(),
                    serde_json::Value::String(ack_id.to_string()),
                );
            }

            let modified_payload = Bytes::from(serde_json::to_vec(&payload_with_ack)?);
            client
                .send_message(worker_id, handler, modified_payload)
                .await?;
        }

        for (worker_id, ack_rx) in ack_receivers {
            match tokio::time::timeout(timeout, ack_rx).await {
                Ok(Ok(())) => {
                    debug!("Received ACK from worker {}", worker_id);
                }
                Ok(Err(_)) => {
                    warn!("ACK channel closed for worker {}", worker_id);
                }
                Err(_) => {
                    warn!("Timeout waiting for ACK from worker {}", worker_id);
                }
            }
        }

        Ok(())
    }

    pub async fn list_workers(&self) -> Vec<InstanceId> {
        let state = self.state.read().await;
        state.worker_instances.clone()
    }

    pub fn leader_instance(&self) -> InstanceId {
        let state = self.state.blocking_read();
        state.leader_instance
    }
}
