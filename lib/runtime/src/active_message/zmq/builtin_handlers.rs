// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, broadcast};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::{
        ActiveMessage, ActiveMessageContext, HandlerEvent, HandlerId, NoReturnHandler,
        ResponseHandler,
    },
    responses::{
        HealthCheckResponse, JoinCohortResponse, ListHandlersResponse, RegisterServiceResponse,
        RemoveServiceResponse, RequestShutdownResponse, WaitForHandlerResponse,
    },
};

use super::manager::ManagerState;

#[derive(Debug)]
pub struct RegisterServiceHandler {
    client: Arc<super::client::ZmqActiveMessageClient>,
}

impl RegisterServiceHandler {
    pub fn new(client: Arc<super::client::ZmqActiveMessageClient>) -> Self {
        Self { client }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RegisterServicePayload {
    instance_id: String,
    /// Legacy single endpoint field for backward compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint: Option<String>,
    /// TCP endpoint for cross-host communication
    #[serde(skip_serializing_if = "Option::is_none")]
    tcp_endpoint: Option<String>,
    /// IPC endpoint for same-host optimization
    #[serde(skip_serializing_if = "Option::is_none")]
    ipc_endpoint: Option<String>,
}

#[async_trait]
impl ResponseHandler for RegisterServiceHandler {
    async fn handle(&self, input: Bytes, _ctx: ActiveMessageContext) -> Result<Bytes> {
        let payload: RegisterServicePayload = serde_json::from_slice(&input)?;

        let instance_id = uuid::Uuid::parse_str(&payload.instance_id)?;

        // Create PeerInfo with dual endpoint support, handling backward compatibility
        let peer = if payload.tcp_endpoint.is_some() || payload.ipc_endpoint.is_some() {
            // New dual endpoint format
            PeerInfo::new_dual(
                instance_id,
                payload.tcp_endpoint.clone(),
                payload.ipc_endpoint.clone(),
            )
        } else if let Some(ref endpoint) = payload.endpoint {
            // Legacy single endpoint format
            PeerInfo::new(instance_id, endpoint.clone())
        } else {
            anyhow::bail!("No endpoint provided in registration payload");
        };

        let registration_result = self.client.connect_to_peer(peer).await;
        let registered = registration_result.is_ok();

        if registered {
            let endpoint_desc = match (&payload.tcp_endpoint, &payload.ipc_endpoint) {
                (Some(tcp), Some(ipc)) => format!("TCP: {}, IPC: {}", tcp, ipc),
                (Some(tcp), None) => format!("TCP: {}", tcp),
                (None, Some(ipc)) => format!("IPC: {}", ipc),
                (None, None) => payload
                    .endpoint
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
                    .clone(),
            };
            info!(
                "Registered service {} at {}",
                payload.instance_id, endpoint_desc
            );
        } else {
            debug!(
                "Failed to register service {}: {:?}",
                payload.instance_id, registration_result
            );
        }

        // For response, use the primary endpoint (TCP if available, otherwise legacy endpoint)
        let response_endpoint = payload
            .tcp_endpoint
            .or(payload.endpoint)
            .unwrap_or_else(|| "unknown".to_string());

        let response = RegisterServiceResponse {
            registered,
            instance_id: payload.instance_id,
            endpoint: response_endpoint,
        };

        let serialized = serde_json::to_vec(&response)?;
        Ok(Bytes::from(serialized))
    }

    fn name(&self) -> &str {
        "_register_service"
    }
}

#[derive(Debug)]
pub struct ListHandlersHandler {
    state: Arc<RwLock<ManagerState>>,
}

impl ListHandlersHandler {
    pub fn new(state: Arc<RwLock<ManagerState>>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl ResponseHandler for ListHandlersHandler {
    async fn handle(&self, _input: Bytes, _ctx: ActiveMessageContext) -> Result<Bytes> {
        let state = self.state.read().await;
        let handlers: Vec<HandlerId> = state.handlers.keys().cloned().collect();
        drop(state);

        debug!("Available handlers: {:?}", handlers);

        let response = ListHandlersResponse { handlers };
        let serialized = serde_json::to_vec(&response)?;
        Ok(Bytes::from(serialized))
    }

    fn name(&self) -> &str {
        "_list_handlers"
    }
}

#[derive(Debug)]
pub struct WaitForHandlerHandler {
    state: Arc<RwLock<ManagerState>>,
    handler_events_tx: broadcast::Sender<HandlerEvent>,
}

impl WaitForHandlerHandler {
    pub fn new(
        state: Arc<RwLock<ManagerState>>,
        handler_events_tx: broadcast::Sender<HandlerEvent>,
    ) -> Self {
        Self {
            state,
            handler_events_tx,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct WaitForHandlerPayload {
    handler_name: String,
    timeout_ms: Option<u64>,
}

#[async_trait]
impl ResponseHandler for WaitForHandlerHandler {
    async fn handle(&self, input: Bytes, _ctx: ActiveMessageContext) -> Result<Bytes> {
        let payload: WaitForHandlerPayload = serde_json::from_slice(&input)?;

        let state = self.state.read().await;
        if state.handlers.contains_key(&payload.handler_name) {
            debug!("Handler '{}' already registered", payload.handler_name);

            let response = WaitForHandlerResponse {
                handler_name: payload.handler_name,
                available: true,
            };
            let serialized = serde_json::to_vec(&response)?;
            return Ok(Bytes::from(serialized));
        }
        drop(state);

        let mut rx = self.handler_events_tx.subscribe();

        let timeout = payload
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(Duration::from_secs(30));

        let deadline = tokio::time::Instant::now() + timeout;
        let mut handler_found = false;

        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Ok(HandlerEvent::Registered { name, .. }) => {
                            if name == payload.handler_name {
                                debug!("Handler '{}' is now registered", payload.handler_name);
                                handler_found = true;
                                break;
                            }
                        }
                        Ok(HandlerEvent::Deregistered { .. }) => {}
                        Err(_) => {
                            anyhow::bail!("Handler event channel closed");
                        }
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    debug!(
                        "Timeout waiting for handler '{}'",
                        payload.handler_name
                    );
                    break;
                }
            }
        }

        let response = WaitForHandlerResponse {
            handler_name: payload.handler_name.clone(),
            available: handler_found,
        };

        if handler_found {
            let serialized = serde_json::to_vec(&response)?;
            Ok(Bytes::from(serialized))
        } else {
            // Return error response - this will be converted to ResponseEnvelope::Err
            anyhow::bail!("Timeout waiting for handler '{}'", payload.handler_name)
        }
    }

    fn name(&self) -> &str {
        "_wait_for_handler"
    }
}

#[derive(Debug)]
pub struct HealthCheckHandler;

#[async_trait]
impl ResponseHandler for HealthCheckHandler {
    async fn handle(&self, _input: Bytes, _ctx: ActiveMessageContext) -> Result<Bytes> {
        debug!("Health check received");

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let response = HealthCheckResponse {
            status: "ok".to_string(),
            timestamp,
        };

        let serialized = serde_json::to_vec(&response)?;
        Ok(Bytes::from(serialized))
    }

    fn name(&self) -> &str {
        "_health_check"
    }
}

#[derive(Debug)]
pub struct AckHandler {
    client: Arc<super::client::ZmqActiveMessageClient>,
}

impl AckHandler {
    pub fn new(client: Arc<super::client::ZmqActiveMessageClient>) -> Self {
        Self { client }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct AckPayload {
    ack_id: String,
    status: AckStatus,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
enum AckStatus {
    Ack,
    Nack { error: String },
}

#[async_trait]
impl NoReturnHandler for AckHandler {
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext) {
        let payload: AckPayload = match serde_json::from_slice(&input) {
            Ok(payload) => payload,
            Err(e) => {
                tracing::error!("Failed to deserialize ACK payload: {}", e);
                return;
            }
        };

        let ack_id = match uuid::Uuid::parse_str(&payload.ack_id) {
            Ok(id) => id,
            Err(e) => {
                tracing::error!("Invalid ACK ID '{}': {}", payload.ack_id, e);
                return;
            }
        };

        match payload.status {
            AckStatus::Ack => {
                if let Err(e) = self.client.complete_ack(ack_id, ctx.sender_instance).await {
                    tracing::error!("Failed to complete ACK for {}: {}", ack_id, e);
                } else {
                    debug!("Processed ACK for {}", ack_id);
                }
            }
            AckStatus::Nack { error } => {
                if let Err(e) = self
                    .client
                    .complete_nack(ack_id, ctx.sender_instance, error.clone())
                    .await
                {
                    tracing::error!("Failed to complete NACK for {}: {}", ack_id, e);
                } else {
                    debug!("Processed NACK for {}: {}", ack_id, error);
                }
            }
        }
    }

    fn name(&self) -> &str {
        "_ack"
    }
}

// JoinCohortHandler for cohort membership management
#[derive(Debug)]
pub struct JoinCohortHandler {
    cohort: Arc<super::cohort::LeaderWorkerCohort>,
}

impl JoinCohortHandler {
    pub fn new(cohort: Arc<super::cohort::LeaderWorkerCohort>) -> Self {
        Self { cohort }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct JoinCohortPayload {
    instance_id: String,
    rank: Option<usize>, // Worker's self-reported rank
}

#[async_trait]
impl ResponseHandler for JoinCohortHandler {
    async fn handle(&self, input: Bytes, _ctx: ActiveMessageContext) -> Result<Bytes> {
        let payload: JoinCohortPayload = serde_json::from_slice(&input)?;
        let instance_id = uuid::Uuid::parse_str(&payload.instance_id)?;

        // Validate rank consistency and add worker
        let join_response = match self
            .cohort
            .validate_and_add_worker(instance_id, payload.rank)
            .await
        {
            Ok(position) => {
                JoinCohortResponse {
                    accepted: true,
                    reason: None,
                    position: Some(position),
                    expected_rank: payload.rank, // Echo back the rank
                }
            }
            Err(e) => JoinCohortResponse {
                accepted: false,
                reason: Some(e.to_string()),
                position: None,
                expected_rank: None,
            },
        };

        let serialized = serde_json::to_vec(&join_response)?;
        Ok(Bytes::from(serialized))
    }

    fn name(&self) -> &str {
        "_join_cohort"
    }
}

// RemoveServiceHandler for graceful shutdown
#[derive(Debug)]
pub struct RemoveServiceHandler {
    cohort: Arc<super::cohort::LeaderWorkerCohort>,
}

impl RemoveServiceHandler {
    pub fn new(cohort: Arc<super::cohort::LeaderWorkerCohort>) -> Self {
        Self { cohort }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RemoveServicePayload {
    instance_id: String,
    rank: Option<usize>,
}

#[async_trait]
impl ResponseHandler for RemoveServiceHandler {
    async fn handle(&self, input: Bytes, _ctx: ActiveMessageContext) -> Result<Bytes> {
        let payload: RemoveServicePayload = serde_json::from_slice(&input)?;
        let instance_id = uuid::Uuid::parse_str(&payload.instance_id)?;

        let removed = self.cohort.remove_worker(instance_id).await?;

        let response = RemoveServiceResponse {
            removed,
            instance_id: payload.instance_id,
            rank: payload.rank,
        };

        let serialized = serde_json::to_vec(&response)?;
        Ok(Bytes::from(serialized))
    }

    fn name(&self) -> &str {
        "_remove_service"
    }
}

// RequestShutdownHandler for leader-initiated shutdown
#[derive(Debug)]
pub struct RequestShutdownHandler {
    manager_state: Arc<RwLock<super::manager::ManagerState>>,
    cancel_token: CancellationToken,
}

impl RequestShutdownHandler {
    pub fn new(
        manager_state: Arc<RwLock<super::manager::ManagerState>>,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            manager_state,
            cancel_token,
        }
    }
}

#[async_trait]
impl ResponseHandler for RequestShutdownHandler {
    async fn handle(&self, _input: Bytes, ctx: ActiveMessageContext) -> Result<Bytes> {
        info!("Received shutdown request from leader");

        // Start shutdown process in background
        let manager_state = self.manager_state.clone();
        let cancel_token = self.cancel_token.clone();
        let client_id = ctx.client().instance_id();

        tokio::spawn(async move {
            // 1. Stop accepting new tasks (close all TaskTrackers)
            let state = manager_state.read().await;
            for (name, entry) in &state.handlers {
                debug!("Closing task tracker for handler: {}", name);
                entry.task_tracker.cancellation_token().cancel(); // Prevents new tasks
            }
            drop(state);

            // 2. Drain existing tasks
            let state = manager_state.read().await;
            for (name, entry) in &state.handlers {
                debug!("Draining tasks for handler: {}", name);
                entry.task_tracker.join().await;
            }
            let leader_id = state.client.instance_id(); // Assuming we can get the leader ID
            let client = state.client.clone();
            drop(state);

            // 3. Send remove_service to leader
            debug!("Sending remove service notification to leader");
            let payload = serde_json::json!({
                "instance_id": client_id.to_string(),
                "rank": std::env::var("RANK").ok().and_then(|r| r.parse::<usize>().ok())
            });

            if let Err(e) = client
                .system_active_message("_remove_service")
                .payload(payload)
                .expect("Valid payload")
                .fire_and_forget(leader_id)
                .await
            {
                warn!("Failed to send remove_service to leader: {}", e);
            }

            // 4. Trigger local shutdown
            info!("Triggering worker shutdown");
            cancel_token.cancel();
        });

        // Acknowledge the shutdown request
        let response = RequestShutdownResponse { acknowledged: true };
        let serialized = serde_json::to_vec(&response)?;
        Ok(Bytes::from(serialized))
    }

    fn name(&self) -> &str {
        "_request_shutdown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_message::{
        client::ActiveMessageClient, handler::ActiveMessage, response::SingleResponseSender,
    };
    use std::sync::Arc;
    use tokio::sync::{RwLock, oneshot};
    use uuid::Uuid;

    #[derive(Debug, Clone)]
    struct MockClient {
        instance_id: Uuid,
        endpoint: String,
    }

    #[async_trait]
    impl ActiveMessageClient for MockClient {
        fn instance_id(&self) -> uuid::Uuid {
            self.instance_id
        }

        fn endpoint(&self) -> &str {
            &self.endpoint
        }

        async fn send_message(
            &self,
            _target: uuid::Uuid,
            _handler: &str,
            _payload: bytes::Bytes,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn broadcast_message(
            &self,
            _handler: &str,
            _payload: bytes::Bytes,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn list_peers(&self) -> anyhow::Result<Vec<crate::active_message::client::PeerInfo>> {
            Ok(vec![])
        }

        async fn connect_to_peer(
            &self,
            _peer: crate::active_message::client::PeerInfo,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn disconnect_from_peer(&self, _instance_id: uuid::Uuid) -> anyhow::Result<()> {
            Ok(())
        }

        async fn await_handler(
            &self,
            _instance_id: uuid::Uuid,
            _handler: &str,
            _timeout: Option<std::time::Duration>,
        ) -> anyhow::Result<bool> {
            Ok(true)
        }

        async fn list_handlers(&self, _instance_id: uuid::Uuid) -> anyhow::Result<Vec<String>> {
            Ok(vec![])
        }

        async fn send_raw_message(
            &self,
            _target: uuid::Uuid,
            _message: ActiveMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn register_acceptance(
            &self,
            _message_id: Uuid,
            _sender: oneshot::Sender<()>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn register_response(
            &self,
            _message_id: Uuid,
            _sender: oneshot::Sender<bytes::Bytes>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn register_ack(
            &self,
            _ack_id: Uuid,
            _timeout: std::time::Duration,
        ) -> anyhow::Result<oneshot::Receiver<Result<(), String>>> {
            let (_tx, rx) = oneshot::channel();
            Ok(rx)
        }

        async fn has_incoming_connection_from(&self, _instance_id: uuid::Uuid) -> bool {
            false // Mock always returns false
        }

        fn clone_as_arc(&self) -> std::sync::Arc<dyn ActiveMessageClient> {
            Arc::new(MockClient {
                instance_id: self.instance_id,
                endpoint: self.endpoint.clone(),
            })
        }
    }

    #[tokio::test]
    async fn test_health_check_handler_response() {
        let handler = HealthCheckHandler;
        let mock_client = MockClient {
            instance_id: Uuid::new_v4(),
            endpoint: "test://localhost".to_string(),
        };

        let message = ActiveMessage {
            message_id: Uuid::new_v4(),
            handler_name: "_health_check".to_string(),
            sender_instance: Uuid::new_v4(),
            payload: bytes::Bytes::new(),
            metadata: serde_json::Value::Null,
        };

        // Create context for new signature
        let ctx = ActiveMessageContext::new(
            message.message_id,
            message.sender_instance,
            message.handler_name.clone(),
            message.metadata.clone(),
            Arc::new(mock_client),
            None,
        );

        // Test the new response handler directly
        let result = handler.handle(message.payload, ctx).await;
        assert!(result.is_ok());
        let response_bytes = result.unwrap();
        let response: HealthCheckResponse = serde_json::from_slice(&response_bytes).unwrap();
        assert_eq!(response.status, "ok");
    }

    #[tokio::test]
    async fn test_list_handlers_response_structure() {
        let response = ListHandlersResponse {
            handlers: vec!["handler1".to_string(), "handler2".to_string()],
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: ListHandlersResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert_eq!(deserialized.handlers.len(), 2);
        assert!(deserialized.handlers.contains(&"handler1".to_string()));
        assert!(deserialized.handlers.contains(&"handler2".to_string()));
    }

    #[tokio::test]
    async fn test_health_check_response_structure() {
        let response = HealthCheckResponse {
            status: "ok".to_string(),
            timestamp: 1234567890,
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: HealthCheckResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert_eq!(deserialized.status, "ok");
        assert_eq!(deserialized.timestamp, 1234567890);
    }

    #[tokio::test]
    async fn test_register_service_response_structure() {
        let response = RegisterServiceResponse {
            registered: true,
            instance_id: "test-id".to_string(),
            endpoint: "test://endpoint".to_string(),
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: RegisterServiceResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert!(deserialized.registered);
        assert_eq!(deserialized.instance_id, "test-id");
        assert_eq!(deserialized.endpoint, "test://endpoint");
    }

    #[tokio::test]
    async fn test_wait_for_handler_response_structure() {
        let response = WaitForHandlerResponse {
            handler_name: "test_handler".to_string(),
            available: true,
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: WaitForHandlerResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert_eq!(deserialized.handler_name, "test_handler");
        assert!(deserialized.available);
    }

    #[tokio::test]
    async fn test_join_cohort_response_structure() {
        use crate::active_message::responses::JoinCohortResponse;

        let response = JoinCohortResponse {
            accepted: true,
            reason: None,
            position: Some(0),
            expected_rank: Some(1),
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: JoinCohortResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert!(deserialized.accepted);
        assert_eq!(deserialized.reason, None);
        assert_eq!(deserialized.position, Some(0));
        assert_eq!(deserialized.expected_rank, Some(1));

        // Test rejection case
        let rejection_response = JoinCohortResponse {
            accepted: false,
            reason: Some("Cohort is full".to_string()),
            position: None,
            expected_rank: None,
        };

        let serialized = serde_json::to_string(&rejection_response).expect("Should serialize");
        let deserialized: JoinCohortResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert!(!deserialized.accepted);
        assert_eq!(deserialized.reason, Some("Cohort is full".to_string()));
        assert_eq!(deserialized.position, None);
        assert_eq!(deserialized.expected_rank, None);
    }

    #[tokio::test]
    async fn test_remove_service_response_structure() {
        use crate::active_message::responses::RemoveServiceResponse;

        let response = RemoveServiceResponse {
            removed: true,
            instance_id: "worker-123".to_string(),
            rank: Some(2),
        };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: RemoveServiceResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert!(deserialized.removed);
        assert_eq!(deserialized.instance_id, "worker-123");
        assert_eq!(deserialized.rank, Some(2));

        // Test without rank
        let no_rank_response = RemoveServiceResponse {
            removed: false,
            instance_id: "worker-456".to_string(),
            rank: None,
        };

        let serialized = serde_json::to_string(&no_rank_response).expect("Should serialize");
        let deserialized: RemoveServiceResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert!(!deserialized.removed);
        assert_eq!(deserialized.instance_id, "worker-456");
        assert_eq!(deserialized.rank, None);
    }

    #[tokio::test]
    async fn test_request_shutdown_response_structure() {
        use crate::active_message::responses::RequestShutdownResponse;

        let response = RequestShutdownResponse { acknowledged: true };

        let serialized = serde_json::to_string(&response).expect("Should serialize");
        let deserialized: RequestShutdownResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert!(deserialized.acknowledged);

        // Test negative acknowledgment
        let nack_response = RequestShutdownResponse {
            acknowledged: false,
        };

        let serialized = serde_json::to_string(&nack_response).expect("Should serialize");
        let deserialized: RequestShutdownResponse =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert!(!deserialized.acknowledged);
    }
}
