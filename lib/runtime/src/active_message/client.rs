// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use serde::Serialize;
use std::time::Duration;
use tokio::sync::oneshot;
use uuid::Uuid;

use super::builder::{MessageBuilder, NeedsDeliveryMode};
use super::handler::{ActiveMessage, HandlerId, InstanceId};
use super::responses::{
    HealthCheckResponse, JoinCohortResponse, ListHandlersResponse, RegisterServiceResponse,
    WaitForHandlerResponse,
};
use super::utils::extract_host;

/// Trait for types that can be converted to message payload bytes
pub trait IntoPayload: Send {
    fn into_payload(self) -> Result<Bytes>;
}

// Implement for Bytes (pass-through)
impl IntoPayload for Bytes {
    fn into_payload(self) -> Result<Bytes> {
        Ok(self)
    }
}

// Implement for references to Serialize types
impl<T: Serialize + Sync> IntoPayload for &T {
    fn into_payload(self) -> Result<Bytes> {
        let serialized = serde_json::to_vec(self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize payload: {}", e))?;
        Ok(Bytes::from(serialized))
    }
}

pub type Endpoint = String;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeerInfo {
    pub instance_id: InstanceId,
    /// Primary endpoint (TCP) - maintained for backward compatibility
    pub endpoint: Endpoint,
    /// Optional TCP endpoint for cross-host communication
    pub tcp_endpoint: Option<Endpoint>,
    /// Optional IPC endpoint for same-host optimization
    pub ipc_endpoint: Option<Endpoint>,
}

impl PeerInfo {
    /// Create PeerInfo with single endpoint (backward compatibility)
    pub fn new(instance_id: InstanceId, endpoint: impl Into<String>) -> Self {
        let endpoint = endpoint.into();
        Self {
            instance_id,
            endpoint: endpoint.clone(),
            tcp_endpoint: Some(endpoint),
            ipc_endpoint: None,
        }
    }

    /// Create PeerInfo with both TCP and IPC endpoints
    pub fn new_dual(
        instance_id: InstanceId,
        tcp_endpoint: Option<Endpoint>,
        ipc_endpoint: Option<Endpoint>,
    ) -> Self {
        // Use TCP endpoint as primary for backward compatibility
        let primary_endpoint = tcp_endpoint
            .as_ref()
            .cloned()
            .unwrap_or_else(|| "tcp://unknown:0".to_string());

        Self {
            instance_id,
            endpoint: primary_endpoint,
            tcp_endpoint,
            ipc_endpoint,
        }
    }

    /// Get the best endpoint for connection based on local preferences
    pub fn select_endpoint(&self, my_endpoint: &str) -> Option<&Endpoint> {
        // Prefer IPC for same-host communication if available
        if let Some(ref ipc_ep) = self.ipc_endpoint
            && self.is_local(my_endpoint)
        {
            return Some(ipc_ep);
        }

        // Fall back to TCP endpoint
        self.tcp_endpoint.as_ref().or(Some(&self.endpoint))
    }

    pub fn is_local(&self, my_endpoint: &str) -> bool {
        // Extract our host from our endpoint
        let my_host = match extract_host(my_endpoint) {
            Some(host) => host,
            None => return false,
        };

        // If peer has IPC endpoint, check if we're on the same host as their TCP endpoint
        if self.ipc_endpoint.is_some() {
            let peer_tcp_endpoint = self.tcp_endpoint.as_ref().unwrap_or(&self.endpoint);
            if let Some(peer_host) = extract_host(peer_tcp_endpoint) {
                return my_host == peer_host;
            }
        }

        // Fall back to host comparison using available TCP endpoint
        let peer_endpoint = self.tcp_endpoint.as_ref().unwrap_or(&self.endpoint);

        // IPC endpoints are always local
        if peer_endpoint.starts_with("ipc://") {
            return true;
        }

        // Compare hosts for TCP endpoints
        if let Some(peer_host) = extract_host(peer_endpoint) {
            my_host == peer_host
        } else {
            false
        }
    }
}

/// Validate that handler name doesn't use reserved system handler prefix
pub(crate) fn validate_handler_name(handler: &str) -> Result<()> {
    if handler.starts_with('_') {
        anyhow::bail!(
            "Cannot directly call system handler '{}'. Use client convenience methods instead: health_check(), register_service(), list_handlers(), await_handler()",
            handler
        );
    }
    Ok(())
}

#[async_trait]
pub trait ActiveMessageClient: Send + Sync + std::fmt::Debug {
    fn instance_id(&self) -> InstanceId;

    fn endpoint(&self) -> &str;

    async fn send_message(&self, target: InstanceId, handler: &str, payload: Bytes) -> Result<()>;

    async fn broadcast_message(&self, handler: &str, payload: Bytes) -> Result<()>;

    async fn list_peers(&self) -> Result<Vec<PeerInfo>>;

    async fn connect_to_peer(&self, peer: PeerInfo) -> Result<()>;

    async fn disconnect_from_peer(&self, instance_id: InstanceId) -> Result<()>;

    async fn await_handler(
        &self,
        instance_id: InstanceId,
        handler: &str,
        timeout: Option<Duration>,
    ) -> Result<bool>;

    async fn list_handlers(&self, instance_id: InstanceId) -> Result<Vec<HandlerId>>;

    /// Send a health check request and get the response
    async fn health_check(&self, instance_id: InstanceId) -> Result<HealthCheckResponse>
    where
        Self: Sized,
    {
        let status = self
            .system_active_message("_health_check")
            .expect_response::<HealthCheckResponse>()
            .send(instance_id)
            .await?;

        status.await_response().await
    }

    /// Register a service with a remote instance
    async fn register_service(&self, instance_id: InstanceId, service: PeerInfo) -> Result<bool>
    where
        Self: Sized,
    {
        // Create payload with dual endpoint support
        let mut payload = serde_json::json!({
            "instance_id": service.instance_id.to_string(),
        });

        // Add endpoints based on what's available
        if service.tcp_endpoint.is_some() || service.ipc_endpoint.is_some() {
            // Use new dual endpoint format
            if let Some(ref tcp_ep) = service.tcp_endpoint {
                payload["tcp_endpoint"] = serde_json::Value::String(tcp_ep.clone());
            }
            if let Some(ref ipc_ep) = service.ipc_endpoint {
                payload["ipc_endpoint"] = serde_json::Value::String(ipc_ep.clone());
            }
        } else {
            // Fallback to legacy endpoint format for backward compatibility
            payload["endpoint"] = serde_json::Value::String(service.endpoint);
        }

        let status = self
            .system_active_message("_register_service")
            .payload(payload)?
            .expect_response::<RegisterServiceResponse>()
            .send(instance_id)
            .await?;

        let response: RegisterServiceResponse = status.await_response().await?;
        Ok(response.registered)
    }

    /// Join a leader-worker cohort
    async fn join_cohort(
        &self,
        leader_instance_id: InstanceId,
        rank: Option<usize>,
    ) -> Result<JoinCohortResponse>
    where
        Self: Sized,
    {
        let payload = serde_json::json!({
            "instance_id": self.instance_id().to_string(),
            "rank": rank,
        });

        let status = self
            .system_active_message("_join_cohort")
            .payload(payload)?
            .expect_response::<JoinCohortResponse>()
            .send(leader_instance_id)
            .await?;

        status.await_response().await
    }

    // New builder pattern API
    /// Create an active message builder
    fn active_message(&self, handler: &str) -> Result<MessageBuilder<'_>>
    where
        Self: Sized,
    {
        MessageBuilder::new(self, handler)
    }

    /// Internal method to create active message builder for system handlers (bypasses validation)
    fn system_active_message(&self, handler: &str) -> MessageBuilder<'_, NeedsDeliveryMode>
    where
        Self: Sized,
    {
        // Use the unchecked constructor for system handlers
        MessageBuilder::new_unchecked(self, handler)
    }

    /// Send a message with payload serialization - DEPRECATED
    ///
    /// # Deprecated
    /// Use `client.active_message(handler).payload(data).target_instance(target).execute()` instead for better type safety and features.
    #[deprecated(
        note = "Use client.active_message(handler).payload(data).target_instance(target).execute() instead"
    )]
    async fn send<P: IntoPayload>(
        &self,
        target: InstanceId,
        handler: &str,
        payload: P,
    ) -> Result<()>
    where
        Self: Sized,
    {
        validate_handler_name(handler)?;
        let bytes = payload.into_payload()?;
        self.send_message(target, handler, bytes).await
    }

    /// Broadcast a message with payload serialization - DEPRECATED
    ///
    /// # Deprecated
    /// Use `client.active_message(handler).payload(data).target_instance(target).execute()` for each target instead.
    #[deprecated(note = "Use active_message builder pattern instead")]
    async fn broadcast<P: IntoPayload>(&self, handler: &str, payload: P) -> Result<()>
    where
        Self: Sized,
    {
        validate_handler_name(handler)?;
        let bytes = payload.into_payload()?;
        self.broadcast_message(handler, bytes).await
    }

    // Internal methods for builder pattern support
    /// Send raw ActiveMessage (internal use by builder)
    async fn send_raw_message(&self, target: InstanceId, message: ActiveMessage) -> Result<()>;

    /// Register for acceptance notification (internal use by builder)
    async fn register_acceptance(
        &self,
        message_id: Uuid,
        sender: oneshot::Sender<()>,
    ) -> Result<()>;

    /// Register for response notification (internal use by builder)
    async fn register_response(
        &self,
        message_id: Uuid,
        sender: oneshot::Sender<Bytes>,
    ) -> Result<()>;

    /// Register for ACK/NACK notification (internal use by builder)
    async fn register_ack(
        &self,
        ack_id: Uuid,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<Result<(), String>>>;

    /// Check if a peer has an incoming connection to us (internal use by builder)
    async fn has_incoming_connection_from(&self, instance_id: InstanceId) -> bool;

    /// Clone this client as an Arc<dyn ActiveMessageClient> for use in contexts
    fn clone_as_arc(&self) -> std::sync::Arc<dyn ActiveMessageClient>;
}
