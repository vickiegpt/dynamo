// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-agnostic message router that handles business logic for active message processing.
//!
//! This module contains all the message routing, response correlation, and peer management
//! logic that was previously embedded in the ZMQ transport layer. By separating this
//! business logic from transport concerns, we achieve:
//!
//! - Clean separation between transport and application logic
//! - Easier testing of routing logic without transport dependencies
//! - Support for multiple transport implementations
//! - Better maintainability and modularity

use anyhow::Result;
use bytes::Bytes;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::active_message::{
    client::ActiveMessageClient,
    dispatcher::{DispatchMessage, SenderIdentity},
    handler::{ActiveMessage, InstanceId},
    response_manager::SharedResponseManager,
};

/// Transport-agnostic message router that processes active messages.
///
/// The MessageRouter is responsible for:
/// - Routing messages based on handler names and metadata
/// - Handling response correlation (ACK/NACK/responses)
/// - Managing peer auto-registration
/// - Converting raw messages to dispatch format
/// - Coordinating with the SharedResponseManager for response completion
pub struct MessageRouter {
    /// Shared response manager for completing responses, ACKs, and receipts
    response_manager: SharedResponseManager,
    /// Client for auto-registration and peer management
    client: Arc<dyn ActiveMessageClient>,
    /// Channel to send processed messages to the dispatcher
    dispatch_tx: mpsc::Sender<DispatchMessage>,
}

impl MessageRouter {
    /// Create a new MessageRouter with the given dependencies
    pub fn new(
        response_manager: SharedResponseManager,
        client: Arc<dyn ActiveMessageClient>,
        dispatch_tx: mpsc::Sender<DispatchMessage>,
    ) -> Self {
        Self {
            response_manager,
            client,
            dispatch_tx,
        }
    }

    /// Process an incoming active message.
    ///
    /// This method handles all message routing logic that was previously
    /// embedded in the ZMQ manager's receive loop. It determines message
    /// type and routes accordingly:
    ///
    /// - Internal messages (_accept, _response) are handled directly
    /// - Response messages are correlated with pending requests
    /// - Regular messages are auto-registered and forwarded to dispatcher
    pub async fn route_message(&self, message: ActiveMessage) -> Result<()> {
        // Handle special internal messages that bypass the dispatcher
        if message.handler_name == "_accept" {
            return self.handle_acceptance(message).await;
        }

        // Check if this is a response message based on metadata (v2 pattern)
        // v2 handlers send responses with original handler name but include _response_to metadata
        if let Some(response_to) = message
            .metadata
            .get("_response_to")
            .and_then(|v| v.as_str())
        {
            let response_to = response_to.to_string(); // Clone before moving message
            debug!("Received response message for request {}", response_to);
            return self.handle_response_unified(&response_to, message).await;
        }

        // Legacy v1 response pattern (uses "_response" handler name)
        if message.handler_name == "_response" {
            return self.handle_response(message).await;
        }

        // Note: Auto-registration is handled by the transport layer (ZMQ manager)
        // before messages reach the MessageRouter. This keeps the router transport-agnostic.

        // Convert ActiveMessage to DispatchMessage
        let dispatch_message = self.convert_to_dispatch_message(message);

        // Forward to MessageDispatcher
        if let Err(e) = self.dispatch_tx.send(dispatch_message).await {
            error!("Failed to send message to dispatcher: {}", e);
        }

        Ok(())
    }

    /// Handle acceptance messages for pending requests
    async fn handle_acceptance(&self, message: ActiveMessage) -> Result<()> {
        if let Some(accept_for_str) = message.metadata.get("_accept_for").and_then(|v| v.as_str())
            && let Ok(accept_id) = Uuid::parse_str(accept_for_str)
        {
            self.response_manager.complete_acceptance(accept_id);
            return Ok(());
        }
        error!("Invalid acceptance message: {:?}", message);
        Ok(())
    }

    /// Handle unified response messages (with metadata-based routing)
    async fn handle_response_unified(
        &self,
        response_to: &str,
        message: ActiveMessage,
    ) -> Result<()> {
        let response_id = Uuid::parse_str(response_to)?;

        debug!(
            "Handling v2 response to {} from {}",
            response_id, message.sender_instance
        );
        debug!("Response payload size: {} bytes", message.payload.len());

        // Try to parse the payload as JSON to determine if it's an ACK/NACK or full response
        if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&message.payload) {
            debug!("Parsed response as JSON: {:?}", json_value);
            if let Some(status) = json_value.get("status").and_then(|s| s.as_str()) {
                debug!("Found status field: {}", status);
                match status {
                    "ok" => {
                        // This is an ACK
                        debug!("Completing as ACK");
                        self.response_manager.complete_ack(response_id, Ok(()));
                        return Ok(());
                    }
                    "error" => {
                        // This is a NACK
                        let error_msg = json_value
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown error")
                            .to_string();
                        debug!("Completing as NACK: {}", error_msg);
                        self.response_manager
                            .complete_ack(response_id, Err(error_msg));
                        return Ok(());
                    }
                    _ => {
                        // Not an ACK/NACK status, treat as full response
                        debug!("Unknown status value, treating as full response");
                    }
                }
            } else {
                debug!("No status field found, treating as full response");
            }
        } else {
            debug!("Could not parse as JSON, treating as full response");
        }

        // This is a full response message
        debug!(
            "Completing as full response with {} bytes",
            message.payload.len()
        );
        self.response_manager
            .complete_response(response_id, message.payload);
        Ok(())
    }

    /// Handle legacy v1 response messages
    async fn handle_response(&self, message: ActiveMessage) -> Result<()> {
        if let Some(response_to_str) = message
            .metadata
            .get("_response_to")
            .and_then(|v| v.as_str())
            && let Ok(response_id) = Uuid::parse_str(response_to_str)
        {
            debug!(
                "Handling legacy response to {} from {}",
                response_id, message.sender_instance
            );

            // Try to parse the payload as JSON to determine if it's an ACK/NACK or full response
            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&message.payload) {
                // Check if this is an ACK/NACK message based on the status field
                if let Some(status) = json_value.get("status").and_then(|s| s.as_str()) {
                    match status {
                        "ok" => {
                            // This is an ACK - complete it as an ACK
                            debug!("Routing ACK response {} to complete_ack", response_id);
                            self.response_manager.complete_ack(response_id, Ok(()));
                            return Ok(());
                        }
                        "error" => {
                            // This is a NACK - extract error message and complete as NACK
                            let error_msg = json_value
                                .get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown error")
                                .to_string();
                            debug!(
                                "Routing NACK response {} to complete_nack: {}",
                                response_id, error_msg
                            );
                            self.response_manager
                                .complete_ack(response_id, Err(error_msg));
                            return Ok(());
                        }
                        _ => {
                            // Unknown status, treat as full response
                            debug!(
                                "Unknown status '{}' in response {}, treating as full response",
                                status, response_id
                            );
                        }
                    }
                }
                // JSON but no status field - this is a full response
                debug!(
                    "Routing full JSON response {} to complete_response",
                    response_id
                );
            } else {
                // Not JSON - definitely a full response
                debug!(
                    "Routing non-JSON response {} to complete_response",
                    response_id
                );
            }

            // Not an ACK/NACK or couldn't parse as JSON - treat as full response
            self.response_manager
                .complete_response(response_id, message.payload);
            return Ok(());
        }
        error!("Invalid response message: {:?}", message);
        Ok(())
    }

    /// Convert ActiveMessage to DispatchMessage for the v2 dispatcher
    fn convert_to_dispatch_message(&self, message: ActiveMessage) -> DispatchMessage {
        use std::time::Instant;

        // Determine sender identity - for now, assume all senders are known
        // The existing auto-registration logic will handle unknown senders
        // TODO: Check if sender is actually registered and use Unknown if not
        let sender_identity = SenderIdentity::Known(message.sender_instance);

        // Convert metadata from serde_json::Value to bytes if present
        let metadata = if message.metadata != serde_json::Value::Null {
            match serde_json::to_vec(&message.metadata) {
                Ok(bytes) => Some(bytes::Bytes::from(bytes)),
                Err(e) => {
                    error!("Failed to serialize metadata: {}", e);
                    None
                }
            }
        } else {
            None
        };

        DispatchMessage {
            message_id: message.message_id,
            handler_name: message.handler_name,
            payload: message.payload,
            sender_identity,
            metadata,
            received_at: Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use uuid::Uuid;

    // Mock client for testing
    #[derive(Debug, Clone)]
    struct MockClient {
        instance_id: Uuid,
        endpoint: String,
    }

    #[async_trait::async_trait]
    impl ActiveMessageClient for MockClient {
        fn instance_id(&self) -> Uuid {
            self.instance_id
        }

        fn endpoint(&self) -> &str {
            &self.endpoint
        }

        fn peer_info(&self) -> crate::active_message::client::PeerInfo {
            crate::active_message::client::PeerInfo::new(self.instance_id, &self.endpoint)
        }

        async fn send_message(
            &self,
            _target: Uuid,
            _handler: &str,
            _payload: Bytes,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn broadcast_message(&self, _handler: &str, _payload: Bytes) -> anyhow::Result<()> {
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

        async fn disconnect_from_peer(&self, _instance_id: Uuid) -> anyhow::Result<()> {
            Ok(())
        }

        async fn await_handler(
            &self,
            _instance_id: Uuid,
            _handler: &str,
            _timeout: Option<std::time::Duration>,
        ) -> anyhow::Result<bool> {
            Ok(true)
        }

        async fn list_handlers(&self, _instance_id: Uuid) -> anyhow::Result<Vec<String>> {
            Ok(vec![])
        }

        async fn send_raw_message(
            &self,
            _target: Uuid,
            _message: crate::active_message::handler::ActiveMessage,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn register_acceptance(
            &self,
            _message_id: Uuid,
            _sender: tokio::sync::oneshot::Sender<()>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn register_response(
            &self,
            _message_id: Uuid,
            _sender: tokio::sync::oneshot::Sender<Bytes>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn register_ack(
            &self,
            _ack_id: Uuid,
            _timeout: std::time::Duration,
        ) -> anyhow::Result<tokio::sync::oneshot::Receiver<Result<(), String>>> {
            let (_tx, rx) = tokio::sync::oneshot::channel();
            Ok(rx)
        }

        async fn register_receipt(
            &self,
            _receipt_id: Uuid,
            _timeout: std::time::Duration,
        ) -> anyhow::Result<
            tokio::sync::oneshot::Receiver<
                Result<crate::active_message::receipt_ack::ReceiptAck, String>,
            >,
        > {
            let (_tx, rx) = tokio::sync::oneshot::channel();
            Ok(rx)
        }

        async fn has_incoming_connection_from(&self, _instance_id: Uuid) -> bool {
            false
        }

        fn clone_as_arc(&self) -> Arc<dyn ActiveMessageClient> {
            Arc::new(MockClient {
                instance_id: self.instance_id,
                endpoint: self.endpoint.clone(),
            })
        }
    }

    #[tokio::test]
    async fn test_message_router_creation() {
        use crate::active_message::response_manager::ResponseManager;

        let response_manager = Arc::new(ResponseManager::new());
        let client = Arc::new(MockClient {
            instance_id: Uuid::new_v4(),
            endpoint: "test://localhost".to_string(),
        });
        let (dispatch_tx, _dispatch_rx) = mpsc::channel(100);

        let router = MessageRouter::new(response_manager, client, dispatch_tx);

        // Just verify we can create a router - full testing would require more complex setup
        assert!(std::ptr::addr_of!(router) as usize != 0);
    }
}
