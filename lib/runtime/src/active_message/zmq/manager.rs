// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};
use tokio::sync::{Mutex, RwLock, broadcast, mpsc};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::active_message::{
    client::ActiveMessageClient,
    dispatcher::{ControlMessage, DispatchMessage, MessageDispatcher, SenderIdentity},
    handler::{ActiveMessage, HandlerEvent, HandlerId, InstanceId},
    manager::ActiveMessageManager,
    response_manager::{ResponseManager, SharedResponseManager},
};

use super::{client::ZmqActiveMessageClient, discovery, transport::ZmqTransport};

/// Builder for ZmqActiveMessageManager with configurable options
#[derive(Debug, Clone)]
pub struct ZmqActiveMessageManagerBuilder {}

impl Default for ZmqActiveMessageManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ZmqActiveMessageManagerBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {}
    }

    /// Build the ZmqActiveMessageManager with the configured options
    pub async fn build(
        self,
        endpoint: String,
        cancel_token: CancellationToken,
    ) -> Result<ZmqActiveMessageManager> {
        ZmqActiveMessageManager::new_with_builder(endpoint, cancel_token, self).await
    }
}

pub struct ManagerState {
    pub instance_id: InstanceId,
    pub tcp_endpoint: Option<String>,
    pub ipc_endpoint: Option<String>,
    pub client: Arc<ZmqActiveMessageClient>,
    pub handler_events_tx: broadcast::Sender<HandlerEvent>,
}

impl std::fmt::Debug for ManagerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagerState")
            .field("instance_id", &self.instance_id)
            .field("tcp_endpoint", &self.tcp_endpoint)
            .field("ipc_endpoint", &self.ipc_endpoint)
            .field("client", &self.client)
            .finish()
    }
}

pub struct ZmqActiveMessageManager {
    state: Arc<RwLock<ManagerState>>,
    client: Arc<ZmqActiveMessageClient>,
    handler_events_tx: broadcast::Sender<HandlerEvent>,
    cancel_token: CancellationToken,
    receiver_task: Arc<Mutex<Option<tokio::task::JoinHandle<Result<()>>>>>,
    ack_cleanup_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    message_task_tracker: TaskTracker,
    response_manager: SharedResponseManager,

    // v2 Dispatcher integration
    dispatch_tx: mpsc::Sender<DispatchMessage>,
    control_tx: mpsc::Sender<ControlMessage>,
    dispatcher_task: Arc<Mutex<Option<tokio::task::JoinHandle<Result<()>>>>>,
}

impl ZmqActiveMessageManager {
    pub fn client(&self) -> Arc<ZmqActiveMessageClient> {
        self.client.clone()
    }

    pub fn manager_state(&self) -> Arc<RwLock<ManagerState>> {
        self.state.clone()
    }

    /// Register a handler with the message dispatcher
    pub async fn register_handler(
        &self,
        name: String,
        handler: Arc<dyn crate::active_message::dispatcher::ActiveMessageDispatcher>,
    ) -> Result<()> {
        use crate::active_message::dispatcher::ControlMessage;

        self.control_tx
            .send(ControlMessage::Register {
                name,
                dispatcher: handler,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to register handler: {}", e))
    }

    /// Get PeerInfo representing this manager with dual endpoints
    pub async fn peer_info(&self) -> crate::active_message::client::PeerInfo {
        let state = self.state.read().await;
        crate::active_message::client::PeerInfo::new_dual(
            state.instance_id,
            state.tcp_endpoint.clone(),
            state.ipc_endpoint.clone(),
        )
    }

    /// Create a builder for configuring ZmqActiveMessageManager options
    pub fn builder() -> ZmqActiveMessageManagerBuilder {
        ZmqActiveMessageManagerBuilder::new()
    }

    pub async fn new(endpoint: String, cancel_token: CancellationToken) -> Result<Self> {
        // Use the builder with default settings for backward compatibility
        Self::new_with_builder(
            endpoint,
            cancel_token,
            ZmqActiveMessageManagerBuilder::new(),
        )
        .await
    }

    pub(crate) async fn new_with_builder(
        endpoint: String,
        cancel_token: CancellationToken,
        _builder: ZmqActiveMessageManagerBuilder,
    ) -> Result<Self> {
        let instance_id = InstanceId::new_v4();

        let context = tmq::Context::new();

        // Create TCP transport from provided endpoint
        let tcp_transport = ZmqTransport::new_puller_bound(&context, &endpoint)?;
        let tcp_endpoint = tcp_transport
            .local_endpoint()
            .ok_or_else(|| anyhow::anyhow!("Failed to get bound TCP endpoint"))?
            .clone();

        // Create IPC transport for same-host optimization
        let ipc_endpoint_addr = discovery::create_ipc_endpoint(instance_id);
        let ipc_transport = ZmqTransport::new_puller_bound(&context, &ipc_endpoint_addr)?;
        let ipc_endpoint = ipc_transport
            .local_endpoint()
            .ok_or_else(|| anyhow::anyhow!("Failed to get bound IPC endpoint"))?
            .clone();

        info!(
            "ZmqActiveMessageManager bound to TCP: {} and IPC: {} (instance: {})",
            tcp_endpoint, ipc_endpoint, instance_id
        );

        // Create MPSC channel for merging messages from both transports
        let (message_tx, message_rx) = mpsc::unbounded_channel();

        // Create shared response manager for concurrent access from both client and manager
        let response_manager = Arc::new(ResponseManager::new());

        // Use TCP endpoint for client advertisement (primary endpoint)
        let client = Arc::new(ZmqActiveMessageClient::new(
            instance_id,
            tcp_endpoint.clone(),
            response_manager.clone(),
        ));

        let (handler_events_tx, _) = broadcast::channel(1024);

        let state = Arc::new(RwLock::new(ManagerState {
            instance_id,
            tcp_endpoint: Some(tcp_endpoint),
            ipc_endpoint: Some(ipc_endpoint),
            client: client.clone(),
            handler_events_tx: handler_events_tx.clone(),
        }));

        // Initialize concurrency controls
        let message_task_tracker = TaskTracker::new();

        // Set up MessageDispatcher channels
        let (dispatch_tx, dispatch_rx) = mpsc::channel(1000); // Buffered channel for dispatch
        let (control_tx, control_rx) = mpsc::channel(100); // Control channel for handler registration

        // Create MessageDispatcher
        let dispatcher = MessageDispatcher::new(
            client.clone() as Arc<dyn ActiveMessageClient>,
            dispatch_rx,
            control_rx,
            message_task_tracker.clone(),
        );

        // Spawn dispatcher task
        let dispatcher_task = tokio::spawn(async move { dispatcher.run().await });

        let manager = Self {
            state: state.clone(),
            client: client.clone(),
            handler_events_tx: handler_events_tx.clone(),
            cancel_token: cancel_token.clone(),
            receiver_task: Arc::new(Mutex::new(None)),
            ack_cleanup_task: Arc::new(Mutex::new(None)),
            message_task_tracker,
            response_manager: response_manager.clone(),
            dispatch_tx,
            control_tx,
            dispatcher_task: Arc::new(Mutex::new(Some(dispatcher_task))),
        };

        manager.register_builtin_handlers().await?;

        // Spawn transport reader tasks for both TCP and IPC
        let _tcp_reader_task = tokio::spawn(Self::transport_reader_task(
            tcp_transport,
            message_tx.clone(),
            cancel_token.clone(),
            "TCP".to_string(),
        ));

        let _ipc_reader_task = tokio::spawn(Self::transport_reader_task(
            ipc_transport,
            message_tx,
            cancel_token.clone(),
            "IPC".to_string(),
        ));

        // Spawn main receive loop that reads from merged channel
        let receiver_task = tokio::spawn(Self::receive_loop(
            state.clone(),
            message_rx,
            cancel_token.clone(),
            manager.dispatch_tx.clone(),
            response_manager.clone(),
        ));

        let ack_cleanup_task =
            tokio::spawn(Self::ack_cleanup_loop(response_manager.clone(), cancel_token.clone()));

        // Store the task handles
        *manager.receiver_task.lock().await = Some(receiver_task);
        *manager.ack_cleanup_task.lock().await = Some(ack_cleanup_task);

        // Note: We're not storing tcp_reader_task and ipc_reader_task separately
        // They will be cancelled when cancel_token is triggered

        Ok(manager)
    }

    async fn transport_reader_task(
        mut transport: ZmqTransport,
        message_tx: mpsc::UnboundedSender<ActiveMessage>,
        cancel_token: CancellationToken,
        transport_name: String,
    ) {
        debug!("Starting {} transport reader task", transport_name);

        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    debug!("{} transport reader task cancelled", transport_name);
                    break;
                }

                result = transport.receive() => {
                    match result {
                        Ok(message) => {
                            debug!("Received message on {} transport", transport_name);
                            if message_tx.send(message).is_err() {
                                debug!("{} transport reader: message channel closed", transport_name);
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Failed to receive message on {} transport: {}", transport_name, e);
                            break;
                        }
                    }
                }
            }
        }

        debug!("{} transport reader task finished", transport_name);
    }

    async fn ack_cleanup_loop(
        response_manager: SharedResponseManager,
        cancel_token: CancellationToken,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let cleaned = response_manager.cleanup_expired();
                    if cleaned > 0 {
                        debug!("Cleaned up {} expired entries", cleaned);
                    }
                }
                _ = cancel_token.cancelled() => {
                    debug!("ACK cleanup loop cancelled");
                    break;
                }
            }
        }
    }

    async fn register_builtin_handlers(&self) -> Result<()> {
        // Register system handlers using the MessageDispatcher
        use crate::active_message::system_handlers::register_system_handlers;

        let state = self.state.read().await;
        let client = state.client.clone();
        drop(state);

        // Register system handlers with the dispatcher
        register_system_handlers(
            &self.control_tx,
            client.clone() as Arc<dyn ActiveMessageClient>,
            self.message_task_tracker.clone(),
        )
        .await?;

        Ok(())
    }

    async fn receive_loop(
        state: Arc<RwLock<ManagerState>>,
        mut message_rx: mpsc::UnboundedReceiver<ActiveMessage>,
        cancel_token: CancellationToken,
        dispatch_tx: mpsc::Sender<DispatchMessage>,
        response_manager: SharedResponseManager,
    ) -> Result<()> {
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    info!("Receive loop cancelled");
                    break;
                }

                message = message_rx.recv() => {
                    match message {
                        Some(message) => {
                            // Handle special internal messages that bypass the dispatcher
                            if message.handler_name == "_accept" {
                                if let Err(e) = Self::handle_acceptance(state.clone(), message, response_manager.clone()).await {
                                    error!("Failed to handle acceptance: {}", e);
                                }
                                continue;
                            }

                            // Check if this is a response message based on metadata (v2 pattern)
                            // v2 handlers send responses with original handler name but include _response_to metadata
                            if let Some(response_to) = message.metadata.get("_response_to").and_then(|v| v.as_str()) {
                                let response_to = response_to.to_string();  // Clone before moving message
                                debug!("Received response message for request {}", response_to);
                                if let Err(e) = Self::handle_response_unified(state.clone(), &response_to, message, response_manager.clone()).await {
                                    error!("Failed to handle response: {}", e);
                                }
                                continue;
                            }

                            // Legacy v1 response pattern (uses "_response" handler name)
                            if message.handler_name == "_response" {
                                if let Err(e) = Self::handle_response(state.clone(), message, response_manager.clone()).await {
                                    error!("Failed to handle response: {}", e);
                                }
                                continue;
                            }

                            // Auto-register sender before processing the message
                            // This ensures the peer is registered when handlers try to send responses
                            {
                                let state_read = state.read().await;
                                let sender_endpoint = message
                                    .metadata
                                    .get("_sender_endpoint")
                                    .and_then(|v| v.as_str());
                                state_read
                                    .client
                                    .track_incoming_and_auto_register(message.sender_instance, sender_endpoint)
                                    .await;
                            }

                            // Convert ActiveMessage to DispatchMessage
                            let dispatch_message = Self::convert_to_dispatch_message(message, &state).await;

                            // Forward to MessageDispatcher
                            if let Err(e) = dispatch_tx.send(dispatch_message).await {
                                error!("Failed to send message to dispatcher: {}", e);
                                break;
                            }
                        }
                        None => {
                            debug!("Message channel closed");
                            break;
                        }
                    }
                }
            }
        }

        info!("Receive loop finished");

        Ok(())
    }

    /// Convert ActiveMessage to DispatchMessage for the v2 dispatcher
    async fn convert_to_dispatch_message(
        message: ActiveMessage,
        _state: &Arc<RwLock<ManagerState>>,
    ) -> DispatchMessage {
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

    async fn handle_acceptance(
        _state: Arc<RwLock<ManagerState>>,
        message: ActiveMessage,
        response_manager: SharedResponseManager,
    ) -> Result<()> {
        if let Some(accept_for_str) = message.metadata.get("_accept_for").and_then(|v| v.as_str())
            && let Ok(accept_id) = Uuid::parse_str(accept_for_str)
        {
            response_manager.complete_acceptance(accept_id);
            return Ok(());
        }
        error!("Invalid acceptance message: {:?}", message);
        Ok(())
    }

    /// Handle unified response messages (with metadata-based routing)
    async fn handle_response_unified(
        _state: Arc<RwLock<ManagerState>>,
        response_to: &str,
        message: ActiveMessage,
        response_manager: SharedResponseManager,
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
                        response_manager.complete_ack(response_id, Ok(()));
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
                        response_manager.complete_ack(response_id, Err(error_msg));
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
        response_manager.complete_response(response_id, message.payload);
        Ok(())
    }

    /// Handle legacy v1 response messages
    async fn handle_response(
        _state: Arc<RwLock<ManagerState>>,
        message: ActiveMessage,
        response_manager: SharedResponseManager,
    ) -> Result<()> {
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
                            response_manager.complete_ack(response_id, Ok(()));
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
                            response_manager.complete_ack(response_id, Err(error_msg));
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
            response_manager.complete_response(response_id, message.payload);
            return Ok(());
        }
        error!("Invalid response message: {:?}", message);
        Ok(())
    }
}

#[async_trait]
impl ActiveMessageManager for ZmqActiveMessageManager {
    fn client(&self) -> Arc<dyn ActiveMessageClient> {
        self.client.clone()
    }

    async fn deregister_handler(&self, name: &str) -> Result<()> {
        // v2 handlers are managed by MessageDispatcher
        // Send deregistration through control channel
        let control_msg = ControlMessage::Unregister {
            name: name.to_string(),
        };

        if let Err(e) = self.control_tx.send(control_msg).await {
            anyhow::bail!(
                "Failed to deregister handler '{}' with dispatcher: {}",
                name,
                e
            );
        }

        let state = self.state.read().await;
        let event = HandlerEvent::Deregistered {
            name: name.to_string(),
            instance: state.instance_id,
        };
        let _ = state.handler_events_tx.send(event);

        debug!("Deregistered handler: {}", name);
        Ok(())
    }

    async fn list_handlers(&self) -> Vec<HandlerId> {
        // v2 handlers are managed by MessageDispatcher
        // For now, return empty list since we don't have a way to query the dispatcher
        // This could be enhanced later if needed
        Vec::new()
    }

    fn handler_events(&self) -> broadcast::Receiver<HandlerEvent> {
        self.handler_events_tx.subscribe()
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down ZmqActiveMessageManager");
        self.cancel_token.cancel();

        // Send shutdown signal to the dispatcher
        debug!("Sending shutdown to dispatcher");
        if let Err(e) = self.control_tx.send(ControlMessage::Shutdown).await {
            warn!("Failed to send shutdown to dispatcher: {}", e);
        }

        // Join background tasks first
        if let Some(receiver_task) = self.receiver_task.lock().await.take() {
            debug!("Joining receiver task");
            if let Err(e) = receiver_task.await {
                warn!("Receiver task join error: {:?}", e);
            }
        }

        if let Some(ack_cleanup_task) = self.ack_cleanup_task.lock().await.take() {
            debug!("Joining ACK cleanup task");
            if let Err(e) = ack_cleanup_task.await {
                warn!("ACK cleanup task join error: {:?}", e);
            }
        }

        // Wait for dispatcher to shut down
        if let Some(dispatcher_task) = self.dispatcher_task.lock().await.take() {
            debug!("Joining dispatcher task");
            if let Err(e) = dispatcher_task.await {
                warn!("Dispatcher task join error: {:?}", e);
            }
        }

        // v2 handlers are managed by MessageDispatcher
        // The message_task_tracker will handle cleanup of v2 handler tasks
        debug!("Waiting for all message handlers to complete");
        self.message_task_tracker.close();
        self.message_task_tracker.wait().await;

        info!("ZmqActiveMessageManager shutdown complete");
        Ok(())
    }
}
