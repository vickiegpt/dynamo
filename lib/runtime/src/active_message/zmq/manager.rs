// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{Mutex, RwLock, Semaphore, broadcast, mpsc};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::active_message::{
    client::ActiveMessageClient,
    handler::{
        ActiveMessage, ActiveMessageContext, HandlerEvent, HandlerId, HandlerType, InstanceId,
    },
    manager::{ActiveMessageManager, HandlerConfig},
    response::SingleResponseSender,
};

use super::{
    builtin_handlers::{
        AckHandler, HealthCheckHandler, ListHandlersHandler, RegisterServiceHandler,
        WaitForHandlerHandler,
    },
    client::ZmqActiveMessageClient,
    discovery,
    transport::ZmqTransport,
};

use crate::utils::tasks::tracker::{
    LogOnlyPolicy, TaskTracker as DynamoTaskTracker, UnlimitedScheduler,
};

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

pub(crate) struct HandlerEntry {
    pub handler: HandlerType,
    pub task_tracker: DynamoTaskTracker,
    pub semaphore: Option<Arc<Semaphore>>,
}

pub struct ManagerState {
    pub instance_id: InstanceId,
    pub tcp_endpoint: Option<String>,
    pub ipc_endpoint: Option<String>,
    pub(crate) handlers: HashMap<HandlerId, HandlerEntry>,
    pub client: Arc<ZmqActiveMessageClient>,
    pub handler_events_tx: broadcast::Sender<HandlerEvent>,
}

impl std::fmt::Debug for ManagerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagerState")
            .field("instance_id", &self.instance_id)
            .field("tcp_endpoint", &self.tcp_endpoint)
            .field("ipc_endpoint", &self.ipc_endpoint)
            .field("handlers", &self.handlers.keys())
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
}

impl ZmqActiveMessageManager {
    pub fn client(&self) -> Arc<ZmqActiveMessageClient> {
        self.client.clone()
    }

    pub fn manager_state(&self) -> Arc<RwLock<ManagerState>> {
        self.state.clone()
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
        let tcp_transport = ZmqTransport::new_subscriber_bound(&context, &endpoint)?;
        let tcp_endpoint = tcp_transport
            .local_endpoint()
            .ok_or_else(|| anyhow::anyhow!("Failed to get bound TCP endpoint"))?
            .clone();

        // Create IPC transport for same-host optimization
        let ipc_endpoint_addr = discovery::create_ipc_endpoint(instance_id);
        let ipc_transport = ZmqTransport::new_subscriber_bound(&context, &ipc_endpoint_addr)?;
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

        // Use TCP endpoint for client advertisement (primary endpoint)
        let client = Arc::new(ZmqActiveMessageClient::new(
            instance_id,
            tcp_endpoint.clone(),
        ));

        let (handler_events_tx, _) = broadcast::channel(1024);

        let state = Arc::new(RwLock::new(ManagerState {
            instance_id,
            tcp_endpoint: Some(tcp_endpoint),
            ipc_endpoint: Some(ipc_endpoint),
            handlers: HashMap::new(),
            client: client.clone(),
            handler_events_tx: handler_events_tx.clone(),
        }));

        // Initialize concurrency controls
        let message_task_tracker = TaskTracker::new();

        let manager = Self {
            state: state.clone(),
            client: client.clone(),
            handler_events_tx: handler_events_tx.clone(),
            cancel_token: cancel_token.clone(),
            receiver_task: Arc::new(Mutex::new(None)),
            ack_cleanup_task: Arc::new(Mutex::new(None)),
            message_task_tracker,
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
            manager.message_task_tracker.clone(),
        ));

        let ack_cleanup_task =
            tokio::spawn(Self::ack_cleanup_loop(client.clone(), cancel_token.clone()));

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
        client: Arc<ZmqActiveMessageClient>,
        cancel_token: CancellationToken,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    client.cleanup_expired_acks().await;
                }
                _ = cancel_token.cancelled() => {
                    debug!("ACK cleanup loop cancelled");
                    break;
                }
            }
        }
    }

    async fn register_builtin_handlers(&self) -> Result<()> {
        let state = self.state.read().await;
        let client = state.client.clone();
        let handler_events_tx = state.handler_events_tx.clone();
        drop(state);

        // Register built-in handlers using the new base traits
        let ack_handler = HandlerType::no_return(AckHandler::new(client.clone()));
        self.register_handler_typed(ack_handler, None).await?;

        let register_service = HandlerType::response(RegisterServiceHandler::new(client.clone()));
        self.register_handler_typed(register_service, None).await?;

        let list_handlers = HandlerType::response(ListHandlersHandler::new(self.state.clone()));
        self.register_handler_typed(list_handlers, None).await?;

        let wait_for_handler = HandlerType::response(WaitForHandlerHandler::new(
            self.state.clone(),
            handler_events_tx,
        ));
        self.register_handler_typed(wait_for_handler, None).await?;

        let health_check = HandlerType::response(HealthCheckHandler);
        self.register_handler_typed(health_check, None).await?;

        Ok(())
    }

    async fn receive_loop(
        state: Arc<RwLock<ManagerState>>,
        mut message_rx: mpsc::UnboundedReceiver<ActiveMessage>,
        cancel_token: CancellationToken,
        message_task_tracker: TaskTracker,
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
                            // Spawn message handling immediately for parallelism
                            let state_clone = state.clone();

                            message_task_tracker.spawn(async move {
                                if let Err(e) = Self::handle_message(state_clone, message).await {
                                    error!("Failed to handle message: {}", e);
                                }
                            });
                        }
                        None => {
                            debug!("Message channel closed");
                            break;
                        }
                    }
                }
            }
        }

        // Wait for all spawned message handling tasks to complete
        info!("Waiting for message handling tasks to complete...");
        message_task_tracker.close();
        message_task_tracker.wait().await;
        info!("All message handling tasks completed");

        Ok(())
    }

    async fn handle_message(
        state: Arc<RwLock<ManagerState>>,
        message: ActiveMessage,
    ) -> Result<()> {
        // Handle internal response routing first
        if message.handler_name == "_accept" {
            return Self::handle_acceptance(state, message).await;
        }

        if message.handler_name == "_response" {
            return Self::handle_response(state, message).await;
        }

        // Track incoming connection from sender and auto-register if endpoint provided
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

        let state_read = state.read().await;
        let handler_name = message.handler_name.clone();

        let handler_entry = match state_read.handlers.get(&handler_name) {
            Some(entry) => entry,
            None => {
                warn!("No handler found for: {}", handler_name);
                return Ok(());
            }
        };

        let handler = handler_entry.handler.clone();
        let client = state_read.client.clone();
        let task_tracker = handler_entry.task_tracker.clone();
        let semaphore = handler_entry.semaphore.clone();

        drop(state_read);

        debug!("Dispatching message to handler: {}", handler_name);

        // Acquire per-handler semaphore permit if concurrency limit is set
        let _permit = if let Some(ref handler_semaphore) = semaphore {
            Some(
                handler_semaphore
                    .acquire()
                    .await
                    .expect("Handler semaphore closed"),
            )
        } else {
            None
        };

        // Extract acceptance and response configuration
        let acceptance_mode = message
            .metadata
            .get("_mode")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let accept_id = if matches!(acceptance_mode.as_deref(), Some("confirmed")) {
            message
                .metadata
                .get("_accept_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok())
        } else {
            None
        };
        let with_response_accept_id = if matches!(acceptance_mode.as_deref(), Some("with_response"))
        {
            message
                .metadata
                .get("_accept_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok())
        } else {
            None
        };

        // Set up response sender for Response handlers
        let response_sender = if matches!(acceptance_mode.as_deref(), Some("with_response")) {
            if let Some(response_id_str) = message
                .metadata
                .get("_response_id")
                .and_then(|v| v.as_str())
            {
                if let Ok(response_id) = Uuid::parse_str(response_id_str) {
                    Some(SingleResponseSender::new(
                        client.clone(),
                        message.sender_instance,
                        response_id,
                        message.handler_name.clone(),
                    ))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        task_tracker.spawn(async move {
            // Validate handler type matches client expectations first
            let is_handler_type_valid = match (&handler, acceptance_mode.as_deref()) {
                (HandlerType::NoReturn(_), Some("fire_and_forget")) => true,
                (HandlerType::Ack(_), Some("confirmed")) => true,
                (HandlerType::Response(_), Some("with_response")) => true,
                (HandlerType::NoReturn(_), None) => true, // No mode means fire_and_forget for backward compatibility
                _ => false,
            };

            if !is_handler_type_valid {
                let handler_type_name = match &handler {
                    HandlerType::NoReturn(_) => "NoReturn",
                    HandlerType::Ack(_) => "Ack",
                    HandlerType::Response(_) => "Response",
                };
                let expected_mode = acceptance_mode.as_deref().unwrap_or("fire_and_forget");
                let error_msg = format!(
                    "Handler type mismatch: client expects '{}' but handler is '{}'",
                    expected_mode, handler_type_name
                );

                // Send NACK for type mismatch if ACK was expected
                if let Some(accept_id) = accept_id {
                    if let Err(e) = client
                        .send_nack(message.sender_instance, accept_id, error_msg.clone())
                        .await
                    {
                        error!(
                            "Failed to send NACK for handler type mismatch {}: {}",
                            accept_id, e
                        );
                    } else {
                        debug!(
                            "Sent NACK for handler type mismatch {}: {}",
                            accept_id, error_msg
                        );
                    }
                }

                // Send NACK for type mismatch if response was expected
                if let Some(accept_id) = with_response_accept_id {
                    if let Err(e) = client
                        .send_nack(message.sender_instance, accept_id, error_msg.clone())
                        .await
                    {
                        error!(
                            "Failed to send NACK for handler type mismatch {}: {}",
                            accept_id, e
                        );
                    } else {
                        debug!(
                            "Sent NACK for handler type mismatch {}: {}",
                            accept_id, error_msg
                        );
                    }
                }

                error!("Handler type validation failed: {}", error_msg);
                return Ok(());
            }

            // Validate payload if ACK is expected
            // For AckHandler, delay ACK until after handler execution
            let should_send_early_ack =
                accept_id.is_some() && !matches!(handler, HandlerType::Ack(_));

            if should_send_early_ack {
                let accept_id = accept_id.unwrap();
                match handler.validate_schema(&message.payload) {
                    Ok(()) => {
                        // Payload validation succeeded - send ACK for non-AckHandler
                        if let Err(e) = client.send_ack(message.sender_instance, accept_id).await {
                            error!("Failed to send ACK for {}: {}", accept_id, e);
                        } else {
                            debug!("Sent ACK for message {}", accept_id);
                        }
                    }
                    Err(validation_error) => {
                        // Payload validation failed - send NACK
                        let error_msg = format!("Payload validation failed: {}", validation_error);
                        if let Err(e) = client
                            .send_nack(message.sender_instance, accept_id, error_msg.clone())
                            .await
                        {
                            error!("Failed to send NACK for {}: {}", accept_id, e);
                        } else {
                            debug!("Sent NACK for message {}: {}", accept_id, error_msg);
                        }
                        // Don't execute handler if validation failed
                        return Ok(());
                    }
                }
            } else if let Some(accept_id) = accept_id {
                // For AckHandler, just validate schema but don't send ACK yet
                if let Err(validation_error) = handler.validate_schema(&message.payload) {
                    // Payload validation failed - send NACK
                    let error_msg = format!("Payload validation failed: {}", validation_error);
                    if let Err(e) = client
                        .send_nack(message.sender_instance, accept_id, error_msg.clone())
                        .await
                    {
                        error!("Failed to send NACK for {}: {}", accept_id, e);
                    } else {
                        debug!("Sent NACK for message {}: {}", accept_id, error_msg);
                    }
                    // Don't execute handler if validation failed
                    return Ok(());
                }
            }

            // Handle acceptance for with_response mode
            if let Some(accept_id) = with_response_accept_id {
                // Send acceptance message
                let accept_message = ActiveMessage {
                    message_id: Uuid::new_v4(),
                    handler_name: "_accept".to_string(),
                    sender_instance: client.instance_id(),
                    payload: Bytes::new(),
                    metadata: serde_json::json!({
                        "_accept_for": accept_id.to_string()
                    }),
                };

                if let Err(e) = client
                    .send_raw_message(message.sender_instance, accept_message)
                    .await
                {
                    error!("Failed to send acceptance for {}: {}", accept_id, e);
                } else {
                    debug!("Sent acceptance for message {}", accept_id);
                }
            }

            // Extract sender_instance before moving message to handler
            let sender_instance = message.sender_instance;

            // Execute handler based on type
            match &handler {
                HandlerType::NoReturn(h) => {
                    // Create ActiveMessageContext from message
                    let ctx = super::super::handler::ActiveMessageContext::new(
                        message.message_id,
                        message.sender_instance,
                        message.handler_name.clone(),
                        message.metadata.clone(),
                        client.clone_as_arc(),
                        None, // No cancellation for now
                    );
                    h.handle(message.payload, ctx).await;
                }
                HandlerType::Ack(h) => {
                    // Create ActiveMessageContext from message
                    let ctx = super::super::handler::ActiveMessageContext::new(
                        message.message_id,
                        message.sender_instance,
                        message.handler_name.clone(),
                        message.metadata.clone(),
                        client.clone_as_arc(),
                        None, // No cancellation for now
                    );
                    // ACK handlers need to handle ACK/NACK based on Result
                    match h.handle(message.payload, ctx).await {
                        Ok(()) => {
                            // Success - send ACK if needed
                            debug!("Handler '{}' completed successfully", handler_name);
                            if let Some(accept_id) = accept_id {
                                if let Err(e) = client.send_ack(sender_instance, accept_id).await {
                                    error!("Failed to send ACK for {}: {}", accept_id, e);
                                } else {
                                    debug!("Sent ACK for message {}", accept_id);
                                }
                            }
                        }
                        Err(e) => {
                            // Error - send NACK if needed
                            error!("Handler '{}' failed: {}", handler_name, e);
                            if let Some(accept_id) = accept_id {
                                let error_msg = format!("Handler execution failed: {}", e);
                                if let Err(send_err) = client
                                    .send_nack(sender_instance, accept_id, error_msg.clone())
                                    .await
                                {
                                    error!("Failed to send NACK for {}: {}", accept_id, send_err);
                                } else {
                                    debug!(
                                        "Sent NACK for message {} due to handler error: {}",
                                        accept_id, error_msg
                                    );
                                }
                            }
                        }
                    }
                }
                HandlerType::Response(h) => {
                    if let Some(sender) = response_sender {
                        // Create ActiveMessageContext from message
                        let ctx = super::super::handler::ActiveMessageContext::new(
                            message.message_id,
                            message.sender_instance,
                            message.handler_name.clone(),
                            message.metadata.clone(),
                            client.clone_as_arc(),
                            None, // No cancellation for now
                        );
                        // Response handler returns Bytes
                        match h.handle(message.payload, ctx).await {
                            Ok(response_bytes) => {
                                if let Err(e) = sender.send_raw(response_bytes).await {
                                    error!("Failed to send response: {}", e);
                                }
                            }
                            Err(e) => {
                                if let Err(send_err) = sender.send_err(e).await {
                                    error!("Failed to send error response: {}", send_err);
                                }
                            }
                        }
                    } else {
                        error!(
                            "Response handler '{}' called without response sender",
                            handler_name
                        );
                    }
                }
            }
            Ok(())
        });

        Ok(())
    }

    async fn handle_acceptance(
        state: Arc<RwLock<ManagerState>>,
        message: ActiveMessage,
    ) -> Result<()> {
        if let Some(accept_for_str) = message.metadata.get("_accept_for").and_then(|v| v.as_str())
            && let Ok(accept_id) = Uuid::parse_str(accept_for_str)
        {
            let state_read = state.read().await;
            return state_read
                .client
                .complete_acceptance(accept_id, message.sender_instance)
                .await;
        }
        error!("Invalid acceptance message: {:?}", message);
        Ok(())
    }

    async fn handle_response(
        state: Arc<RwLock<ManagerState>>,
        message: ActiveMessage,
    ) -> Result<()> {
        if let Some(response_to_str) = message
            .metadata
            .get("_response_to")
            .and_then(|v| v.as_str())
            && let Ok(response_id) = Uuid::parse_str(response_to_str)
        {
            let state_read = state.read().await;
            return state_read
                .client
                .complete_response(response_id, message.sender_instance, message.payload)
                .await;
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

    /// Register a handler using the new handler type system
    async fn register_handler_typed(
        &self,
        handler: HandlerType,
        config: Option<HandlerConfig>,
    ) -> Result<()> {
        let handler_name = handler.name().to_string();

        let config = config.unwrap_or_default();
        let task_tracker = config.task_tracker.unwrap_or_else(|| {
            DynamoTaskTracker::new(UnlimitedScheduler::new(), LogOnlyPolicy::new())
                .expect("Failed to create default task tracker")
        });

        // Create per-handler semaphore if concurrency limit is set
        let semaphore = config
            .max_concurrent_messages
            .map(|max| Arc::new(Semaphore::new(max)));

        let mut state = self.state.write().await;

        if state.handlers.contains_key(&handler_name) {
            anyhow::bail!("Handler '{}' already registered", handler_name);
        }

        let entry = HandlerEntry {
            handler,
            task_tracker,
            semaphore,
        };

        state.handlers.insert(handler_name.clone(), entry);

        let event = HandlerEvent::Registered {
            name: handler_name.clone(),
            instance: state.instance_id,
        };

        let _ = state.handler_events_tx.send(event);

        debug!("Registered handler: {}", handler_name);

        Ok(())
    }

    async fn deregister_handler(&self, name: &str) -> Result<()> {
        let mut state = self.state.write().await;

        if state.handlers.remove(name).is_none() {
            anyhow::bail!("Handler '{}' not found", name);
        }

        let event = HandlerEvent::Deregistered {
            name: name.to_string(),
            instance: state.instance_id,
        };

        let _ = state.handler_events_tx.send(event);

        debug!("Deregistered handler: {}", name);

        Ok(())
    }

    async fn list_handlers(&self) -> Vec<HandlerId> {
        let state = self.state.read().await;
        state.handlers.keys().cloned().collect()
    }

    fn handler_events(&self) -> broadcast::Receiver<HandlerEvent> {
        self.handler_events_tx.subscribe()
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down ZmqActiveMessageManager");
        self.cancel_token.cancel();

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

        // Join handler task trackers
        let state = self.state.read().await;
        for (name, entry) in &state.handlers {
            debug!("Joining task tracker for handler: {}", name);
            entry.task_tracker.join().await;
        }

        Ok(())
    }
}

// Additional convenience methods for closure registration
impl ZmqActiveMessageManager {
    /// Register a closure that doesn't return a value (NoReturnHandler)
    ///
    /// This method works with raw `Bytes` and requires manual deserialization.
    ///
    /// # Example
    /// ```rust
    /// manager.register_no_return_closure("log", |input, ctx| async move {
    ///     println!("Received {} bytes from {}", input.len(), ctx.sender_instance());
    /// }).await?;
    /// ```
    pub async fn register_no_return_closure<F, Fut>(
        &self,
        name: impl Into<String>,
        closure: F,
    ) -> Result<()>
    where
        F: Fn(Bytes, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let handler = HandlerType::from_no_return_closure(name, closure);
        self.register_handler_typed(handler, None).await
    }

    /// Register a closure that returns Result<()> (AckHandler)
    ///
    /// This method works with raw `Bytes` and requires manual deserialization.
    ///
    /// # Example
    /// ```rust
    /// manager.register_ack_closure("validate", |input, _ctx| async move {
    ///     let data: ValidationRequest = serde_json::from_slice(&input)?;
    ///     validate_data(&data)?;
    ///     Ok(())
    /// }).await?;
    /// ```
    pub async fn register_ack_closure<F, Fut>(
        &self,
        name: impl Into<String>,
        closure: F,
    ) -> Result<()>
    where
        F: Fn(Bytes, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let handler = HandlerType::from_ack_closure(name, closure);
        self.register_handler_typed(handler, None).await
    }

    /// Register a closure that returns Result<Bytes> (ResponseHandler)
    ///
    /// This method works with raw `Bytes` and requires manual serialization/deserialization.
    ///
    /// # Example
    /// ```rust
    /// manager.register_response_closure("compute", |input, _ctx| async move {
    ///     let request: ComputeRequest = serde_json::from_slice(&input)?;
    ///     let result = compute_something(request);
    ///     let response = serde_json::to_vec(&result)?;
    ///     Ok(Bytes::from(response))
    /// }).await?;
    /// ```
    pub async fn register_response_closure<F, Fut>(
        &self,
        name: impl Into<String>,
        closure: F,
    ) -> Result<()>
    where
        F: Fn(Bytes, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Bytes>> + Send + 'static,
    {
        let handler = HandlerType::from_response_closure(name, closure);
        self.register_handler_typed(handler, None).await
    }

    // Typed closure registration methods for better ergonomics

    /// Register a typed unary (request/response) closure
    ///
    /// # Example
    /// ```rust
    /// manager.register_unary("compute", |req: ComputeRequest, _ctx| async move {
    ///     Ok(ComputeResponse { result: req.x + req.y })
    /// }).await?;
    /// ```
    pub async fn register_unary<Req, Res, F, Fut>(
        &self,
        name: impl Into<String>,
        closure: F,
    ) -> Result<()>
    where
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
        F: Fn(Req, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Res>> + Send + 'static,
    {
        let handler = HandlerType::from_unary_closure(name, closure);
        self.register_handler_typed(handler, None).await
    }

    /// Register a typed void (no return) closure
    ///
    /// # Example
    /// ```rust
    /// manager.register_void("log", |msg: String, _ctx| async move {
    ///     println!("Received: {}", msg);
    /// }).await?;
    /// ```
    pub async fn register_void<Input, F, Fut>(
        &self,
        name: impl Into<String>,
        closure: F,
    ) -> Result<()>
    where
        Input: serde::de::DeserializeOwned + Send + 'static,
        F: Fn(Input, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let handler = HandlerType::from_void_closure(name, closure);
        self.register_handler_typed(handler, None).await
    }

    /// Register a typed acknowledgment closure
    ///
    /// # Example
    /// ```rust
    /// manager.register_typed_ack("validate", |data: ValidationRequest, _ctx| async move {
    ///     if data.is_valid() {
    ///         Ok(())
    ///     } else {
    ///         Err(anyhow!("Invalid data"))
    ///     }
    /// }).await?;
    /// ```
    pub async fn register_typed_ack<Input, F, Fut>(
        &self,
        name: impl Into<String>,
        closure: F,
    ) -> Result<()>
    where
        Input: serde::de::DeserializeOwned + Send + 'static,
        F: Fn(Input, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let handler = HandlerType::from_typed_ack_closure(name, closure);
        self.register_handler_typed(handler, None).await
    }
}
