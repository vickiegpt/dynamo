// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::SinkExt;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};
use tmq::{Context, Message, Multipart, push};
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::active_message::{
    client::{ActiveMessageClient, Endpoint, PeerInfo},
    handler::{ActiveMessage, HandlerId, InstanceId},
    receipt_ack::ReceiptAck,
    response_manager::SharedResponseManager,
};

struct ClientState {
    peers: HashMap<InstanceId, PeerInfo>,
    incoming_peers: HashSet<InstanceId>, // Track who has connected TO us
    publisher_channels: HashMap<String, mpsc::UnboundedSender<ActiveMessage>>,
    publisher_tasks: HashMap<String, JoinHandle<()>>,
}

impl std::fmt::Debug for ClientState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientState")
            .field("peers", &self.peers)
            .field("incoming_peers", &self.incoming_peers)
            .field("publisher_channels_count", &self.publisher_channels.len())
            .field("publisher_tasks_count", &self.publisher_tasks.len())
            .finish()
    }
}

#[derive(Clone)]
pub struct ZmqActiveMessageClient {
    instance_id: InstanceId,
    endpoint: Endpoint,
    state: Arc<RwLock<ClientState>>,
    context: Arc<Context>,
    response_manager: SharedResponseManager,
}

impl std::fmt::Debug for ZmqActiveMessageClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZmqActiveMessageClient")
            .field("instance_id", &self.instance_id)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

impl ZmqActiveMessageClient {
    pub fn new(
        instance_id: InstanceId,
        endpoint: Endpoint,
        response_manager: SharedResponseManager,
    ) -> Self {
        let state = ClientState {
            peers: HashMap::new(),
            incoming_peers: HashSet::new(),
            publisher_channels: HashMap::new(),
            publisher_tasks: HashMap::new(),
        };

        Self {
            instance_id,
            endpoint,
            state: Arc::new(RwLock::new(state)),
            context: Arc::new(Context::new()),
            response_manager,
        }
    }

    pub async fn register_ack(
        &self,
        ack_id: Uuid,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<Result<(), String>>> {
        let (tx, rx) = oneshot::channel();

        self.response_manager.register_ack(ack_id, tx, timeout);

        debug!("Registered ACK expectation: {}", ack_id);

        Ok(rx)
    }

    pub async fn register_receipt(
        &self,
        receipt_id: Uuid,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<Result<ReceiptAck, String>>> {
        let (tx, rx) = oneshot::channel();

        self.response_manager
            .register_receipt(receipt_id, tx, timeout);

        debug!("Registered receipt expectation: {}", receipt_id);

        Ok(rx)
    }

    pub async fn complete_receipt(&self, receipt_ack: ReceiptAck) -> Result<()> {
        if self
            .response_manager
            .complete_receipt(receipt_ack.message_id, Ok(receipt_ack.clone()))
        {
            debug!("Completing receipt for message {}", receipt_ack.message_id);
        } else {
            warn!(
                "Received receipt ACK for unknown message {}",
                receipt_ack.message_id
            );
        }

        Ok(())
    }

    pub async fn send_ack(&self, target: InstanceId, ack_id: Uuid) -> Result<()> {
        let payload = serde_json::json!({
            "ack_id": ack_id.to_string(),
            "status": {
                "type": "Ack"
            }
        });

        let payload_bytes = Bytes::from(serde_json::to_vec(&payload)?);

        self.send_message(target, "_ack", payload_bytes).await
    }

    pub async fn send_nack(&self, target: InstanceId, ack_id: Uuid, error: String) -> Result<()> {
        let payload = serde_json::json!({
            "ack_id": ack_id.to_string(),
            "status": {
                "type": "Nack",
                "data": {
                    "error": error
                }
            }
        });

        let payload_bytes = Bytes::from(serde_json::to_vec(&payload)?);

        self.send_message(target, "_ack", payload_bytes).await
    }

    pub(super) async fn complete_ack(&self, ack_id: Uuid, sender: InstanceId) -> Result<()> {
        if self.response_manager.complete_ack(ack_id, Ok(())) {
            debug!("Completed ACK: {} from {}", ack_id, sender);
            Ok(())
        } else {
            error!("Received unexpected ACK: {} from {}", ack_id, sender);
            anyhow::bail!("Unexpected ACK: {} from {}", ack_id, sender)
        }
    }

    pub(super) async fn complete_nack(
        &self,
        ack_id: Uuid,
        sender: InstanceId,
        error: String,
    ) -> Result<()> {
        if self
            .response_manager
            .complete_ack(ack_id, Err(error.clone()))
        {
            debug!(
                "Completed NACK: {} from {} with error: {}",
                ack_id, sender, error
            );
            Ok(())
        } else {
            error!("Received unexpected NACK: {} from {}", ack_id, sender);
            anyhow::bail!("Unexpected NACK: {} from {}", ack_id, sender)
        }
    }

    pub async fn cleanup_expired_acks(&self) {
        let cleaned = self.response_manager.cleanup_expired();
        if cleaned > 0 {
            debug!("Cleaned up {} expired entries", cleaned);
        }
    }

    pub async fn has_incoming_connection_from(&self, instance_id: InstanceId) -> bool {
        let state = self.state.read().await;
        state.incoming_peers.contains(&instance_id)
    }

    pub(super) async fn track_incoming_and_auto_register(
        &self,
        sender_instance: InstanceId,
        sender_endpoint: Option<&str>,
    ) {
        let mut state = self.state.write().await;

        // Track incoming connection
        state.incoming_peers.insert(sender_instance);

        // Auto-register sender if endpoint is provided and we don't already know them
        if let Some(endpoint) = sender_endpoint
            && let std::collections::hash_map::Entry::Vacant(e) = state.peers.entry(sender_instance)
        {
            let peer_info =
                crate::active_message::client::PeerInfo::new(sender_instance, endpoint.to_string());
            e.insert(peer_info);
            debug!(
                "Auto-registered peer {} at {} for response delivery",
                sender_instance, endpoint
            );
        }
    }

    /// Dedicated publisher task that owns a ZMQ socket for a specific endpoint
    async fn publisher_task(
        endpoint: String,
        context: Arc<Context>,
        mut receiver: mpsc::UnboundedReceiver<ActiveMessage>,
    ) {
        debug!("Starting push task for endpoint: {}", endpoint);

        // Connect socket
        let mut push_socket = match push(&context).connect(&endpoint) {
            Ok(socket) => socket,
            Err(e) => {
                error!("Failed to connect push socket to {}: {}", endpoint, e);
                return;
            }
        };

        debug!("Push task for {} ready to process messages", endpoint);

        // Process messages
        while let Some(message) = receiver.recv().await {
            match super::transport::ZmqTransport::serialize_message(&message) {
                Ok(multipart) => {
                    if let Err(e) = push_socket.send(multipart).await {
                        error!("Failed to send message via push to {}: {}", endpoint, e);
                    } else {
                        debug!("Sent message via push to {}", endpoint);
                    }
                }
                Err(e) => {
                    error!("Failed to serialize message for {}: {}", endpoint, e);
                }
            }
        }

        debug!("Push task for {} shutting down", endpoint);
    }

    async fn send_raw(&self, endpoint: &str, message: &ActiveMessage) -> Result<()> {
        let mut state = self.state.write().await;

        // Get or create publisher channel
        let sender = match state.publisher_channels.get(endpoint) {
            Some(sender) => sender.clone(),
            None => {
                // Create new channel and spawn publisher task
                let (tx, rx) = mpsc::unbounded_channel();
                let context = self.context.clone();
                let endpoint_owned = endpoint.to_string();

                let task = tokio::spawn(Self::publisher_task(endpoint_owned.clone(), context, rx));

                state
                    .publisher_channels
                    .insert(endpoint_owned.clone(), tx.clone());
                state.publisher_tasks.insert(endpoint_owned, task);
                debug!("Created new publisher task for endpoint: {}", endpoint);
                tx
            }
        };
        drop(state);

        // Send message through channel (non-blocking)
        sender
            .send(message.clone())
            .map_err(|_| anyhow::anyhow!("Publisher channel closed for endpoint: {}", endpoint))?;

        Ok(())
    }

    /// Shutdown all publisher tasks and cleanup resources
    pub async fn shutdown_publishers(&self) {
        let mut state = self.state.write().await;

        // Close all channels first to signal tasks to shutdown
        state.publisher_channels.clear();

        // Wait for all tasks to complete
        for (endpoint, task) in state.publisher_tasks.drain() {
            debug!("Shutting down publisher task for endpoint: {}", endpoint);
            task.abort();
            let _ = task.await;
        }

        debug!("All publisher tasks shut down");
    }
}

#[async_trait]
impl ActiveMessageClient for ZmqActiveMessageClient {
    fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn peer_info(&self) -> PeerInfo {
        PeerInfo::new(self.instance_id, &self.endpoint)
    }

    async fn send_message(&self, target: InstanceId, handler: &str, payload: Bytes) -> Result<()> {
        let state = self.state.read().await;

        let peer = state
            .peers
            .get(&target)
            .ok_or_else(|| anyhow::anyhow!("Unknown peer: {}", target))?;

        let message = ActiveMessage::new(
            Uuid::new_v4(),
            handler.to_string(),
            self.instance_id,
            payload,
            serde_json::Value::Null,
        );

        // Use smart endpoint selection to prefer IPC for same-host communication
        let endpoint = peer
            .select_endpoint(&self.endpoint)
            .ok_or_else(|| anyhow::anyhow!("No available endpoint for peer: {}", target))?
            .clone();
        drop(state);

        debug!(
            "Sending message to {} via {} endpoint: {}",
            target,
            if endpoint.starts_with("ipc://") {
                "IPC"
            } else {
                "TCP"
            },
            endpoint
        );

        self.send_raw(&endpoint, &message).await?;

        debug!(
            "Sent message to handler '{}' on peer {} at {}",
            handler, target, endpoint
        );

        Ok(())
    }

    async fn broadcast_message(&self, handler: &str, payload: Bytes) -> Result<()> {
        let state = self.state.read().await;
        let peers: Vec<PeerInfo> = state.peers.values().cloned().collect();
        drop(state);

        for peer in peers {
            if let Err(e) = self
                .send_message(peer.instance_id, handler, payload.clone())
                .await
            {
                warn!("Failed to send message to peer {}: {}", peer.instance_id, e);
            }
        }

        Ok(())
    }

    async fn list_peers(&self) -> Result<Vec<PeerInfo>> {
        let state = self.state.read().await;
        Ok(state.peers.values().cloned().collect())
    }

    async fn connect_to_peer(&self, peer: PeerInfo) -> Result<()> {
        let mut state = self.state.write().await;

        if state.peers.contains_key(&peer.instance_id) {
            debug!("Peer {} already connected", peer.instance_id);
            return Ok(());
        }

        state.peers.insert(peer.instance_id, peer.clone());

        debug!(
            "Added peer {} at {} to registry",
            peer.instance_id, peer.endpoint
        );

        Ok(())
    }

    async fn disconnect_from_peer(&self, instance_id: InstanceId) -> Result<()> {
        let mut state = self.state.write().await;
        state.peers.remove(&instance_id);

        debug!("Disconnected from peer {}", instance_id);

        Ok(())
    }

    async fn await_handler(
        &self,
        instance_id: InstanceId,
        handler: &str,
        timeout: Option<Duration>,
    ) -> Result<bool> {
        let payload = serde_json::json!({
            "handler_name": handler,
            "timeout_ms": timeout.map(|d| d.as_millis()),
        });

        let mut builder = self
            .system_active_message("_wait_for_handler")
            .payload(payload)?
            .expect_response::<crate::active_message::responses::WaitForHandlerResponse>();

        if let Some(t) = timeout {
            builder = builder.timeout(t);
        }

        let status = builder.send(instance_id).await?;
        let response: crate::active_message::responses::WaitForHandlerResponse =
            status.await_response().await?;
        Ok(response.available)
    }

    async fn list_handlers(&self, instance_id: InstanceId) -> Result<Vec<HandlerId>> {
        let status = self
            .system_active_message("_list_handlers")
            .expect_response::<crate::active_message::responses::ListHandlersResponse>()
            .send(instance_id)
            .await?;

        let response: crate::active_message::responses::ListHandlersResponse =
            status.await_response().await?;
        Ok(response.handlers)
    }

    async fn send_raw_message(&self, target: InstanceId, message: ActiveMessage) -> Result<()> {
        let state = self.state.read().await;
        let peer = state
            .peers
            .get(&target)
            .ok_or_else(|| anyhow::anyhow!("Peer {} not found", target))?;

        // Use smart endpoint selection to prefer IPC for same-host communication
        let endpoint = peer
            .select_endpoint(&self.endpoint)
            .ok_or_else(|| anyhow::anyhow!("No available endpoint for peer: {}", target))?
            .clone();
        drop(state);

        self.send_raw(&endpoint, &message).await?;

        debug!(
            "Sent raw message to handler '{}' on peer {} at {}",
            message.handler_name, target, endpoint
        );

        Ok(())
    }

    async fn register_acceptance(
        &self,
        message_id: Uuid,
        sender: oneshot::Sender<()>,
    ) -> Result<()> {
        self.response_manager
            .register_acceptance(message_id, sender, Duration::from_secs(30));

        debug!("Registered acceptance for message {}", message_id);
        Ok(())
    }

    async fn register_response(
        &self,
        message_id: Uuid,
        sender: oneshot::Sender<Bytes>,
    ) -> Result<()> {
        self.response_manager
            .register_response(message_id, sender, Duration::from_secs(30));

        debug!("Registered response for message {}", message_id);
        Ok(())
    }

    async fn register_ack(
        &self,
        ack_id: Uuid,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<Result<(), String>>> {
        self.register_ack(ack_id, timeout).await
    }

    async fn register_receipt(
        &self,
        receipt_id: Uuid,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<Result<ReceiptAck, String>>> {
        self.register_receipt(receipt_id, timeout).await
    }

    async fn has_incoming_connection_from(&self, instance_id: InstanceId) -> bool {
        self.has_incoming_connection_from(instance_id).await
    }

    fn clone_as_arc(&self) -> std::sync::Arc<dyn ActiveMessageClient> {
        Arc::new(self.clone())
    }
}

impl ZmqActiveMessageClient {
    pub(super) async fn complete_acceptance(
        &self,
        accept_id: Uuid,
        sender: InstanceId,
    ) -> Result<()> {
        if self.response_manager.complete_acceptance(accept_id) {
            debug!("Completed acceptance: {} from {}", accept_id, sender);
            Ok(())
        } else {
            error!(
                "Received unexpected acceptance: {} from {}",
                accept_id, sender
            );
            anyhow::bail!("Unexpected acceptance: {} from {}", accept_id, sender)
        }
    }

    pub(super) async fn complete_response(
        &self,
        response_id: Uuid,
        sender: InstanceId,
        payload: Bytes,
    ) -> Result<()> {
        if self
            .response_manager
            .complete_response(response_id, payload)
        {
            debug!("Completed response: {} from {}", response_id, sender);
            Ok(())
        } else {
            error!(
                "Received unexpected response: {} from {}",
                response_id, sender
            );
            anyhow::bail!("Unexpected response: {} from {}", response_id, sender)
        }
    }

    pub async fn cleanup_expired_notifications(&self) {
        let cleaned = self.response_manager.cleanup_expired();
        if cleaned > 0 {
            debug!("Cleaned up {} expired entries", cleaned);
        }
    }
}
