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
use tmq::{Context, Message, Multipart, publish};
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::active_message::{
    client::{ActiveMessageClient, Endpoint, PeerInfo},
    handler::{ActiveMessage, HandlerId, InstanceId},
    zmq::ZmqTransport,
};

struct AckEntry {
    sender: oneshot::Sender<Result<(), String>>,
    deadline: tokio::time::Instant,
}

impl std::fmt::Debug for AckEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AckEntry")
            .field("deadline", &self.deadline)
            .finish()
    }
}

struct AcceptanceEntry {
    sender: oneshot::Sender<()>,
    deadline: tokio::time::Instant,
}

impl std::fmt::Debug for AcceptanceEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcceptanceEntry")
            .field("deadline", &self.deadline)
            .finish()
    }
}

struct ResponseEntry {
    sender: oneshot::Sender<Bytes>,
    deadline: tokio::time::Instant,
}

impl std::fmt::Debug for ResponseEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseEntry")
            .field("deadline", &self.deadline)
            .finish()
    }
}

struct ClientState {
    peers: HashMap<InstanceId, PeerInfo>,
    incoming_peers: HashSet<InstanceId>, // Track who has connected TO us
    pending_acks: HashMap<Uuid, AckEntry>,
    pending_acceptances: HashMap<Uuid, AcceptanceEntry>,
    pending_responses: HashMap<Uuid, ResponseEntry>,
    publisher_channels: HashMap<String, mpsc::UnboundedSender<ActiveMessage>>,
    publisher_tasks: HashMap<String, JoinHandle<()>>,
}

impl std::fmt::Debug for ClientState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientState")
            .field("peers", &self.peers)
            .field("incoming_peers", &self.incoming_peers)
            .field("pending_acks", &self.pending_acks)
            .field("pending_acceptances", &self.pending_acceptances)
            .field("pending_responses", &self.pending_responses)
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
    pub fn new(instance_id: InstanceId, endpoint: Endpoint) -> Self {
        let state = ClientState {
            peers: HashMap::new(),
            incoming_peers: HashSet::new(),
            pending_acks: HashMap::new(),
            pending_acceptances: HashMap::new(),
            pending_responses: HashMap::new(),
            publisher_channels: HashMap::new(),
            publisher_tasks: HashMap::new(),
        };

        Self {
            instance_id,
            endpoint,
            state: Arc::new(RwLock::new(state)),
            context: Arc::new(Context::new()),
        }
    }

    pub async fn register_ack(
        &self,
        ack_id: Uuid,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<Result<(), String>>> {
        let (tx, rx) = oneshot::channel();
        let deadline = tokio::time::Instant::now() + timeout;

        let mut state = self.state.write().await;
        if state.pending_acks.contains_key(&ack_id) {
            anyhow::bail!("ACK {} already registered", ack_id);
        }

        state.pending_acks.insert(
            ack_id,
            AckEntry {
                sender: tx,
                deadline,
            },
        );

        debug!("Registered ACK expectation: {}", ack_id);

        Ok(rx)
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
        let mut state = self.state.write().await;

        if let Some(entry) = state.pending_acks.remove(&ack_id) {
            let _ = entry.sender.send(Ok(()));
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
        let mut state = self.state.write().await;

        if let Some(entry) = state.pending_acks.remove(&ack_id) {
            let _ = entry.sender.send(Err(error.clone()));
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
        let mut state = self.state.write().await;
        let now = tokio::time::Instant::now();

        state.pending_acks.retain(|ack_id, entry| {
            if now >= entry.deadline {
                warn!("ACK {} expired (not received in time)", ack_id);
                false
            } else {
                true
            }
        });
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
        debug!("Starting publisher task for endpoint: {}", endpoint);

        // Connect socket
        let mut pub_socket = match publish(&context).connect(&endpoint) {
            Ok(socket) => socket,
            Err(e) => {
                error!("Failed to connect publisher socket to {}: {}", endpoint, e);
                return;
            }
        };

        // Small delay for slow joiner (only once at startup)
        tokio::time::sleep(Duration::from_millis(2)).await;
        debug!("Publisher task for {} ready to process messages", endpoint);

        // Process messages
        while let Some(message) = receiver.recv().await {
            match ZmqTransport::serialize_message(&message) {
                Ok(multipart) => {
                    if let Err(e) = pub_socket.send(multipart).await {
                        error!(
                            "Failed to send message via publisher to {}: {}",
                            endpoint, e
                        );
                    } else {
                        debug!("Sent message via publisher to {}", endpoint);
                    }
                }
                Err(e) => {
                    error!("Failed to serialize message for {}: {}", endpoint, e);
                }
            }
        }

        debug!("Publisher task for {} shutting down", endpoint);
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

    async fn send_message(&self, target: InstanceId, handler: &str, payload: Bytes) -> Result<()> {
        let state = self.state.read().await;

        let peer = state
            .peers
            .get(&target)
            .ok_or_else(|| anyhow::anyhow!("Unknown peer: {}", target))?;

        let message = ActiveMessage::new(handler, payload).with_sender(self.instance_id);

        let endpoint = peer.endpoint.clone();
        drop(state);

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

        let mut builder =
            self.system_message("_wait_for_handler")
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
            .system_message("_list_handlers")
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

        let endpoint = peer.endpoint.clone();
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
        let mut state = self.state.write().await;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);

        state
            .pending_acceptances
            .insert(message_id, AcceptanceEntry { sender, deadline });

        debug!("Registered acceptance for message {}", message_id);
        Ok(())
    }

    async fn register_response(
        &self,
        message_id: Uuid,
        sender: oneshot::Sender<Bytes>,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);

        state
            .pending_responses
            .insert(message_id, ResponseEntry { sender, deadline });

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

    async fn has_incoming_connection_from(&self, instance_id: InstanceId) -> bool {
        self.has_incoming_connection_from(instance_id).await
    }
}

impl ZmqActiveMessageClient {
    pub(super) async fn complete_acceptance(
        &self,
        accept_id: Uuid,
        sender: InstanceId,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        if let Some(entry) = state.pending_acceptances.remove(&accept_id) {
            let _ = entry.sender.send(());
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
        let mut state = self.state.write().await;
        if let Some(entry) = state.pending_responses.remove(&response_id) {
            let _ = entry.sender.send(payload);
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
        let mut state = self.state.write().await;
        let now = tokio::time::Instant::now();

        state.pending_acceptances.retain(|accept_id, entry| {
            if now >= entry.deadline {
                warn!("Acceptance {} expired (not received in time)", accept_id);
                false
            } else {
                true
            }
        });

        state.pending_responses.retain(|response_id, entry| {
            if now >= entry.deadline {
                warn!("Response {} expired (not received in time)", response_id);
                false
            } else {
                true
            }
        });
    }
}
