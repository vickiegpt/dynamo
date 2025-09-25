// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::SinkExt;
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};
use tmq::{Context, Message, Multipart, publish};
use tokio::sync::{RwLock, oneshot};
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::active_message::{
    client::{ActiveMessageClient, Endpoint, PeerInfo},
    handler::{ActiveMessage, HandlerId, InstanceId},
};

struct AckEntry {
    sender: oneshot::Sender<()>,
    deadline: tokio::time::Instant,
}

impl std::fmt::Debug for AckEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AckEntry")
            .field("deadline", &self.deadline)
            .finish()
    }
}

struct ClientState {
    peers: HashMap<InstanceId, PeerInfo>,
    pending_acks: HashMap<Uuid, AckEntry>,
}

impl std::fmt::Debug for ClientState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientState")
            .field("peers", &self.peers)
            .field("pending_acks", &self.pending_acks)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct ZmqActiveMessageClient {
    instance_id: InstanceId,
    endpoint: Endpoint,
    state: Arc<RwLock<ClientState>>,
}

impl ZmqActiveMessageClient {
    pub fn new(instance_id: InstanceId, endpoint: Endpoint) -> Self {
        let state = ClientState {
            peers: HashMap::new(),
            pending_acks: HashMap::new(),
        };

        Self {
            instance_id,
            endpoint,
            state: Arc::new(RwLock::new(state)),
        }
    }

    pub async fn register_ack(
        &self,
        ack_id: Uuid,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<()>> {
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
        });

        let payload_bytes = Bytes::from(serde_json::to_vec(&payload)?);

        self.send_message(target, "_ack", payload_bytes).await
    }

    pub(super) async fn complete_ack(&self, ack_id: Uuid, sender: InstanceId) -> Result<()> {
        let mut state = self.state.write().await;

        if let Some(entry) = state.pending_acks.remove(&ack_id) {
            let _ = entry.sender.send(());
            debug!("Completed ACK: {} from {}", ack_id, sender);
            Ok(())
        } else {
            error!("Received unexpected ACK: {} from {}", ack_id, sender);
            anyhow::bail!("Unexpected ACK: {} from {}", ack_id, sender)
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

    async fn send_raw(&self, endpoint: &str, message: &ActiveMessage) -> Result<()> {
        let context = Context::new();
        let mut pub_socket = publish(&context).connect(endpoint)?;

        tokio::time::sleep(Duration::from_millis(10)).await;

        let serialized = serde_json::to_vec(message)?;
        let mut parts = VecDeque::new();
        parts.push_back(Message::from(serialized));
        let multipart = Multipart(parts);

        pub_socket.send(multipart).await?;

        Ok(())
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
    ) -> Result<()> {
        let payload = serde_json::json!({
            "handler_name": handler,
            "timeout_ms": timeout.map(|d| d.as_millis()),
        });

        let payload_bytes = Bytes::from(serde_json::to_vec(&payload)?);

        self.send_message(instance_id, "_wait_for_handler", payload_bytes)
            .await
    }

    async fn list_handlers(&self, instance_id: InstanceId) -> Result<Vec<HandlerId>> {
        let payload = Bytes::new();

        self.send_message(instance_id, "_list_handlers", payload)
            .await?;

        Ok(vec![])
    }
}
