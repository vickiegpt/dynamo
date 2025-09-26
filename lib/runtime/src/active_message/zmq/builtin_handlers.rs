// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, info};

use crate::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::{ActiveMessage, ActiveMessageHandler, HandlerEvent, HandlerId},
    response::ResponseContext,
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
    endpoint: String,
}

#[async_trait]
impl ActiveMessageHandler for RegisterServiceHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        _response: ResponseContext,
    ) -> Result<()> {
        let payload: RegisterServicePayload = message.deserialize()?;

        let instance_id = uuid::Uuid::parse_str(&payload.instance_id)?;
        let peer = PeerInfo::new(instance_id, payload.endpoint.clone());

        self.client.connect_to_peer(peer).await?;

        info!(
            "Registered service {} at {}",
            payload.instance_id, payload.endpoint
        );

        Ok(())
    }

    fn name(&self) -> &str {
        "_register_service"
    }

    fn schema(&self) -> Option<&serde_json::Value> {
        static SCHEMA: once_cell::sync::Lazy<serde_json::Value> =
            once_cell::sync::Lazy::new(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "instance_id": { "type": "string" },
                        "endpoint": { "type": "string" }
                    },
                    "required": ["instance_id", "endpoint"]
                })
            });
        Some(&SCHEMA)
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
impl ActiveMessageHandler for ListHandlersHandler {
    async fn handle(
        &self,
        _message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        _response: ResponseContext,
    ) -> Result<()> {
        let state = self.state.read().await;
        let handlers: Vec<HandlerId> = state.handlers.keys().cloned().collect();

        debug!("Available handlers: {:?}", handlers);

        Ok(())
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
impl ActiveMessageHandler for WaitForHandlerHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        _response: ResponseContext,
    ) -> Result<()> {
        let payload: WaitForHandlerPayload = message.deserialize()?;

        let state = self.state.read().await;
        if state.handlers.contains_key(&payload.handler_name) {
            debug!("Handler '{}' already registered", payload.handler_name);
            return Ok(());
        }
        drop(state);

        let mut rx = self.handler_events_tx.subscribe();

        let timeout = payload
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(Duration::from_secs(30));

        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Ok(HandlerEvent::Registered { name, .. }) => {
                            if name == payload.handler_name {
                                debug!("Handler '{}' is now registered", payload.handler_name);
                                return Ok(());
                            }
                        }
                        Ok(HandlerEvent::Deregistered { .. }) => {}
                        Err(_) => {
                            anyhow::bail!("Handler event channel closed");
                        }
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    anyhow::bail!(
                        "Timeout waiting for handler '{}'",
                        payload.handler_name
                    );
                }
            }
        }
    }

    fn name(&self) -> &str {
        "_wait_for_handler"
    }

    fn schema(&self) -> Option<&serde_json::Value> {
        static SCHEMA: once_cell::sync::Lazy<serde_json::Value> =
            once_cell::sync::Lazy::new(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "handler_name": { "type": "string" },
                        "timeout_ms": { "type": "number" }
                    },
                    "required": ["handler_name"]
                })
            });
        Some(&SCHEMA)
    }
}

#[derive(Debug)]
pub struct HealthCheckHandler;

#[async_trait]
impl ActiveMessageHandler for HealthCheckHandler {
    async fn handle(
        &self,
        _message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        _response: ResponseContext,
    ) -> Result<()> {
        debug!("Health check received");
        Ok(())
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
}

#[async_trait]
impl ActiveMessageHandler for AckHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        _response: ResponseContext,
    ) -> Result<()> {
        let payload: AckPayload = message.deserialize()?;
        let ack_id = uuid::Uuid::parse_str(&payload.ack_id)?;

        self.client
            .complete_ack(ack_id, message.sender_instance)
            .await?;

        Ok(())
    }

    fn name(&self) -> &str {
        "_ack"
    }

    fn schema(&self) -> Option<&serde_json::Value> {
        static SCHEMA: once_cell::sync::Lazy<serde_json::Value> =
            once_cell::sync::Lazy::new(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "ack_id": { "type": "string" }
                    },
                    "required": ["ack_id"]
                })
            });
        Some(&SCHEMA)
    }
}
