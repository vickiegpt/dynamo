// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use std::time::Duration;

use super::handler::{ActiveMessage, HandlerId, InstanceId};

pub type Endpoint = String;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeerInfo {
    pub instance_id: InstanceId,
    pub endpoint: Endpoint,
}

impl PeerInfo {
    pub fn new(instance_id: InstanceId, endpoint: impl Into<String>) -> Self {
        Self {
            instance_id,
            endpoint: endpoint.into(),
        }
    }

    pub fn is_local(&self, my_endpoint: &str) -> bool {
        if self.endpoint.starts_with("ipc://") {
            return true;
        }

        if let (Some(my_host), Some(peer_host)) =
            (extract_host(my_endpoint), extract_host(&self.endpoint))
        {
            my_host == peer_host
        } else {
            false
        }
    }
}

fn extract_host(endpoint: &str) -> Option<String> {
    if let Some(stripped) = endpoint.strip_prefix("tcp://") {
        stripped.split(':').next().map(|s| s.to_string())
    } else {
        None
    }
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
    ) -> Result<()>;

    async fn list_handlers(&self, instance_id: InstanceId) -> Result<Vec<HandlerId>>;
}
