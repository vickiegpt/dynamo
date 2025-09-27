// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use std::collections::VecDeque;
use tmq::{
    AsZmqSocket, Context, Message, Multipart,
    publish::{Publish, publish},
    subscribe::{Subscribe, subscribe},
};
use uuid::Uuid;

use crate::active_message::{client::Endpoint, handler::ActiveMessage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    Tcp,
    Ipc,
}

pub enum ZmqTransport {
    Publisher {
        socket: Publish,
        endpoint: Option<Endpoint>,
        transport_type: TransportType,
    },
    Subscriber {
        socket: Subscribe,
        endpoint: Option<Endpoint>,
        transport_type: TransportType,
    },
}

impl std::fmt::Debug for ZmqTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Publisher {
                endpoint,
                transport_type,
                ..
            } => f
                .debug_struct("Publisher")
                .field("endpoint", endpoint)
                .field("transport_type", transport_type)
                .finish(),
            Self::Subscriber {
                endpoint,
                transport_type,
                ..
            } => f
                .debug_struct("Subscriber")
                .field("endpoint", endpoint)
                .field("transport_type", transport_type)
                .finish(),
        }
    }
}

impl ZmqTransport {
    pub fn new_publisher_bound(context: &Context, address: &str) -> Result<Self> {
        let transport_type = Self::detect_transport_type(address);
        let socket = publish(context).bind(address)?;

        let bound_endpoint = socket
            .get_socket()
            .get_last_endpoint()
            .expect("Failed to retrieve bound endpoint")
            .expect("Socket did not report bound endpoint");

        Ok(Self::Publisher {
            socket,
            endpoint: Some(bound_endpoint),
            transport_type,
        })
    }

    pub fn new_subscriber_bound(context: &Context, address: &str) -> Result<Self> {
        let transport_type = Self::detect_transport_type(address);
        let socket = subscribe(context).bind(address)?.subscribe(b"")?;

        let bound_endpoint = socket
            .get_socket()
            .get_last_endpoint()
            .expect("Failed to retrieve bound endpoint")
            .expect("Socket did not report bound endpoint");

        Ok(Self::Subscriber {
            socket,
            endpoint: Some(bound_endpoint),
            transport_type,
        })
    }

    pub fn new_publisher_connected(context: &Context, endpoint: &str) -> Result<Self> {
        let transport_type = Self::detect_transport_type(endpoint);
        let socket = publish(context).connect(endpoint)?;

        Ok(Self::Publisher {
            socket,
            endpoint: Some(endpoint.to_string()),
            transport_type,
        })
    }

    pub fn new_subscriber_connected(context: &Context, endpoint: &str) -> Result<Self> {
        let transport_type = Self::detect_transport_type(endpoint);
        let socket = subscribe(context).connect(endpoint)?.subscribe(b"")?;

        Ok(Self::Subscriber {
            socket,
            endpoint: Some(endpoint.to_string()),
            transport_type,
        })
    }

    fn detect_transport_type(address: &str) -> TransportType {
        if address.starts_with("ipc://") {
            TransportType::Ipc
        } else {
            TransportType::Tcp
        }
    }

    pub fn serialize_message(message: &ActiveMessage) -> Result<Multipart> {
        let mut parts = VecDeque::new();

        // Part 1: Metadata (everything except payload)
        let metadata = serde_json::json!({
            "message_id": message.message_id,
            "handler_name": message.handler_name,
            "sender_instance": message.sender_instance,
            "metadata": message.metadata,
        });
        parts.push_back(Message::from(serde_json::to_vec(&metadata)?));

        // Part 2: Raw payload bytes (no JSON serialization overhead)
        parts.push_back(Message::from(message.payload.as_ref()));

        Ok(Multipart(parts))
    }

    fn deserialize_message(multipart: Multipart) -> Result<ActiveMessage> {
        if multipart.len() < 2 {
            anyhow::bail!(
                "Invalid multipart message: expected 2 parts, got {}",
                multipart.len()
            );
        }

        // Part 1: Metadata
        let metadata_bytes = &*multipart[0];
        let metadata: serde_json::Value = serde_json::from_slice(metadata_bytes)?;

        // Extract fields from metadata
        let message_id = metadata["message_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing message_id in metadata"))?
            .parse::<Uuid>()
            .map_err(|e| anyhow::anyhow!("Invalid message_id: {}", e))?;

        let handler_name = metadata["handler_name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing handler_name in metadata"))?
            .to_string();

        let sender_instance = metadata["sender_instance"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing sender_instance in metadata"))?
            .parse::<Uuid>()
            .map_err(|e| anyhow::anyhow!("Invalid sender_instance: {}", e))?;

        let message_metadata = metadata["metadata"].clone();

        // Part 2: Raw payload
        let payload = Bytes::from(multipart[1].to_vec());

        Ok(ActiveMessage {
            message_id,
            handler_name,
            sender_instance,
            payload,
            metadata: message_metadata,
        })
    }

    pub fn local_endpoint(&self) -> Option<&Endpoint> {
        match self {
            Self::Publisher { endpoint, .. } => endpoint.as_ref(),
            Self::Subscriber { endpoint, .. } => endpoint.as_ref(),
        }
    }

    pub async fn receive(&mut self) -> Result<ActiveMessage> {
        match self {
            Self::Subscriber { socket, .. } => {
                if let Some(Ok(multipart)) = socket.next().await {
                    Self::deserialize_message(multipart)
                } else {
                    anyhow::bail!("Failed to receive message")
                }
            }
            Self::Publisher { .. } => {
                anyhow::bail!("Cannot receive on publisher socket")
            }
        }
    }
}
