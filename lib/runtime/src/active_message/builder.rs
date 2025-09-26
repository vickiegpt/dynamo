// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Message builder with typestate pattern for ergonomic and safe message construction.

use anyhow::Result;
use bytes::Bytes;
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use std::time::Duration;
use tokio::sync::oneshot;
use uuid::Uuid;

use super::client::ActiveMessageClient;
use super::handler::{ActiveMessage, InstanceId};
use super::status::{DetachedConfirm, MessageStatus, SendAndConfirm, WithResponse};

/// TypeState marker: Builder needs delivery mode selection
pub struct NeedsDeliveryMode;

/// TypeState marker: Builder configured with response expectation
pub struct WithResponseExpected;

/// Message builder with typestate for compile-time safety
pub struct MessageBuilder<'a, Mode = NeedsDeliveryMode> {
    client: &'a dyn ActiveMessageClient,
    handler_name: String,
    payload: Option<Bytes>,
    timeout: Duration,
    _mode: PhantomData<Mode>,
}

/// Methods available for initial builder state
impl<'a> MessageBuilder<'a, NeedsDeliveryMode> {
    /// Create a new message builder
    pub fn new(client: &'a dyn ActiveMessageClient, handler: &str) -> Self {
        Self {
            client,
            handler_name: handler.to_string(),
            payload: None,
            timeout: Duration::from_secs(30),
            _mode: PhantomData,
        }
    }

    /// Set payload from serializable type
    pub fn payload<T: Serialize>(mut self, data: T) -> Self {
        self.payload = Some(Bytes::from(
            serde_json::to_vec(&data).expect("Failed to serialize payload"),
        ));
        self
    }

    /// Set raw payload bytes
    pub fn raw_payload(mut self, data: Bytes) -> Self {
        self.payload = Some(data);
        self
    }

    /// Set timeout for response waiting
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = duration;
        self
    }

    /// Fire and forget - no confirmation or responses
    pub async fn fire_and_forget(self, target: InstanceId) -> Result<()> {
        let message = ActiveMessage {
            message_id: Uuid::new_v4(),
            handler_name: self.handler_name,
            sender_instance: self.client.instance_id(),
            payload: self.payload.unwrap_or_default(),
            metadata: serde_json::json!({
                "_mode": "fire_and_forget"
            }),
        };

        self.client.send_raw_message(target, message).await
    }

    /// DEFAULT: Send and wait for acceptance confirmation
    pub async fn send(self, target: InstanceId) -> Result<MessageStatus<SendAndConfirm>> {
        self.send_and_confirm(target).await
    }

    /// Explicit: Send and wait for acceptance confirmation
    pub async fn send_and_confirm(self, target: InstanceId) -> Result<MessageStatus<SendAndConfirm>> {
        let message_id = Uuid::new_v4();
        let (accept_tx, accept_rx) = oneshot::channel();

        // Register for acceptance notification
        self.client
            .register_acceptance(message_id, accept_tx)
            .await?;

        let message = ActiveMessage {
            message_id,
            handler_name: self.handler_name,
            sender_instance: self.client.instance_id(),
            payload: self.payload.unwrap_or_default(),
            metadata: serde_json::json!({
                "_mode": "confirmed",
                "_accept_id": message_id.to_string()
            }),
        };

        self.client.send_raw_message(target, message).await?;

        // Wait for acceptance
        tokio::time::timeout(self.timeout, accept_rx)
            .await
            .map_err(|_| anyhow::anyhow!("Handler acceptance timeout after {:?}", self.timeout))?
            .map_err(|_| anyhow::anyhow!("Handler acceptance channel dropped"))?;

        Ok(MessageStatus::new(message_id, None, None, self.timeout))
    }

    /// Send without waiting, get status object for later confirmation
    pub async fn send_detached(self, target: InstanceId) -> Result<MessageStatus<DetachedConfirm>> {
        let message_id = Uuid::new_v4();
        let (accept_tx, accept_rx) = oneshot::channel();

        // Register for acceptance notification
        self.client
            .register_acceptance(message_id, accept_tx)
            .await?;

        let message = ActiveMessage {
            message_id,
            handler_name: self.handler_name,
            sender_instance: self.client.instance_id(),
            payload: self.payload.unwrap_or_default(),
            metadata: serde_json::json!({
                "_mode": "confirmed",
                "_accept_id": message_id.to_string()
            }),
        };

        self.client.send_raw_message(target, message).await?;

        Ok(MessageStatus::new(
            message_id,
            Some(accept_rx),
            None,
            self.timeout,
        ))
    }

    /// Configure to expect additional response beyond acceptance
    pub fn expect_response<R: DeserializeOwned + 'static>(
        self,
    ) -> MessageBuilder<'a, WithResponseExpected> {
        MessageBuilder {
            client: self.client,
            handler_name: self.handler_name,
            payload: self.payload,
            timeout: self.timeout,
            _mode: PhantomData,
        }
    }
}

/// Methods available when response is expected
impl<'a> MessageBuilder<'a, WithResponseExpected> {
    /// Set payload from serializable type
    pub fn payload<T: Serialize>(mut self, data: T) -> Self {
        self.payload = Some(Bytes::from(
            serde_json::to_vec(&data).expect("Failed to serialize payload"),
        ));
        self
    }

    /// Set raw payload bytes
    pub fn raw_payload(mut self, data: Bytes) -> Self {
        self.payload = Some(data);
        self
    }

    /// Set timeout for response waiting
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = duration;
        self
    }

    /// Send and await acceptance, return status with response awaiter
    pub async fn send(self, target: InstanceId) -> Result<MessageStatus<WithResponse>> {
        let message_id = Uuid::new_v4();
        let (accept_tx, accept_rx) = oneshot::channel();
        let (response_tx, response_rx) = oneshot::channel();

        // Register for both acceptance and response
        self.client
            .register_acceptance(message_id, accept_tx)
            .await?;
        self.client
            .register_response(message_id, response_tx)
            .await?;

        let message = ActiveMessage {
            message_id,
            handler_name: self.handler_name,
            sender_instance: self.client.instance_id(),
            payload: self.payload.unwrap_or_default(),
            metadata: serde_json::json!({
                "_mode": "with_response",
                "_accept_id": message_id.to_string(),
                "_response_id": message_id.to_string()
            }),
        };

        self.client.send_raw_message(target, message).await?;

        // Wait for acceptance
        tokio::time::timeout(self.timeout, accept_rx)
            .await
            .map_err(|_| anyhow::anyhow!("Handler acceptance timeout after {:?}", self.timeout))?
            .map_err(|_| anyhow::anyhow!("Handler acceptance channel dropped"))?;

        Ok(MessageStatus::new(
            message_id,
            None,
            Some(response_rx),
            self.timeout,
        ))
    }

    /// Send detached - don't wait for acceptance, return status with both awaiters
    pub async fn send_detached(self, target: InstanceId) -> Result<MessageStatus<DetachedConfirm>> {
        // For simplicity, with_response mode always waits for acceptance first
        // We can add a DetachedWithResponse mode later if needed
        self.send(target)
            .await
            .map(|status| MessageStatus::new(status.message_id, None, None, status.timeout))
    }
}