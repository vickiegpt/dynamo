// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport abstraction for active message system.
//!
//! This module provides both high-level and low-level transport abstractions:
//! - `Transport`: High-level transport that works with ActiveMessage structs (legacy)
//! - `RawTransport`: Low-level transport that only handles raw bytes (preferred)
//!
//! The RawTransport trait represents a pure transport abstraction that:
//! - Only deals with raw bytes, not business logic
//! - Handles connection lifecycle and message delivery
//! - Can be implemented for any underlying transport (ZMQ, TCP, HTTP, etc.)

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

use super::client::Endpoint;
use super::handler::ActiveMessage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    Tcp,
    Ipc,
}

/// High-level transport trait that works with ActiveMessage structs.
///
/// This is the legacy transport interface that still mixes business logic
/// with transport concerns. Prefer RawTransport for new implementations.
#[async_trait]
pub trait Transport: Send + Sync + std::fmt::Debug {
    async fn bind(&mut self, address: &str) -> Result<Endpoint>;

    async fn connect(&mut self, endpoint: &Endpoint) -> Result<()>;

    async fn disconnect(&mut self, endpoint: &Endpoint) -> Result<()>;

    async fn send(&mut self, message: &ActiveMessage) -> Result<()>;

    async fn receive(&mut self) -> Result<ActiveMessage>;

    fn transport_type(&self) -> TransportType;

    fn local_endpoint(&self) -> Option<&Endpoint>;
}

/// Low-level transport trait that only handles raw bytes.
///
/// This trait represents a pure transport abstraction that separates
/// transport concerns from business logic. Implementations should:
/// - Only handle byte-level message delivery
/// - Manage connection lifecycle
/// - Not parse or interpret message contents
/// - Be agnostic to the active message protocol
#[async_trait]
pub trait RawTransport: Send + Sync + std::fmt::Debug {
    /// Bind the transport to a local address and return the bound endpoint.
    async fn bind(&mut self, address: &str) -> Result<String>;

    /// Connect to a remote endpoint.
    async fn connect(&mut self, endpoint: &str) -> Result<()>;

    /// Disconnect from a remote endpoint.
    async fn disconnect(&mut self, endpoint: &str) -> Result<()>;

    /// Send raw bytes to a specific target.
    /// The target format is transport-specific (e.g., socket address for TCP).
    async fn send(&mut self, target: &str, data: &[u8]) -> Result<()>;

    /// Receive raw bytes from the transport.
    /// Returns the data and sender information (transport-specific format).
    async fn receive(&mut self) -> Result<(Bytes, String)>;

    /// Get the transport type.
    fn transport_type(&self) -> TransportType;

    /// Get the local endpoint this transport is bound to.
    fn local_endpoint(&self) -> Option<&str>;

    /// Check if the transport is connected to a specific endpoint.
    fn is_connected(&self, endpoint: &str) -> bool;
}

/// Transport factory for creating different transport implementations.
pub trait TransportFactory: Send + Sync {
    /// Create a new raw transport instance.
    fn create_raw_transport(&self, transport_type: TransportType) -> Result<Box<dyn RawTransport>>;
}
