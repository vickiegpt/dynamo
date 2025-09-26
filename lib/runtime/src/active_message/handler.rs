// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::future::Future;
use std::sync::Arc;
use uuid::Uuid;

use super::client::ActiveMessageClient;

pub type HandlerId = String;
pub type InstanceId = Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveMessage {
    pub message_id: Uuid,
    pub handler_name: HandlerId,
    pub sender_instance: InstanceId,
    pub payload: Bytes,
    pub metadata: serde_json::Value,
}

impl ActiveMessage {
    pub fn new(handler_name: impl Into<String>, payload: Bytes) -> Self {
        Self {
            message_id: Uuid::new_v4(),
            handler_name: handler_name.into(),
            sender_instance: Uuid::new_v4(),
            payload,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_sender(mut self, sender: InstanceId) -> Self {
        self.sender_instance = sender;
        self
    }

    /// Deserialize payload to typed request
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.payload)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize payload: {}", e))
    }

    /// Get payload as JSON value
    pub fn as_json(&self) -> Result<serde_json::Value> {
        serde_json::from_slice(&self.payload)
            .map_err(|e| anyhow::anyhow!("Failed to parse payload as JSON: {}", e))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HandlerEvent {
    Registered {
        name: HandlerId,
        instance: InstanceId,
    },
    Deregistered {
        name: HandlerId,
        instance: InstanceId,
    },
}

pub const SYSTEM_HANDLER_PREFIX: &str = "_";

pub fn is_system_handler(name: &str) -> bool {
    name.starts_with(SYSTEM_HANDLER_PREFIX)
}

pub fn validate_handler_name(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("Handler name cannot be empty");
    }

    if is_system_handler(name) {
        anyhow::bail!(
            "User-space handlers cannot start with '{}'. This prefix is reserved for system handlers.",
            SYSTEM_HANDLER_PREFIX
        );
    }

    Ok(())
}

// New handler traits based on return type patterns

/// Handler for messages that don't need any response
#[async_trait]
pub trait NoReturnHandler: Send + Sync + std::fmt::Debug {
    /// Handle a message with no response expected
    async fn handle(&self, message: ActiveMessage, client: &dyn ActiveMessageClient);

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }

    /// Get compiled JSON schema for payload validation (cached)
    fn compiled_schema(&self) -> Option<&jsonschema::JSONSchema> {
        None
    }

    /// Validate payload against schema
    fn validate_schema(&self, payload: &Bytes) -> Result<()> {
        // First try to use cached compiled schema
        if let Some(compiled) = self.compiled_schema() {
            let value: serde_json::Value = serde_json::from_slice(payload)?;
            compiled.validate(&value).map_err(|errors| {
                let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                anyhow::anyhow!("Schema validation failed: {}", error_messages.join(", "))
            })?;
        } else if let Some(schema) = self.schema() {
            // Fallback to compiling schema on each call (less efficient)
            let value: serde_json::Value = serde_json::from_slice(payload)?;
            let instance = jsonschema::JSONSchema::compile(schema)
                .map_err(|e| anyhow::anyhow!("Invalid schema: {}", e))?;

            instance.validate(&value).map_err(|errors| {
                let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                anyhow::anyhow!("Schema validation failed: {}", error_messages.join(", "))
            })?;
        }
        Ok(())
    }
}

/// Handler that returns Result<()> - sends ACK on success, NACK on error
#[async_trait]
pub trait AckHandler: Send + Sync + std::fmt::Debug {
    /// Handle a message that expects acknowledgment
    async fn handle(&self, message: ActiveMessage, client: &dyn ActiveMessageClient) -> Result<()>;

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }

    /// Get compiled JSON schema for payload validation (cached)
    fn compiled_schema(&self) -> Option<&jsonschema::JSONSchema> {
        None
    }

    /// Validate payload against schema
    fn validate_schema(&self, payload: &Bytes) -> Result<()> {
        // First try to use cached compiled schema
        if let Some(compiled) = self.compiled_schema() {
            let value: serde_json::Value = serde_json::from_slice(payload)?;
            compiled.validate(&value).map_err(|errors| {
                let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                anyhow::anyhow!("Schema validation failed: {}", error_messages.join(", "))
            })?;
        } else if let Some(schema) = self.schema() {
            // Fallback to compiling schema on each call (less efficient)
            let value: serde_json::Value = serde_json::from_slice(payload)?;
            let instance = jsonschema::JSONSchema::compile(schema)
                .map_err(|e| anyhow::anyhow!("Invalid schema: {}", e))?;

            instance.validate(&value).map_err(|errors| {
                let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                anyhow::anyhow!("Schema validation failed: {}", error_messages.join(", "))
            })?;
        }
        Ok(())
    }
}

/// Handler that returns Result<T> where T is serializable - sends typed response or error
#[async_trait]
pub trait ResponseHandler: Send + Sync + std::fmt::Debug {
    /// The response type this handler returns
    type Response: Serialize + Send + 'static;

    /// Handle a message that expects a typed response
    async fn handle(
        &self,
        message: ActiveMessage,
        client: &dyn ActiveMessageClient,
    ) -> Result<Self::Response>;

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }

    /// Get compiled JSON schema for payload validation (cached)
    fn compiled_schema(&self) -> Option<&jsonschema::JSONSchema> {
        None
    }

    /// Validate payload against schema
    fn validate_schema(&self, payload: &Bytes) -> Result<()> {
        // First try to use cached compiled schema
        if let Some(compiled) = self.compiled_schema() {
            let value: serde_json::Value = serde_json::from_slice(payload)?;
            compiled.validate(&value).map_err(|errors| {
                let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                anyhow::anyhow!("Schema validation failed: {}", error_messages.join(", "))
            })?;
        } else if let Some(schema) = self.schema() {
            // Fallback to compiling schema on each call (less efficient)
            let value: serde_json::Value = serde_json::from_slice(payload)?;
            let instance = jsonschema::JSONSchema::compile(schema)
                .map_err(|e| anyhow::anyhow!("Invalid schema: {}", e))?;

            instance.validate(&value).map_err(|errors| {
                let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                anyhow::anyhow!("Schema validation failed: {}", error_messages.join(", "))
            })?;
        }
        Ok(())
    }
}

/// Type-erased wrapper for ResponseHandler to avoid generic parameters in storage
#[async_trait]
pub trait ResponseHandlerWrapper: Send + Sync + std::fmt::Debug {
    /// Handle message and automatically send response using the provided sender
    async fn handle_and_send(
        &self,
        message: ActiveMessage,
        client: &dyn ActiveMessageClient,
        sender: super::response::SingleResponseSender,
    ) -> Result<()>;

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }

    /// Get compiled JSON schema for payload validation (cached)
    fn compiled_schema(&self) -> Option<&jsonschema::JSONSchema> {
        None
    }

    /// Validate payload against schema
    fn validate_schema(&self, payload: &Bytes) -> Result<()>;
}

/// Concrete implementation of ResponseHandlerWrapper for any ResponseHandler
#[derive(Debug)]
pub struct ResponseHandlerWrapperImpl<T: ResponseHandler> {
    handler: T,
}

impl<T: ResponseHandler> ResponseHandlerWrapperImpl<T> {
    pub fn new(handler: T) -> Self {
        Self { handler }
    }
}

#[async_trait]
impl<T: ResponseHandler> ResponseHandlerWrapper for ResponseHandlerWrapperImpl<T> {
    async fn handle_and_send(
        &self,
        message: ActiveMessage,
        client: &dyn ActiveMessageClient,
        sender: super::response::SingleResponseSender,
    ) -> Result<()> {
        match self.handler.handle(message, client).await {
            Ok(response) => sender.send(response).await,
            Err(error) => sender.send_err(error).await,
        }
    }

    fn name(&self) -> &str {
        self.handler.name()
    }

    fn schema(&self) -> Option<&serde_json::Value> {
        self.handler.schema()
    }

    fn compiled_schema(&self) -> Option<&jsonschema::JSONSchema> {
        self.handler.compiled_schema()
    }

    fn validate_schema(&self, payload: &Bytes) -> Result<()> {
        self.handler.validate_schema(payload)
    }
}

/// Unified handler type that can hold any of the three handler patterns
#[derive(Debug, Clone)]
pub enum HandlerType {
    /// No return handler - no response expected
    NoReturn(Arc<dyn NoReturnHandler>),
    /// Acknowledgment handler - sends ACK/NACK based on Result<()>
    Ack(Arc<dyn AckHandler>),
    /// Response handler - sends typed response or error
    Response(Arc<dyn ResponseHandlerWrapper>),
}

impl HandlerType {
    /// Create a new no-return handler
    pub fn no_return<T: NoReturnHandler + 'static>(handler: T) -> Self {
        Self::NoReturn(Arc::new(handler))
    }

    /// Create a new fire-and-forget handler (deprecated alias)
    #[deprecated(note = "Use no_return instead to better reflect handler purpose")]
    pub fn fire_and_forget<T: NoReturnHandler + 'static>(handler: T) -> Self {
        Self::NoReturn(Arc::new(handler))
    }

    /// Create a new acknowledgment handler
    pub fn ack<T: AckHandler + 'static>(handler: T) -> Self {
        Self::Ack(Arc::new(handler))
    }

    /// Create a new response handler
    pub fn response<T: ResponseHandler + 'static>(handler: T) -> Self {
        Self::Response(Arc::new(ResponseHandlerWrapperImpl::new(handler)))
    }

    /// Get handler name
    pub fn name(&self) -> &str {
        match self {
            HandlerType::NoReturn(h) => h.name(),
            HandlerType::Ack(h) => h.name(),
            HandlerType::Response(h) => h.name(),
        }
    }

    /// Get schema for validation
    pub fn schema(&self) -> Option<&serde_json::Value> {
        match self {
            HandlerType::NoReturn(h) => h.schema(),
            HandlerType::Ack(h) => h.schema(),
            HandlerType::Response(h) => h.schema(),
        }
    }

    /// Get compiled schema for validation
    pub fn compiled_schema(&self) -> Option<&jsonschema::JSONSchema> {
        match self {
            HandlerType::NoReturn(h) => h.compiled_schema(),
            HandlerType::Ack(h) => h.compiled_schema(),
            HandlerType::Response(h) => h.compiled_schema(),
        }
    }

    /// Validate payload against schema
    pub fn validate_schema(&self, payload: &Bytes) -> Result<()> {
        match self {
            HandlerType::NoReturn(h) => h.validate_schema(payload),
            HandlerType::Ack(h) => h.validate_schema(payload),
            HandlerType::Response(h) => h.validate_schema(payload),
        }
    }

    /// Create a NoReturnHandler from a closure that returns ()
    ///
    /// # Example
    /// ```rust
    /// let handler = HandlerType::from_no_return_closure("log", |msg, _client| async move {
    ///     println!("Received: {:?}", msg);
    /// });
    /// ```
    pub fn from_no_return_closure<F, Fut>(name: impl Into<String>, closure: F) -> Self
    where
        F: Fn(ActiveMessage, &dyn ActiveMessageClient) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        Self::no_return(ClosureNoReturnHandler::new(name.into(), closure))
    }

    /// Create an AckHandler from a closure that returns Result<()>
    ///
    /// # Example
    /// ```rust
    /// let handler = HandlerType::from_ack_closure("validate", |msg, _client| async move {
    ///     validate_data(&msg)?;
    ///     Ok(())
    /// });
    /// ```
    pub fn from_ack_closure<F, Fut>(name: impl Into<String>, closure: F) -> Self
    where
        F: Fn(ActiveMessage, &dyn ActiveMessageClient) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        Self::ack(ClosureAckHandler::new(name.into(), closure))
    }

    /// Create a ResponseHandler from a closure that returns Result<T>
    ///
    /// # Example
    /// ```rust
    /// let handler = HandlerType::from_response_closure("compute", |msg, _client| async move {
    ///     let input: ComputeRequest = msg.deserialize()?;
    ///     let result = compute_something(input);
    ///     Ok(result)
    /// });
    /// ```
    pub fn from_response_closure<F, Fut, T>(name: impl Into<String>, closure: F) -> Self
    where
        F: Fn(ActiveMessage, &dyn ActiveMessageClient) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
        T: Serialize + Send + Sync + 'static,
    {
        Self::response(ClosureResponseHandler::new(name.into(), closure))
    }
}

// Closure wrapper implementations

/// Wrapper for closures that don't return a value
pub struct ClosureNoReturnHandler<F> {
    name: String,
    closure: F,
}

impl<F> std::fmt::Debug for ClosureNoReturnHandler<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosureNoReturnHandler")
            .field("name", &self.name)
            .field("closure", &"<closure>")
            .finish()
    }
}

impl<F> ClosureNoReturnHandler<F> {
    pub fn new(name: String, closure: F) -> Self {
        Self { name, closure }
    }
}

#[async_trait]
impl<F, Fut> NoReturnHandler for ClosureNoReturnHandler<F>
where
    F: Fn(ActiveMessage, &dyn ActiveMessageClient) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    async fn handle(&self, message: ActiveMessage, client: &dyn ActiveMessageClient) {
        (self.closure)(message, client).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Wrapper for closures that return Result<()>
pub struct ClosureAckHandler<F> {
    name: String,
    closure: F,
}

impl<F> std::fmt::Debug for ClosureAckHandler<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosureAckHandler")
            .field("name", &self.name)
            .field("closure", &"<closure>")
            .finish()
    }
}

impl<F> ClosureAckHandler<F> {
    pub fn new(name: String, closure: F) -> Self {
        Self { name, closure }
    }
}

#[async_trait]
impl<F, Fut> AckHandler for ClosureAckHandler<F>
where
    F: Fn(ActiveMessage, &dyn ActiveMessageClient) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    async fn handle(&self, message: ActiveMessage, client: &dyn ActiveMessageClient) -> Result<()> {
        (self.closure)(message, client).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Wrapper for closures that return Result<T>
pub struct ClosureResponseHandler<F, T> {
    name: String,
    closure: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<F, T> std::fmt::Debug for ClosureResponseHandler<F, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosureResponseHandler")
            .field("name", &self.name)
            .field("closure", &"<closure>")
            .finish()
    }
}

impl<F, T> ClosureResponseHandler<F, T> {
    pub fn new(name: String, closure: F) -> Self {
        Self {
            name,
            closure,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<F, Fut, T> ResponseHandler for ClosureResponseHandler<F, T>
where
    F: Fn(ActiveMessage, &dyn ActiveMessageClient) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<T>> + Send + 'static,
    T: Serialize + Send + Sync + 'static,
{
    type Response = T;

    async fn handle(&self, message: ActiveMessage, client: &dyn ActiveMessageClient) -> Result<T> {
        (self.closure)(message, client).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}
