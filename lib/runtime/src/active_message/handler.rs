// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active Message handler trait and implementations

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use jsonschema::JSONSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::active_message::client::ActiveMessageClient;

/// Message metadata and execution context passed to handlers
#[derive(Debug)]
pub struct ActiveMessageContext {
    /// Message ID
    pub message_id: Uuid,
    /// Instance ID of the sender
    pub sender_instance: Uuid,
    /// Name of the target handler
    pub handler_name: String,
    /// Message metadata
    pub metadata: serde_json::Value,
    /// Client for sending messages
    client: Arc<dyn ActiveMessageClient>,
    /// Optional cancellation token for cancellable handlers
    cancel_token: Option<CancellationToken>,
}

impl ActiveMessageContext {
    /// Create a new message context
    pub fn new(
        message_id: Uuid,
        sender_instance: Uuid,
        handler_name: String,
        metadata: serde_json::Value,
        client: Arc<dyn ActiveMessageClient>,
        cancel_token: Option<CancellationToken>,
    ) -> Self {
        Self {
            message_id,
            sender_instance,
            handler_name,
            metadata,
            client,
            cancel_token,
        }
    }

    /// Get the message ID
    pub fn message_id(&self) -> Uuid {
        self.message_id
    }

    /// Get the sender instance ID
    pub fn sender_instance(&self) -> Uuid {
        self.sender_instance
    }

    /// Check if this message has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.as_ref().is_some_and(|t| t.is_cancelled())
    }

    /// Get cancellation token for spawning cancellable tasks
    pub fn cancel_token(&self) -> Option<&CancellationToken> {
        self.cancel_token.as_ref()
    }

    /// Get the handler name
    pub fn handler_name(&self) -> &str {
        &self.handler_name
    }

    /// Get message metadata
    pub fn metadata(&self) -> &serde_json::Value {
        &self.metadata
    }

    /// Get the client for sending messages
    pub fn client(&self) -> &Arc<dyn ActiveMessageClient> {
        &self.client
    }
}

/// The original raw active message structure for transport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveMessage {
    /// Unique identifier for this message
    pub message_id: Uuid,
    /// Name of the target handler
    pub handler_name: String,
    /// Instance ID of the sender
    pub sender_instance: Uuid,
    /// Message payload
    pub payload: Bytes,
    /// Optional metadata
    pub metadata: serde_json::Value,
}

impl ActiveMessage {
    /// Create a new active message
    pub fn new(
        message_id: Uuid,
        handler_name: String,
        sender_instance: Uuid,
        payload: Bytes,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            message_id,
            handler_name,
            sender_instance,
            payload,
            metadata,
        }
    }

    /// Create an ActiveMessage with default values for testing
    pub fn test(handler_name: impl Into<String>, payload: Bytes) -> Self {
        Self::new(
            Uuid::new_v4(),
            handler_name.into(),
            Uuid::new_v4(),
            payload,
            serde_json::Value::Null,
        )
    }

    /// Deserialize the payload as JSON
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.payload).map_err(Into::into)
    }
}

pub type InstanceId = Uuid;
pub type HandlerId = String;

/// Events emitted during handler operations
#[derive(Debug, Clone)]
pub enum HandlerEvent {
    /// Handler was registered
    Registered { name: String, instance: InstanceId },
    /// Handler was deregistered
    Deregistered { name: String, instance: InstanceId },
}

// Base handler traits - no generics, work with raw bytes

/// Handler that doesn't return any value
#[async_trait]
pub trait NoReturnHandler: Send + Sync + std::fmt::Debug {
    /// Handle a message with no response expected
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext);

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }

    /// Get compiled JSON schema for payload validation (cached)
    fn compiled_schema(&self) -> Option<&JSONSchema> {
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
            let instance = JSONSchema::compile(schema)
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
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext) -> Result<()>;

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }

    /// Get compiled JSON schema for payload validation (cached)
    fn compiled_schema(&self) -> Option<&JSONSchema> {
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
            let instance = JSONSchema::compile(schema)
                .map_err(|e| anyhow::anyhow!("Invalid schema: {}", e))?;

            instance.validate(&value).map_err(|errors| {
                let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                anyhow::anyhow!("Schema validation failed: {}", error_messages.join(", "))
            })?;
        }
        Ok(())
    }
}

/// Handler that returns serialized response bytes
#[async_trait]
pub trait ResponseHandler: Send + Sync + std::fmt::Debug {
    /// Handle a message and return serialized response
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext) -> Result<Bytes>;

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }

    /// Get compiled JSON schema for payload validation (cached)
    fn compiled_schema(&self) -> Option<&JSONSchema> {
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
            let instance = JSONSchema::compile(schema)
                .map_err(|e| anyhow::anyhow!("Invalid schema: {}", e))?;

            instance.validate(&value).map_err(|errors| {
                let error_messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                anyhow::anyhow!("Schema validation failed: {}", error_messages.join(", "))
            })?;
        }
        Ok(())
    }
}

// Typed handler traits - provide type safety for users
// These automatically implement the base traits via blanket impls

/// Typed handler for messages that don't need any response
#[async_trait]
pub trait TypedNoReturnHandler<Input>: Send + Sync + std::fmt::Debug
where
    Input: DeserializeOwned + Send + 'static,
{
    /// Handle a typed message with no response expected
    async fn handle_typed(&self, input: Input, ctx: ActiveMessageContext);

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }
}

/// Typed handler that returns Result<()> - sends ACK on success, NACK on error
#[async_trait]
pub trait TypedAckHandler<Input>: Send + Sync + std::fmt::Debug
where
    Input: DeserializeOwned + Send + 'static,
{
    /// Handle a typed message that expects acknowledgment
    async fn handle_typed(&self, input: Input, ctx: ActiveMessageContext) -> Result<()>;

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }
}

/// Typed handler that returns a typed response
#[async_trait]
pub trait TypedResponseHandler<Input, Response>: Send + Sync + std::fmt::Debug
where
    Input: DeserializeOwned + Send + 'static,
    Response: Serialize + Send + 'static,
{
    /// Handle a typed message and return a typed response
    async fn handle_typed(&self, input: Input, ctx: ActiveMessageContext) -> Result<Response>;

    /// Handler name
    fn name(&self) -> &str;

    /// Optional JSON schema for payload validation
    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }
}

// Note: Auto-bridging implementations removed due to conflicts with direct implementations.
// Users can create wrapper types that implement TypedHandler traits and manually implement
// the base traits, or use the closure-based handlers for simple cases.

/// Unified handler type that can hold any of the three handler patterns
#[derive(Debug, Clone)]
pub enum HandlerType {
    /// No return handler - no response expected
    NoReturn(Arc<dyn NoReturnHandler>),
    /// Acknowledgment handler - sends ACK/NACK based on Result<()>
    Ack(Arc<dyn AckHandler>),
    /// Response handler - sends serialized response or error
    Response(Arc<dyn ResponseHandler>),
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
        Self::Response(Arc::new(handler))
    }

    /// Get the handler name
    pub fn name(&self) -> &str {
        match self {
            Self::NoReturn(h) => h.name(),
            Self::Ack(h) => h.name(),
            Self::Response(h) => h.name(),
        }
    }

    /// Validate the payload against the handler's schema
    pub fn validate_schema(&self, payload: &Bytes) -> Result<()> {
        match self {
            Self::NoReturn(h) => h.validate_schema(payload),
            Self::Ack(h) => h.validate_schema(payload),
            Self::Response(h) => h.validate_schema(payload),
        }
    }

    /// Create a NoReturnHandler from a closure
    ///
    /// # Example
    /// ```rust
    /// let handler = HandlerType::from_no_return_closure("log", |msg, _client| async move {
    ///     println!("Received: {:?}", msg);
    /// });
    /// ```
    pub fn from_no_return_closure<F, Fut>(name: impl Into<String>, closure: F) -> Self
    where
        F: Fn(Bytes, ActiveMessageContext) -> Fut + Send + Sync + 'static,
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
        F: Fn(Bytes, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        Self::ack(ClosureAckHandler::new(name.into(), closure))
    }

    /// Create a ResponseHandler from a closure that returns Result<Bytes>
    ///
    /// # Example
    /// ```rust
    /// let handler = HandlerType::from_response_closure("compute", |msg, _client| async move {
    ///     let input: ComputeRequest = serde_json::from_slice(&msg)?;
    ///     let result = compute_something(input);
    ///     Ok(serde_json::to_vec(&result)?.into())
    /// });
    /// ```
    pub fn from_response_closure<F, Fut>(name: impl Into<String>, closure: F) -> Self
    where
        F: Fn(Bytes, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Bytes>> + Send + 'static,
    {
        Self::response(ClosureResponseHandler::new(name.into(), closure))
    }

    /// Create a typed unary handler from a closure that takes typed input and returns typed output
    ///
    /// # Example
    /// ```rust
    /// let handler = HandlerType::from_unary_closure("compute", |req: ComputeRequest, _ctx| async move {
    ///     Ok(ComputeResponse { result: req.x + req.y })
    /// });
    /// ```
    pub fn from_unary_closure<Req, Res, F, Fut>(name: impl Into<String>, closure: F) -> Self
    where
        Req: DeserializeOwned + Send + 'static,
        Res: Serialize + Send + 'static,
        F: Fn(Req, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Res>> + Send + 'static,
    {
        Self::response(TypedUnaryClosureHandler::new(name.into(), closure))
    }

    /// Create a typed void handler from a closure that takes typed input with no return
    ///
    /// # Example
    /// ```rust
    /// let handler = HandlerType::from_void_closure("log", |msg: String, _ctx| async move {
    ///     println!("Received: {}", msg);
    /// });
    /// ```
    pub fn from_void_closure<Input, F, Fut>(name: impl Into<String>, closure: F) -> Self
    where
        Input: DeserializeOwned + Send + 'static,
        F: Fn(Input, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        Self::no_return(TypedVoidClosureHandler::new(name.into(), closure))
    }

    /// Create a typed acknowledgment handler from a closure that takes typed input
    ///
    /// # Example
    /// ```rust
    /// let handler = HandlerType::from_typed_ack_closure("validate", |data: ValidationRequest, _ctx| async move {
    ///     validate(&data)?;
    ///     Ok(())
    /// });
    /// ```
    pub fn from_typed_ack_closure<Input, F, Fut>(name: impl Into<String>, closure: F) -> Self
    where
        Input: DeserializeOwned + Send + 'static,
        F: Fn(Input, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        Self::ack(TypedAckClosureHandler::new(name.into(), closure))
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
    F: Fn(Bytes, ActiveMessageContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext) {
        (self.closure)(input, ctx).await
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
    F: Fn(Bytes, ActiveMessageContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext) -> Result<()> {
        (self.closure)(input, ctx).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Wrapper for closures that return Result<Bytes>
pub struct ClosureResponseHandler<F> {
    name: String,
    closure: F,
}

impl<F> std::fmt::Debug for ClosureResponseHandler<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosureResponseHandler")
            .field("name", &self.name)
            .field("closure", &"<closure>")
            .finish()
    }
}

impl<F> ClosureResponseHandler<F> {
    pub fn new(name: String, closure: F) -> Self {
        Self { name, closure }
    }
}

#[async_trait]
impl<F, Fut> ResponseHandler for ClosureResponseHandler<F>
where
    F: Fn(Bytes, ActiveMessageContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Bytes>> + Send + 'static,
{
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext) -> Result<Bytes> {
        (self.closure)(input, ctx).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Typed closure wrapper implementations for better ergonomics

/// Typed wrapper for request/response closures
pub struct TypedUnaryClosureHandler<Req, Res, F> {
    name: String,
    closure: F,
    _phantom: std::marker::PhantomData<fn(Req) -> Res>,
}

impl<Req, Res, F> std::fmt::Debug for TypedUnaryClosureHandler<Req, Res, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedUnaryClosureHandler")
            .field("name", &self.name)
            .field("closure", &"<closure>")
            .finish()
    }
}

impl<Req, Res, F> TypedUnaryClosureHandler<Req, Res, F> {
    pub fn new(name: String, closure: F) -> Self {
        Self {
            name,
            closure,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<Req, Res, F, Fut> ResponseHandler for TypedUnaryClosureHandler<Req, Res, F>
where
    Req: DeserializeOwned + Send + 'static,
    Res: Serialize + Send + 'static,
    F: Fn(Req, ActiveMessageContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Res>> + Send + 'static,
{
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext) -> Result<Bytes> {
        let request: Req = serde_json::from_slice(&input)?;
        let response = (self.closure)(request, ctx).await?;
        let response_bytes = serde_json::to_vec(&response)?;
        Ok(Bytes::from(response_bytes))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Typed wrapper for void/no-return closures
pub struct TypedVoidClosureHandler<Input, F> {
    name: String,
    closure: F,
    _phantom: std::marker::PhantomData<fn(Input)>,
}

impl<Input, F> std::fmt::Debug for TypedVoidClosureHandler<Input, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedVoidClosureHandler")
            .field("name", &self.name)
            .field("closure", &"<closure>")
            .finish()
    }
}

impl<Input, F> TypedVoidClosureHandler<Input, F> {
    pub fn new(name: String, closure: F) -> Self {
        Self {
            name,
            closure,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<Input, F, Fut> NoReturnHandler for TypedVoidClosureHandler<Input, F>
where
    Input: DeserializeOwned + Send + 'static,
    F: Fn(Input, ActiveMessageContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext) {
        if let Ok(typed_input) = serde_json::from_slice::<Input>(&input) {
            (self.closure)(typed_input, ctx).await;
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Typed wrapper for acknowledgment closures
pub struct TypedAckClosureHandler<Input, F> {
    name: String,
    closure: F,
    _phantom: std::marker::PhantomData<fn(Input)>,
}

impl<Input, F> std::fmt::Debug for TypedAckClosureHandler<Input, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedAckClosureHandler")
            .field("name", &self.name)
            .field("closure", &"<closure>")
            .finish()
    }
}

impl<Input, F> TypedAckClosureHandler<Input, F> {
    pub fn new(name: String, closure: F) -> Self {
        Self {
            name,
            closure,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<Input, F, Fut> AckHandler for TypedAckClosureHandler<Input, F>
where
    Input: DeserializeOwned + Send + 'static,
    F: Fn(Input, ActiveMessageContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    async fn handle(&self, input: Bytes, ctx: ActiveMessageContext) -> Result<()> {
        let typed_input: Input = serde_json::from_slice(&input)?;
        (self.closure)(typed_input, ctx).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_message::client::ActiveMessageClient;
    use serde::{Deserialize, Serialize};
    use std::time::Duration;
    use tokio::sync::Mutex;

    #[derive(Debug, Serialize, Deserialize)]
    struct TestRequest {
        x: i32,
        y: i32,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestResponse {
        sum: i32,
    }

    #[tokio::test]
    async fn test_typed_unary_closure_handler() {
        let handler =
            TypedUnaryClosureHandler::new("add".to_string(), |req: TestRequest, _ctx| async move {
                Ok(TestResponse { sum: req.x + req.y })
            });

        let request = TestRequest { x: 5, y: 10 };
        let request_bytes = Bytes::from(serde_json::to_vec(&request).unwrap());

        let ctx = ActiveMessageContext::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "add".to_string(),
            serde_json::Value::Null,
            std::sync::Arc::new(MockClient {
                instance_id: Uuid::new_v4(),
                endpoint: "test://localhost".to_string(),
            }),
            None,
        );

        let response_bytes = handler.handle(request_bytes, ctx).await.unwrap();
        let response: TestResponse = serde_json::from_slice(&response_bytes).unwrap();

        assert_eq!(response, TestResponse { sum: 15 });
    }

    #[tokio::test]
    async fn test_typed_void_closure_handler() {
        use std::sync::Arc;

        let messages = Arc::new(Mutex::new(Vec::<String>::new()));
        let messages_clone = messages.clone();

        let handler = TypedVoidClosureHandler::new("log".to_string(), move |msg: String, _ctx| {
            let messages = messages_clone.clone();
            async move {
                messages.lock().await.push(msg);
            }
        });

        let message = "Hello, World!";
        let message_bytes = Bytes::from(serde_json::to_vec(&message).unwrap());

        let ctx = ActiveMessageContext::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "log".to_string(),
            serde_json::Value::Null,
            std::sync::Arc::new(MockClient {
                instance_id: Uuid::new_v4(),
                endpoint: "test://localhost".to_string(),
            }),
            None,
        );

        handler.handle(message_bytes, ctx).await;

        let logged = messages.lock().await;
        assert_eq!(logged.len(), 1);
        assert_eq!(logged[0], "Hello, World!");
    }

    #[tokio::test]
    async fn test_typed_ack_closure_handler() {
        let handler =
            TypedAckClosureHandler::new("validate".to_string(), |text: String, _ctx| async move {
                if text.len() >= 5 {
                    Ok(())
                } else {
                    anyhow::bail!("Text too short")
                }
            });

        let ctx = ActiveMessageContext::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "validate".to_string(),
            serde_json::Value::Null,
            std::sync::Arc::new(MockClient {
                instance_id: Uuid::new_v4(),
                endpoint: "test://localhost".to_string(),
            }),
            None,
        );

        // Test with valid input
        let valid_text = "Valid text";
        let valid_bytes = Bytes::from(serde_json::to_vec(&valid_text).unwrap());
        let result = handler.handle(valid_bytes, ctx).await;
        assert!(result.is_ok());

        // Create another context for the second test
        let ctx2 = ActiveMessageContext::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "validate".to_string(),
            serde_json::Value::Null,
            std::sync::Arc::new(MockClient {
                instance_id: Uuid::new_v4(),
                endpoint: "test://localhost".to_string(),
            }),
            None,
        );

        // Test with invalid input
        let invalid_text = "Hi";
        let invalid_bytes = Bytes::from(serde_json::to_vec(&invalid_text).unwrap());
        let result = handler.handle(invalid_bytes, ctx2).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Text too short"));
    }

    // Mock client for testing
    #[derive(Debug)]
    struct MockClient {
        instance_id: Uuid,
        endpoint: String,
    }

    #[async_trait]
    impl ActiveMessageClient for MockClient {
        fn instance_id(&self) -> Uuid {
            self.instance_id
        }

        fn endpoint(&self) -> &str {
            &self.endpoint
        }

        async fn send_message(&self, _: Uuid, _: &str, _: Bytes) -> Result<()> {
            Ok(())
        }

        async fn broadcast_message(&self, _: &str, _: Bytes) -> Result<()> {
            Ok(())
        }

        async fn list_peers(&self) -> Result<Vec<crate::active_message::client::PeerInfo>> {
            Ok(vec![])
        }

        async fn connect_to_peer(&self, _: crate::active_message::client::PeerInfo) -> Result<()> {
            Ok(())
        }

        async fn disconnect_from_peer(&self, _: Uuid) -> Result<()> {
            Ok(())
        }

        async fn await_handler(&self, _: Uuid, _: &str, _: Option<Duration>) -> Result<bool> {
            Ok(true)
        }

        async fn list_handlers(&self, _: Uuid) -> Result<Vec<String>> {
            Ok(vec![])
        }

        async fn send_raw_message(&self, _: Uuid, _: ActiveMessage) -> Result<()> {
            Ok(())
        }

        async fn register_acceptance(
            &self,
            _: Uuid,
            _: tokio::sync::oneshot::Sender<()>,
        ) -> Result<()> {
            Ok(())
        }

        async fn register_response(
            &self,
            _: Uuid,
            _: tokio::sync::oneshot::Sender<Bytes>,
        ) -> Result<()> {
            Ok(())
        }

        async fn register_ack(
            &self,
            _: Uuid,
            _: Duration,
        ) -> Result<tokio::sync::oneshot::Receiver<Result<(), String>>> {
            let (_tx, rx) = tokio::sync::oneshot::channel();
            Ok(rx)
        }

        async fn has_incoming_connection_from(&self, _: Uuid) -> bool {
            false
        }

        fn clone_as_arc(&self) -> std::sync::Arc<dyn ActiveMessageClient> {
            std::sync::Arc::new(MockClient {
                instance_id: self.instance_id,
                endpoint: self.endpoint.clone(),
            })
        }
    }
}
