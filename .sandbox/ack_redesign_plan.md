# ACK System Redesign - Response Context Pattern Implementation

## Overview
Implement Option 1: Response Context Pattern for ergonomic ACK and response handling.

## Core Design

### 1. Response Context Types
```rust
// lib/runtime/src/active_message/response.rs

use tokio::sync::oneshot;
use serde::{Serialize, Deserialize, de::DeserializeOwned};

/// Response context provided to handlers
pub enum ResponseContext {
    /// No response expected
    None,
    /// Simple acknowledgment expected
    Ack(AckSender),
    /// Single typed response expected
    Single(Box<dyn SingleResponseSender>),
    /// Stream of responses expected
    Stream(Box<dyn StreamResponseSender>),
}

/// Simple ACK sender
pub struct AckSender {
    sender: oneshot::Sender<()>,
    message_id: Uuid,
    target: InstanceId,
}

impl AckSender {
    /// Send acknowledgment
    pub async fn send(self) -> Result<()> {
        // This will internally send the ACK message
        self.sender.send(()).map_err(|_| anyhow!("ACK receiver dropped"))?;
        Ok(())
    }

    /// Send acknowledgment with metadata
    pub async fn send_with_metadata(self, metadata: serde_json::Value) -> Result<()> {
        // For future extension
        self.send().await
    }
}

/// Single response sender
pub trait SingleResponseSender: Send {
    async fn send_raw(&mut self, data: Bytes) -> Result<()>;
}

/// Typed single response sender
pub struct TypedSingleResponseSender<T: Serialize> {
    sender: oneshot::Sender<Bytes>,
    message_id: Uuid,
    target: InstanceId,
    _phantom: PhantomData<T>,
}

impl<T: Serialize + Send> TypedSingleResponseSender<T> {
    pub async fn send(self, response: T) -> Result<()> {
        let bytes = Bytes::from(serde_json::to_vec(&response)?);
        self.sender.send(bytes).map_err(|_| anyhow!("Response receiver dropped"))?;
        Ok(())
    }
}

/// Stream response sender (future enhancement)
pub trait StreamResponseSender: Send {
    async fn send_item(&mut self, data: Bytes) -> Result<()>;
    async fn close(self) -> Result<()>;
}
```

### 2. Updated Handler Trait
```rust
// lib/runtime/src/active_message/handler.rs

#[async_trait]
pub trait ActiveMessageHandler: Send + Sync + std::fmt::Debug {
    async fn handle(
        &self,
        message: ActiveMessage,
        client: &dyn ActiveMessageClient,  // Changed from Arc<dyn>
        response: ResponseContext,
    ) -> Result<()>;

    fn name(&self) -> &str;

    fn schema(&self) -> Option<&serde_json::Value> {
        None
    }
}

// Add deserialization helper to ActiveMessage
impl ActiveMessage {
    /// Deserialize payload to typed request
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.payload)
            .map_err(|e| anyhow!("Failed to deserialize payload: {}", e))
    }

    /// Get payload as JSON value
    pub fn as_json(&self) -> Result<serde_json::Value> {
        serde_json::from_slice(&self.payload)
            .map_err(|e| anyhow!("Failed to parse payload as JSON: {}", e))
    }
}
```

### 3. Message Builder API
```rust
// lib/runtime/src/active_message/builder.rs

pub struct MessageBuilder<'a> {
    client: &'a dyn ActiveMessageClient,
    handler_name: String,
    payload: Option<Bytes>,
    response_type: ResponseType,
    timeout: Duration,
}

pub enum ResponseType {
    None,
    Ack,
    Single { type_id: TypeId },
    Stream { type_id: TypeId },
}

impl<'a> MessageBuilder<'a> {
    pub fn new(client: &'a dyn ActiveMessageClient, handler: impl Into<String>) -> Self {
        Self {
            client,
            handler_name: handler.into(),
            payload: None,
            response_type: ResponseType::None,
            timeout: Duration::from_secs(30),
        }
    }

    /// Set payload from serializable type
    pub fn payload<T: Serialize>(mut self, data: T) -> Self {
        self.payload = Some(Bytes::from(serde_json::to_vec(&data).unwrap()));
        self
    }

    /// Set raw payload
    pub fn raw_payload(mut self, data: Bytes) -> Self {
        self.payload = Some(data);
        self
    }

    /// Expect ACK response
    pub fn expect_ack(mut self) -> Self {
        self.response_type = ResponseType::Ack;
        self
    }

    /// Expect single typed response
    pub fn expect_response<R: DeserializeOwned + 'static>(mut self) -> Self {
        self.response_type = ResponseType::Single {
            type_id: TypeId::of::<R>()
        };
        self
    }

    /// Set timeout for response
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = duration;
        self
    }

    /// Send to specific target
    pub async fn send(self, target: InstanceId) -> Result<MessageResponse> {
        let message_id = Uuid::new_v4();

        // Set up response channel based on type
        let response_handle = match self.response_type {
            ResponseType::None => ResponseHandle::None,
            ResponseType::Ack => {
                let (tx, rx) = oneshot::channel();
                // Register ACK expectation
                self.client.register_response(message_id, ResponseExpectation::Ack(tx)).await?;
                ResponseHandle::Ack(rx)
            }
            ResponseType::Single { .. } => {
                let (tx, rx) = oneshot::channel();
                self.client.register_response(message_id, ResponseExpectation::Single(tx)).await?;
                ResponseHandle::Single(rx)
            }
            ResponseType::Stream { .. } => {
                unimplemented!("Stream responses in future PR")
            }
        };

        // Build message with metadata
        let mut metadata = serde_json::json!({});
        if matches!(self.response_type, ResponseType::Ack | ResponseType::Single { .. }) {
            metadata["_response_type"] = serde_json::json!(match self.response_type {
                ResponseType::Ack => "ack",
                ResponseType::Single { .. } => "single",
                _ => "none",
            });
            metadata["_response_id"] = serde_json::json!(message_id.to_string());
        }

        // Send the message
        let message = ActiveMessage {
            message_id,
            handler_name: self.handler_name,
            sender_instance: self.client.instance_id(),
            payload: self.payload.unwrap_or_default(),
            metadata,
        };

        self.client.send_raw_message(target, message).await?;

        Ok(MessageResponse {
            message_id,
            handle: response_handle,
            timeout: self.timeout,
        })
    }

    /// Broadcast to all peers
    pub async fn broadcast(self) -> Result<Vec<MessageResponse>> {
        // Implementation for broadcast with response collection
        unimplemented!("Broadcast in future PR")
    }
}

/// Response handle for awaiting responses
pub struct MessageResponse {
    message_id: Uuid,
    handle: ResponseHandle,
    timeout: Duration,
}

enum ResponseHandle {
    None,
    Ack(oneshot::Receiver<()>),
    Single(oneshot::Receiver<Bytes>),
    Stream(mpsc::Receiver<Bytes>),
}

impl MessageResponse {
    /// Wait for ACK
    pub async fn await_ack(self) -> Result<()> {
        match self.handle {
            ResponseHandle::Ack(rx) => {
                tokio::time::timeout(self.timeout, rx)
                    .await
                    .map_err(|_| anyhow!("ACK timeout after {:?}", self.timeout))?
                    .map_err(|_| anyhow!("ACK sender dropped"))?;
                Ok(())
            }
            _ => Err(anyhow!("Message was not configured to expect ACK")),
        }
    }

    /// Wait for typed response
    pub async fn await_response<T: DeserializeOwned>(self) -> Result<T> {
        match self.handle {
            ResponseHandle::Single(rx) => {
                let bytes = tokio::time::timeout(self.timeout, rx)
                    .await
                    .map_err(|_| anyhow!("Response timeout after {:?}", self.timeout))?
                    .map_err(|_| anyhow!("Response sender dropped"))?;
                Ok(serde_json::from_slice(&bytes)?)
            }
            _ => Err(anyhow!("Message was not configured to expect response")),
        }
    }
}
```

### 4. Client Interface Updates
```rust
// lib/runtime/src/active_message/client.rs

#[async_trait]
pub trait ActiveMessageClient: Send + Sync + std::fmt::Debug {
    // Existing methods...

    /// Create a message builder
    fn message(&self, handler: impl Into<String>) -> MessageBuilder {
        MessageBuilder::new(self, handler)
    }

    /// Send raw message (internal use)
    async fn send_raw_message(&self, target: InstanceId, message: ActiveMessage) -> Result<()>;

    /// Register response expectation (internal use)
    async fn register_response(&self, message_id: Uuid, expectation: ResponseExpectation) -> Result<()>;
}

pub enum ResponseExpectation {
    Ack(oneshot::Sender<()>),
    Single(oneshot::Sender<Bytes>),
    Stream(mpsc::Sender<Bytes>),
}
```

### 5. Update Existing Handlers

Example migration for compute handler:
```rust
// Before
#[async_trait]
impl ActiveMessageHandler for ComputeHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: Arc<dyn ActiveMessageClient>,
    ) -> Result<()> {
        let request: ComputeRequest = serde_json::from_slice(&message.payload)?;
        // Process...
        Ok(())
    }
}

// After
#[async_trait]
impl ActiveMessageHandler for ComputeHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
        response: ResponseContext,
    ) -> Result<()> {
        let request: ComputeRequest = message.deserialize()?;

        let result = match request.operation.as_str() {
            "add" => request.x + request.y,
            "multiply" => request.x * request.y,
            _ => return Err(anyhow!("Unknown operation")),
        };

        // Handle response based on context
        match response {
            ResponseContext::Ack(ack) => {
                ack.send().await?;
            }
            ResponseContext::Single(sender) => {
                // For typed responses in future
                let response = ComputeResponse { result };
                sender.send(response).await?;
            }
            ResponseContext::None => {
                // No response expected
            }
            _ => {}
        }

        Ok(())
    }
}
```

### 6. Built-in ACK Handler Becomes Internal
```rust
// The _ack handler becomes internal to the framework
// Users never manually send to "_ack" handler

// In ZmqActiveMessageManager's message processing:
async fn process_message(&self, message: ActiveMessage) {
    // Check if this is a response to a pending request
    if let Some(response_id) = message.metadata.get("_response_to") {
        let response_id = Uuid::parse_str(response_id.as_str().unwrap()).unwrap();

        // Complete the pending response
        self.complete_response(response_id, message).await;
        return;
    }

    // Regular handler dispatch with response context setup
    let response_context = self.setup_response_context(&message).await;

    // Dispatch to handler
    handler.handle(message, &self.client, response_context).await?;
}
```

## Implementation Steps

### Phase 1: Core Types (2 hours)
1. Create `response.rs` with ResponseContext types
2. Create `builder.rs` with MessageBuilder
3. Update `handler.rs` trait signature
4. Add deserialization helpers to ActiveMessage

### Phase 2: Client Updates (2 hours)
1. Update ActiveMessageClient trait
2. Implement message builder in ZmqActiveMessageClient
3. Add response registration tracking
4. Update message sending to include metadata

### Phase 3: Manager Updates (2 hours)
1. Update ZmqActiveMessageManager to handle response context
2. Internalize ACK handling
3. Set up response routing
4. Update message processing pipeline

### Phase 4: Handler Migration (1 hour)
1. Update all built-in handlers
2. Update example handlers
3. Add backward compatibility layer (optional)

### Phase 5: Testing & Documentation (1 hour)
1. Update integration tests
2. Add builder pattern tests
3. Update documentation examples
4. Migration guide

## Benefits of This Design

1. **Ergonomic API**: Clean, chainable builder pattern
2. **Type Safety**: Compile-time checking of response types
3. **Separation of Concerns**: Response handling separate from business logic
4. **Extensible**: Easy to add new response patterns
5. **Backward Compatible**: Can provide compatibility layer
6. **Testable**: Response context can be mocked

## Example Usage After Implementation

```rust
// Client side - sending with ACK
let response = client
    .message("compute")
    .payload(ComputeRequest { x: 5, y: 10, operation: "add".into() })
    .expect_ack()
    .timeout(Duration::from_secs(5))
    .send(worker_id)
    .await?;

response.await_ack().await?;
println!("Worker acknowledged computation");

// Client side - sending with typed response
let response = client
    .message("compute")
    .payload(ComputeRequest { x: 5, y: 10, operation: "add".into() })
    .expect_response::<ComputeResponse>()
    .send(worker_id)
    .await?;

let result = response.await_response::<ComputeResponse>().await?;
println!("Result: {}", result.value);

// Handler side - automatic ACK
impl ActiveMessageHandler for ComputeHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        client: &dyn ActiveMessageClient,
        response: ResponseContext,
    ) -> Result<()> {
        let request: ComputeRequest = message.deserialize()?;

        // Do computation...

        // One-line ACK if requested
        if let ResponseContext::Ack(ack) = response {
            ack.send().await?;
        }

        Ok(())
    }
}
```

## Migration Strategy

1. Add new traits/types without breaking existing code
2. Provide adapter for old handler signature
3. Gradually migrate handlers
4. Remove old ACK system in future PR
5. Document migration path clearly