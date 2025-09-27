# ACK System Redesign v2 - TypeState Pattern with Acceptance Confirmation

## Core Concept
Use typestate pattern to provide compile-time guarantees about what operations are available based on the delivery mode selected.

## Delivery Modes

### 1. Default: Send and Confirm (Automatic Acceptance)
```rust
// DEFAULT BEHAVIOR - automatically awaits acceptance
let status = client
    .message("compute")
    .payload(ComputeRequest { x: 5, y: 10 })
    .send(worker_id)  // This is actually send_and_confirm() by default
    .await?;  // Blocks until handler accepts and validates

// Status only has additional response methods if configured
// The acceptance is already confirmed at this point
```

### 2. Fire and Forget
```rust
// Explicitly opt into fire-and-forget
client
    .message("compute")
    .payload(ComputeRequest { x: 5, y: 10 })
    .fire_and_forget(worker_id)
    .await?;  // Only waits for local send, not remote acceptance

// Returns () - no status object at all
```

### 3. Detached Confirmation
```rust
// Separate send from confirmation
let status = client
    .message("compute")
    .payload(ComputeRequest { x: 5, y: 10 })
    .send_detached(worker_id)
    .await?;  // Returns immediately after send

// Can await acceptance later
status.await_accepted().await?;
```

## TypeState Implementation

```rust
// lib/runtime/src/active_message/builder.rs

use std::marker::PhantomData;

// Marker types for typestate
pub struct NeedsDeliveryMode;
pub struct FireAndForget;
pub struct SendAndConfirm;
pub struct DetachedConfirm;
pub struct WithResponse;

// The builder with typestate
pub struct MessageBuilder<'a, Mode = NeedsDeliveryMode> {
    client: &'a dyn ActiveMessageClient,
    handler_name: String,
    payload: Option<Bytes>,
    response_type: ResponseType,
    timeout: Duration,
    _mode: PhantomData<Mode>,
}

// Initial builder state
impl<'a> MessageBuilder<'a, NeedsDeliveryMode> {
    pub fn new(client: &'a dyn ActiveMessageClient, handler: impl Into<String>) -> Self {
        Self {
            client,
            handler_name: handler.into(),
            payload: None,
            response_type: ResponseType::None,
            timeout: Duration::from_secs(30),
            _mode: PhantomData,
        }
    }

    pub fn payload<T: Serialize>(mut self, data: T) -> Self {
        self.payload = Some(Bytes::from(serde_json::to_vec(&data).unwrap()));
        self
    }

    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = duration;
        self
    }

    // === Delivery Mode Selection ===

    /// Fire and forget - no confirmation or responses
    pub async fn fire_and_forget(self, target: InstanceId) -> Result<()> {
        let message = self.build_message(MessageMode::FireAndForget);
        self.client.send_raw_message(target, message).await?;
        Ok(())
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
        self.client.register_acceptance(message_id, accept_tx).await?;

        let message = self.build_message_with_id(message_id, MessageMode::Confirmed);
        self.client.send_raw_message(target, message).await?;

        // Wait for acceptance
        tokio::time::timeout(self.timeout, accept_rx)
            .await
            .map_err(|_| anyhow!("Handler acceptance timeout"))?
            .map_err(|_| anyhow!("Handler channel dropped"))?;

        Ok(MessageStatus {
            message_id,
            response_handle: None,
            _mode: PhantomData,
        })
    }

    /// Send without waiting, get status object for later confirmation
    pub async fn send_detached(self, target: InstanceId) -> Result<MessageStatus<DetachedConfirm>> {
        let message_id = Uuid::new_v4();
        let (accept_tx, accept_rx) = oneshot::channel();

        // Register for acceptance notification
        self.client.register_acceptance(message_id, accept_tx).await?;

        let message = self.build_message_with_id(message_id, MessageMode::Confirmed);
        self.client.send_raw_message(target, message).await?;

        Ok(MessageStatus {
            message_id,
            acceptance_rx: Some(accept_rx),
            response_handle: None,
            timeout: self.timeout,
            _mode: PhantomData,
        })
    }

    /// Configure to expect additional response beyond acceptance
    pub fn expect_response<R: DeserializeOwned>(self) -> MessageBuilder<'a, WithResponse> {
        MessageBuilder {
            client: self.client,
            handler_name: self.handler_name,
            payload: self.payload,
            response_type: ResponseType::Single,
            timeout: self.timeout,
            _mode: PhantomData,
        }
    }
}

// Builder with response configured
impl<'a> MessageBuilder<'a, WithResponse> {
    /// Send and await acceptance, return status with response awaiter
    pub async fn send(self, target: InstanceId) -> Result<MessageStatus<WithResponse>> {
        let message_id = Uuid::new_v4();
        let (accept_tx, accept_rx) = oneshot::channel();
        let (response_tx, response_rx) = oneshot::channel();

        // Register for both acceptance and response
        self.client.register_acceptance(message_id, accept_tx).await?;
        self.client.register_response(message_id, response_tx).await?;

        let message = self.build_message_with_id(message_id, MessageMode::WithResponse);
        self.client.send_raw_message(target, message).await?;

        // Wait for acceptance
        tokio::time::timeout(self.timeout, accept_rx)
            .await
            .map_err(|_| anyhow!("Handler acceptance timeout"))?
            .map_err(|_| anyhow!("Handler channel dropped"))?;

        Ok(MessageStatus {
            message_id,
            response_rx: Some(response_rx),
            timeout: self.timeout,
            _mode: PhantomData,
        })
    }
}

// Status objects with different capabilities based on mode
pub struct MessageStatus<Mode> {
    message_id: Uuid,
    acceptance_rx: Option<oneshot::Receiver<()>>,  // Only for DetachedConfirm
    response_rx: Option<oneshot::Receiver<Bytes>>,  // Only for WithResponse
    timeout: Duration,
    _mode: PhantomData<Mode>,
}

// Methods only available for DetachedConfirm mode
impl MessageStatus<DetachedConfirm> {
    pub async fn await_accepted(self) -> Result<()> {
        let rx = self.acceptance_rx.expect("DetachedConfirm should have acceptance_rx");
        tokio::time::timeout(self.timeout, rx)
            .await
            .map_err(|_| anyhow!("Acceptance timeout"))?
            .map_err(|_| anyhow!("Acceptance channel dropped"))?;
        Ok(())
    }
}

// Methods only available for WithResponse mode
impl MessageStatus<WithResponse> {
    pub async fn await_response<T: DeserializeOwned>(self) -> Result<T> {
        let rx = self.response_rx.expect("WithResponse should have response_rx");
        let bytes = tokio::time::timeout(self.timeout, rx)
            .await
            .map_err(|_| anyhow!("Response timeout"))?
            .map_err(|_| anyhow!("Response channel dropped"))?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

// SendAndConfirm has no methods - acceptance already confirmed
impl MessageStatus<SendAndConfirm> {
    // No additional methods - acceptance was already awaited
}
```

## Handler Side - Automatic Acceptance

```rust
// The framework automatically sends acceptance BEFORE calling the handler
// This happens in the manager's message processing pipeline

// lib/runtime/src/active_message/zmq/manager.rs
impl ZmqActiveMessageManager {
    async fn process_incoming_message(&self, raw_message: Vec<u8>) {
        let message: ActiveMessage = serde_json::from_slice(&raw_message)?;

        // Find handler
        let handler = self.get_handler(&message.handler_name)?;

        // Validate schema if present
        handler.validate_schema(&message.payload)?;

        // === AUTOMATIC ACCEPTANCE ===
        // If message expects confirmation, send it NOW before handler runs
        if let Some(mode) = message.metadata.get("_mode").and_then(|v| v.as_str()) {
            match mode {
                "confirmed" | "with_response" => {
                    // Send acceptance notification back to sender
                    let accept_msg = ActiveMessage::internal_acceptance(
                        message.message_id,
                        self.instance_id,
                    );
                    self.client.send_raw_message(message.sender_instance, accept_msg).await?;
                }
                "fire_and_forget" => {
                    // No acceptance needed
                }
                _ => {}
            }
        }

        // Now set up response context if needed
        let response_context = if message.metadata.get("_mode") == Some("with_response") {
            ResponseContext::Single(create_response_sender(message.sender_instance, message.message_id))
        } else {
            ResponseContext::None
        };

        // Call handler AFTER acceptance sent
        handler.handle(message, &self.client, response_context).await?;
    }
}

// Handler implementation is simpler - no acceptance ACK needed
#[async_trait]
impl ActiveMessageHandler for ComputeHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        client: &dyn ActiveMessageClient,
        response: ResponseContext,
    ) -> Result<()> {
        let request: ComputeRequest = message.deserialize()?;

        let result = request.x + request.y;

        // Only handle additional response if configured
        if let ResponseContext::Single(sender) = response {
            sender.send(ComputeResponse { result }).await?;
        }

        // No acceptance ACK needed - framework handled it!
        Ok(())
    }
}
```

## Usage Examples

```rust
// Example 1: Default behavior - wait for acceptance
let status = client
    .message("compute")
    .payload(ComputeRequest { x: 5, y: 10 })
    .send(worker_id)  // Blocks until handler accepts
    .await?;
// At this point, we KNOW the handler accepted and validated the message

// Example 2: Fire and forget
client
    .message("log")
    .payload(LogMessage { text: "Event occurred" })
    .fire_and_forget(worker_id)
    .await?;  // Returns immediately after local send

// Example 3: Detached confirmation
let status = client
    .message("compute")
    .payload(ComputeRequest { x: 5, y: 10 })
    .send_detached(worker_id)
    .await?;

// Do other work...

// Check acceptance later
status.await_accepted().await?;

// Example 4: With additional response
let status = client
    .message("compute")
    .payload(ComputeRequest { x: 5, y: 10 })
    .expect_response::<ComputeResponse>()
    .send(worker_id)  // Waits for acceptance first
    .await?;

let result = status.await_response::<ComputeResponse>().await?;

// Example 5: Compile-time safety
let status = client
    .message("compute")
    .payload(ComputeRequest { x: 5, y: 10 })
    .fire_and_forget(worker_id)
    .await?;

// This won't compile - fire_and_forget returns ()
// status.await_accepted().await?;  // COMPILE ERROR!
```

## Benefits of This Design

1. **TypeState Safety**: Can't call methods that don't make sense for the delivery mode
2. **Sensible Default**: Default behavior includes acceptance confirmation
3. **Clear Intent**: Method names clearly indicate behavior
4. **Framework Handles Acceptance**: Handlers don't need to manually ACK acceptance
5. **Flexible**: Three modes cover all use cases
6. **Performance**: Fire-and-forget for when confirmation isn't needed

## Internal Message Types

```rust
enum InternalMessageType {
    Acceptance { message_id: Uuid },
    Response { message_id: Uuid, payload: Bytes },
    Error { message_id: Uuid, error: String },
}
```

## Migration Path

1. Current handlers that manually send ACKs can be simplified
2. The "_ack" handler becomes "_accept" internally
3. Response handling becomes separate from acceptance
4. Existing send_message calls default to the new confirmed behavior