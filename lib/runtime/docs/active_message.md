# ActiveMessage System

The ActiveMessage system provides a distributed messaging framework built on top of ZeroMQ, designed for leader-worker architectures with dynamic handler registration, ACK tracking, and service discovery.

## Architecture Overview

The system uses a **Publisher-Subscriber** pattern where:
- Each service binds a **SUB socket** to receive messages
- Services create ephemeral **PUB sockets** to send messages (per-use pattern)
- Transport is automatically detected (IPC for same-host, TCP for cross-host)
- Built-in system handlers provide core functionality (prefixed with `_`)

## Core Components

### ActiveMessage

The core message structure that flows through the system:

```rust
use dynamo_runtime::active_message::handler::{ActiveMessage, InstanceId};
use bytes::Bytes;

pub struct ActiveMessage {
    pub message_id: Uuid,           // Unique message identifier
    pub handler_name: String,       // Target handler name
    pub sender_instance: InstanceId, // Sender's instance ID (UUID)
    pub payload: Bytes,             // Message payload (JSON recommended)
    pub metadata: serde_json::Value, // Additional metadata
}
```

Create messages using:
```rust
let message = ActiveMessage::new("my_handler", Bytes::from("payload"))
    .with_metadata(serde_json::json!({"priority": "high"}));
```

### ActiveMessageHandler

Trait for implementing message handlers with automatic JSON schema validation:

```rust
use dynamo_runtime::active_message::{
    handler::{ActiveMessage, ActiveMessageHandler},
    client::ActiveMessageClient,
};
use async_trait::async_trait;

#[derive(Debug)]
pub struct MyHandler;

#[async_trait]
impl ActiveMessageHandler for MyHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        client: Arc<dyn ActiveMessageClient>,
    ) -> Result<()> {
        // Parse payload
        let request: MyRequest = serde_json::from_slice(&message.payload)?;

        // Process message
        println!("Received: {:?}", request);

        // Optionally send response
        client.send_message(
            message.sender_instance,
            "response_handler",
            Bytes::from(serde_json::to_vec(&response)?),
        ).await?;

        Ok(())
    }

    fn name(&self) -> &str {
        "my_handler"
    }

    // Optional: JSON schema validation
    fn schema(&self) -> Option<&serde_json::Value> {
        static SCHEMA: once_cell::sync::Lazy<serde_json::Value> =
            once_cell::sync::Lazy::new(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "number" },
                        "name": { "type": "string" }
                    },
                    "required": ["id", "name"]
                })
            });
        Some(&SCHEMA)
    }
}
```

### ActiveMessageClient

Interface for sending messages and managing peer connections:

```rust
use dynamo_runtime::active_message::client::{ActiveMessageClient, PeerInfo};

// Key methods:
async fn send_message(&self, target: InstanceId, handler: &str, payload: Bytes) -> Result<()>;
async fn broadcast_message(&self, handler: &str, payload: Bytes) -> Result<()>;
async fn connect_to_peer(&self, peer: PeerInfo) -> Result<()>;
async fn await_handler(&self, instance_id: InstanceId, handler: &str, timeout: Option<Duration>) -> Result<()>;
```

### ActiveMessageManager

Interface for managing handlers and service lifecycle:

```rust
use dynamo_runtime::active_message::manager::{ActiveMessageManager, HandlerConfig};

// Key methods:
async fn register_handler(&self, handler: Arc<dyn ActiveMessageHandler>, config: Option<HandlerConfig>) -> Result<()>;
async fn deregister_handler(&self, name: &str) -> Result<()>;
fn client(&self) -> Arc<dyn ActiveMessageClient>;
fn handler_events(&self) -> broadcast::Receiver<HandlerEvent>;
```

## ZMQ Implementation

### ZmqActiveMessageManager

The main entry point for creating an ActiveMessage service:

```rust
use dynamo_runtime::active_message::zmq::ZmqActiveMessageManager;
use tokio_util::sync::CancellationToken;

let cancel_token = CancellationToken::new();

// Create manager (binds SUB socket)
let manager = ZmqActiveMessageManager::new(
    "tcp://0.0.0.0:5555".to_string(), // Endpoint to bind
    cancel_token.clone()
).await?;

// Register handlers
let handler = Arc::new(MyHandler);
manager.register_handler(handler, None).await?;

// Get client for sending messages
let client = manager.client();

// Shutdown cleanly
manager.shutdown().await?;
```

### ZmqActiveMessageClient

Provides message sending with per-use PUB socket pattern:

```rust
// Connect to a peer
let peer = PeerInfo::new(peer_instance_id, "tcp://192.168.1.100:5555");
client.connect_to_peer(peer).await?;

// Send message
client.send_message(
    peer_instance_id,
    "compute_task",
    Bytes::from(serde_json::to_vec(&request)?),
).await?;
```

### LeaderWorkerCohort

Helper for managing groups of workers in leader-worker patterns:

```rust
use dynamo_runtime::active_message::zmq::LeaderWorkerCohort;

let cohort = LeaderWorkerCohort::new(
    leader_instance_id,
    vec![worker1_id, worker2_id], // Worker instance IDs
    manager.zmq_client(), // Need ZMQ-specific client
);

// Wait for all workers to register a handler
cohort.await_handler_on_all_workers("compute", Some(Duration::from_secs(30))).await?;

// Broadcast to all workers
cohort.broadcast_to_workers("compute", payload).await?;

// Broadcast with ACK tracking
cohort.broadcast_to_workers_with_acks("compute", payload, Duration::from_secs(5)).await?;
```

## Built-in System Handlers

The system provides several built-in handlers (all prefixed with `_`):

### `_ack` - Acknowledgment Handler
Handles ACK responses for tracked messages. Used internally by the ACK system.

### `_register_service` - Service Registration
Allows workers to register themselves with a leader:
```json
{
    "instance_id": "uuid-string",
    "endpoint": "tcp://192.168.1.100:5556"
}
```

### `_list_handlers` - Handler Discovery
Returns list of available handlers on a service (for debugging).

### `_wait_for_handler` - Handler Availability
Blocks until a specific handler becomes available:
```json
{
    "handler_name": "compute",
    "timeout_ms": 30000
}
```

### `_health_check` - Health Check
Simple health check endpoint for service monitoring.

## Usage Patterns

### Basic Service Setup

```rust
use dynamo_runtime::active_message::{
    manager::ActiveMessageManager,
    zmq::ZmqActiveMessageManager,
};

// 1. Create manager
let manager = ZmqActiveMessageManager::new(
    "tcp://0.0.0.0:0".to_string(), // 0 = random port
    cancel_token.clone()
).await?;

// 2. Register handlers
let handler = Arc::new(ComputeHandler);
manager.register_handler(handler, None).await?;

// 3. Connect to peers
let client = manager.client();
let peer = PeerInfo::new(peer_id, peer_endpoint);
client.connect_to_peer(peer).await?;

// 4. Send messages
client.send_message(peer_id, "compute", payload).await?;
```

### Leader-Worker Pattern

**Leader:**
```rust
// Leader binds to known port
let manager = ZmqActiveMessageManager::new("tcp://0.0.0.0:5555".to_string(), cancel_token).await?;

// Workers will register themselves via _register_service
let cohort = LeaderWorkerCohort::new(
    manager.client().instance_id(),
    vec![], // Initially empty, workers register dynamically
    manager.zmq_client(),
);

// Broadcast work to all registered workers
cohort.broadcast_to_workers("process_task", task_data).await?;
```

**Worker:**
```rust
// Worker binds to random port
let manager = ZmqActiveMessageManager::new("tcp://0.0.0.0:0".to_string(), cancel_token).await?;

// Register task handler
let handler = Arc::new(TaskHandler);
manager.register_handler(handler, None).await?;

// Connect to leader
let leader_peer = PeerInfo::new(leader_id, "tcp://leader-host:5555");
manager.client().connect_to_peer(leader_peer).await?;

// Register with leader
let registration = serde_json::json!({
    "instance_id": manager.client().instance_id().to_string(),
    "endpoint": manager.client().endpoint(),
});
manager.client().send_message(
    leader_id,
    "_register_service",
    Bytes::from(serde_json::to_vec(&registration)?),
).await?;
```

### ACK Tracking Pattern

The ACK system requires **registration before sending** the message that triggers the ACK:

```rust
use uuid::Uuid;

// 1. Register ACK expectation BEFORE sending
let ack_id = Uuid::new_v4();
let ack_receiver = zmq_client.register_ack(ack_id, Duration::from_secs(10)).await?;

// 2. Send message with ACK ID embedded in payload
let mut request = serde_json::json!({
    "operation": "compute",
    "data": [1, 2, 3],
});
request["_ack_id"] = serde_json::Value::String(ack_id.to_string());

client.send_message(worker_id, "compute_task", Bytes::from(serde_json::to_vec(&request)?)).await?;

// 3. Wait for ACK
match ack_receiver.await {
    Ok(()) => println!("ACK received"),
    Err(_) => println!("ACK timeout"),
}
```

The receiver should send an ACK back:
```rust
// In handler implementation
if let Some(ack_id) = payload.get("_ack_id").and_then(|v| v.as_str()) {
    let ack_payload = serde_json::json!({ "ack_id": ack_id });
    client.send_message(
        message.sender_instance,
        "_ack",
        Bytes::from(serde_json::to_vec(&ack_payload)?),
    ).await?;
}
```

### Handler Configuration

Handlers can be configured with custom TaskTrackers for execution control:

```rust
use dynamo_runtime::{
    active_message::manager::HandlerConfig,
    utils::tasks::tracker::{TaskTracker, UnlimitedScheduler, LogOnlyPolicy}
};

let task_tracker = TaskTracker::builder()
    .scheduler(UnlimitedScheduler::new())
    .error_policy(LogOnlyPolicy::new())
    .build()?;

let config = HandlerConfig::default().with_task_tracker(task_tracker);
manager.register_handler(handler, Some(config)).await?;
```

## Error Handling

- All operations return `anyhow::Result<T>`
- Message deserialization errors are handled gracefully
- Network failures are logged and operations can be retried
- Unmatched ACKs are logged as errors
- Handler panics are caught by TaskTracker

## Transport Detection

The system automatically optimizes transport:
- **IPC sockets** (`ipc://`) for same-host communication
- **TCP sockets** (`tcp://`) for cross-host communication
- Detection is based on endpoint host comparison

## Lifecycle Management

```rust
// Proper shutdown sequence
manager.shutdown().await?; // Joins all handler task trackers
cancel_token.cancel();     // Stops background tasks
```

## Thread Safety

All components are thread-safe:
- `ActiveMessageHandler` requires `Send + Sync`
- Clients and managers can be shared across async tasks
- Message handling is concurrent per handler via TaskTracker

This system is designed to be robust, scalable, and easy to use for building distributed applications with dynamic service discovery and reliable message delivery.