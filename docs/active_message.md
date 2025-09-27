<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Active Message System

## Executive Summary (PR Description)

The Active Message system provides high-performance, type-safe distributed messaging for ML workloads in the Dynamo runtime. It replaces raw RPC patterns with a structured approach to inter-service communication, achieving sub-millisecond latency through innovative publisher task pools.

**Key Features:**
- **Type-safe handlers**: Three message patterns (NoReturn, Ack, Response) with compile-time guarantees
- **Auto-registration**: Automatic bidirectional connection setup without manual configuration
- **Sub-millisecond performance**: 99.45% latency reduction (23ms → 126μs average RTT)
- **Leader-worker cohorts**: Built-in patterns for distributed ML coordination
- **ZeroMQ transport**: High-performance messaging with IPC and TCP support

**Performance Achievements:**
- Fast-path ping-pong: 126μs average RTT, 7,313 ops/sec
- Echo throughput: 43,149 msgs/sec (small payloads)
- Zero message failures with consistent microsecond latency
- Publisher task pool eliminates connection overhead

**Migration:** Drop-in replacement for existing RPC patterns with improved type safety and performance.

---

## Overview

### What is Active Message?

Active Message is a distributed computing pattern where messages carry both data and the computation to be performed on that data. In the Dynamo runtime, this translates to type-safe message handlers that automatically manage communication, serialization, and error handling.

### Why Active Messages for ML Workloads?

Modern ML serving requires:
- **Low latency**: Microsecond-level response times for real-time inference
- **High throughput**: Thousands of concurrent requests per second
- **Type safety**: Compile-time guarantees for distributed message handling
- **Scalability**: Efficient patterns for leader-worker coordination
- **Reliability**: Automatic error handling and connection management

Traditional RPC approaches introduce overhead and require manual connection management. Active Messages provide a structured, high-performance alternative.

## Architecture

### Core Components

```rust
// Message flow: Client -> Manager -> Handler -> Response
Client ──message()──> Manager ──dispatch──> Handler ──result──> Client
   │                     │                     │
   └─── Auto-registration ──────────────────────┘
```

#### 1. Handlers - Three Message Patterns

**NoReturn Handler**: Fire-and-forget messages
```rust
#[async_trait]
impl NoReturnHandler for LogHandler {
    async fn handle(&self, message: ActiveMessage, _client: &dyn ActiveMessageClient) {
        let log_entry: LogEntry = message.deserialize()?;
        println!("Log: {}", log_entry.message);
    }

    fn name(&self) -> &str { "log" }
}
```

**Ack Handler**: Simple success/failure acknowledgment
```rust
#[async_trait]
impl AckHandler for ValidateHandler {
    async fn handle(&self, message: ActiveMessage, _client: &dyn ActiveMessageClient) -> Result<()> {
        let data: ValidationRequest = message.deserialize()?;
        if data.is_valid() {
            Ok(()) // Sends ACK
        } else {
            Err(anyhow::anyhow!("Validation failed")) // Sends NACK
        }
    }

    fn name(&self) -> &str { "validate" }
}
```

**Response Handler**: Full request-response with typed data
```rust
#[async_trait]
impl ResponseHandler for ComputeHandler {
    type Response = ComputeResult;

    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient
    ) -> Result<Self::Response> {
        let request: ComputeRequest = message.deserialize()?;
        let result = perform_computation(request).await?;
        Ok(result)
    }

    fn name(&self) -> &str { "compute" }
}
```

#### 2. Client - Message Builder Pattern

```rust
// Fire-and-forget
client.message("log")?.payload(log_data)?.send_detached(target_id).await?;

// Wait for ACK
client.message("validate")?.payload(data)?.send_and_confirm(target_id).await?;

// Request-response
let result: ComputeResult = client
    .message("compute")?
    .payload(request)?
    .expect_response::<ComputeResult>()
    .send(target_id)
    .await?
    .await_response()
    .await?;
```

#### 3. Manager - Registration and Dispatch

```rust
// Register handlers
let handler = HandlerType::response(ComputeHandler::new());
manager.register_handler_typed(handler, None).await?;

// Auto-discovery and message routing
manager.start().await?;
```

#### 4. Transport Layer - Publisher Task Pool

The ZMQ transport uses a high-performance publisher task pool:

```rust
// Per-endpoint publisher tasks with MPSC channels
struct ClientState {
    publisher_channels: HashMap<String, mpsc::UnboundedSender<ActiveMessage>>,
    publisher_tasks: HashMap<String, JoinHandle<()>>,
}

// Non-blocking sends via channels
async fn send_raw(&self, endpoint: &str, message: &ActiveMessage) -> Result<()> {
    let sender = self.get_or_create_publisher(endpoint).await?;
    sender.send(message.clone())?; // Non-blocking
    Ok(())
}
```

### Auto-Registration

Eliminates manual bidirectional connection setup:

```rust
// Client sends message with endpoint metadata
metadata["_sender_endpoint"] = client.endpoint();

// Server auto-registers the client for response delivery
if let Some(endpoint) = sender_endpoint {
    state.peers.insert(sender_id, PeerInfo::new(sender_id, endpoint));
}
```

Only includes endpoint metadata when:
1. Response expected AND
2. No existing return connection

## Performance

### Benchmark Results

**Fast-Path Ping-Pong Latency:**
```
Average fast-path RTT: 126μs
Min fast-path RTT: 96μs
P95 fast-path RTT: 142μs
Throughput: 7,313 ops/sec
```

**Echo Throughput:**
```
Small payload (64B): 43,149 msgs/sec, 1.1ms avg latency
Large payload (4KB): 9,766 msgs/sec, 1.99ms avg latency
Concurrency limited: 25,063 msgs/sec, 0.95ms avg latency
```

### Performance Features

1. **Publisher Task Pool**: Eliminates per-message connection overhead
2. **Shared ZMQ Context**: Efficient resource utilization
3. **Slow Joiner Mitigation**: 2ms delay only on first connection per endpoint
4. **Fast-Path Optimization**: Sub-millisecond performance after warmup

### Before vs After

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Average RTT | 23ms | 126μs | 99.45% |
| Connection Setup | 50ms delay per message | 2ms once per endpoint | 96% reduction |
| Failed Operations | Frequent hangs | Zero failures | 100% reliability |

## Usage Guide

### Getting Started

1. **Create a Manager**
```rust
let cancel_token = CancellationToken::new();
let manager = ZmqActiveMessageManager::new(
    "tcp://0.0.0.0:5555".to_string(),
    cancel_token
).await?;
```

2. **Register Handlers**
```rust
let handler = HandlerType::response(MyHandler::new());
manager.register_handler_typed(handler, None).await?;
```

3. **Connect and Send Messages**
```rust
let client = manager.zmq_client();
let peer = PeerInfo::new(target_id, "tcp://target:5556".to_string());
client.connect_to_peer(peer).await?;

let response = client
    .message("my_handler")?
    .payload(request_data)?
    .expect_response::<ResponseType>()
    .send(target_id)
    .await?
    .await_response()
    .await?;
```

### Handler Examples

#### Simple Echo Handler
```rust
#[derive(Debug, Clone)]
struct EchoHandler;

#[async_trait]
impl ResponseHandler for EchoHandler {
    type Response = String;

    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
    ) -> Result<Self::Response> {
        let input: String = message.deserialize()?;
        Ok(format!("Echo: {}", input))
    }

    fn name(&self) -> &str { "echo" }
}
```

#### ML Inference Handler
```rust
#[derive(Debug, Clone)]
struct InferenceHandler {
    model: Arc<Model>,
}

#[async_trait]
impl ResponseHandler for InferenceHandler {
    type Response = InferenceResult;

    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
    ) -> Result<Self::Response> {
        let request: InferenceRequest = message.deserialize()?;
        let result = self.model.infer(request.input).await?;
        Ok(InferenceResult {
            output: result,
            model_id: self.model.id(),
            latency: start.elapsed(),
        })
    }

    fn name(&self) -> &str { "inference" }
}
```

## Advanced Features

### Leader-Worker Cohorts

For distributed ML workloads requiring coordinated worker management:

```rust
// Leader setup
let cohort_config = LeaderWorkerCohortConfigBuilder::default()
    .leader_instance(leader_client.instance_id())
    .client(leader_client.clone())
    .cohort_type(CohortType::FixedSize(num_workers))
    .failure_policy(CohortFailurePolicy::TerminateAll)
    .build()?;

let cohort = Arc::new(LeaderWorkerCohort::from_config(cohort_config));

// Parallel operations
let results: Vec<WorkResponse> = cohort
    .par_map("work", |rank, worker_id| async move {
        Ok(WorkRequest { rank, task: format!("task-{}", rank) })
    }, timeout)
    .await?;
```

### Concurrency Control

```rust
// Limit concurrent messages per handler
let manager = ZmqActiveMessageManager::builder()
    .max_concurrent_messages(100)
    .build(endpoint, cancel_token)
    .await?;
```

### Error Handling

```rust
// Automatic retries and timeouts
let result = client
    .message("unreliable_service")?
    .payload(request)?
    .timeout(Duration::from_secs(5))
    .expect_response::<Response>()
    .send(target_id)
    .await?
    .await_response()
    .await;

match result {
    Ok(response) => println!("Success: {:?}", response),
    Err(e) => println!("Service unavailable: {}", e),
}
```

## Migration Guide

### From Raw RPC

**Before (Manual RPC):**
```rust
// Manual connection management
let mut connection = tcp_connect(endpoint).await?;
let request_bytes = serialize_request(&request)?;
connection.write_all(&request_bytes).await?;
let response_bytes = read_response(&mut connection).await?;
let response: Response = deserialize_response(&response_bytes)?;
```

**After (Active Message):**
```rust
// Automatic connection and serialization
let response: Response = client
    .message("handler_name")?
    .payload(request)?
    .expect_response::<Response>()
    .send(target_id)
    .await?
    .await_response()
    .await?;
```

### Converting Existing Services

1. **Identify Message Patterns**:
   - Fire-and-forget → `NoReturnHandler`
   - Success/failure → `AckHandler`
   - Request/response → `ResponseHandler`

2. **Implement Handler Trait**:
```rust
// Old service function
async fn old_compute_service(request: ComputeRequest) -> Result<ComputeResponse> {
    // business logic
}

// New Active Message handler
#[async_trait]
impl ResponseHandler for ComputeHandler {
    type Response = ComputeResponse;

    async fn handle(&self, message: ActiveMessage, _client: &dyn ActiveMessageClient) -> Result<Self::Response> {
        let request: ComputeRequest = message.deserialize()?;
        old_compute_service(request).await  // Reuse existing logic
    }

    fn name(&self) -> &str { "compute" }
}
```

3. **Update Client Code**:
```rust
// Replace direct function calls with message sends
let response = client.message("compute")?.payload(request)?.expect_response().send(target).await?.await_response().await?;
```

## API Reference

### Core Traits

```rust
// Handler patterns
pub trait NoReturnHandler: Send + Sync + Debug {
    async fn handle(&self, message: ActiveMessage, client: &dyn ActiveMessageClient);
    fn name(&self) -> &str;
}

pub trait AckHandler: Send + Sync + Debug {
    async fn handle(&self, message: ActiveMessage, client: &dyn ActiveMessageClient) -> Result<()>;
    fn name(&self) -> &str;
}

pub trait ResponseHandler: Send + Sync + Debug {
    type Response: Serialize + for<'de> Deserialize<'de>;
    async fn handle(&self, message: ActiveMessage, client: &dyn ActiveMessageClient) -> Result<Self::Response>;
    fn name(&self) -> &str;
}

// Client interface
pub trait ActiveMessageClient: Send + Sync {
    fn instance_id(&self) -> InstanceId;
    fn endpoint(&self) -> &str;
    async fn send_message(&self, target: InstanceId, handler: &str, payload: Bytes) -> Result<()>;
    async fn connect_to_peer(&self, peer: PeerInfo) -> Result<()>;
    // ... additional methods
}
```

### Message Builder

```rust
impl MessageBuilder {
    pub fn payload<T: Serialize>(mut self, payload: T) -> Result<Self>;
    pub fn timeout(mut self, timeout: Duration) -> Self;
    pub fn expect_response<R>(self) -> MessageBuilder<ExpectResponse<R>>;
    pub async fn send(self, target: InstanceId) -> Result<MessageStatus>;
    pub async fn send_and_confirm(self, target: InstanceId) -> Result<MessageStatus>;
    pub async fn send_detached(self, target: InstanceId) -> Result<MessageStatus>;
}
```

### Registration

```rust
impl ZmqActiveMessageManager {
    pub async fn register_handler_typed(&self, handler: HandlerType, config: Option<HandlerConfig>) -> Result<()>;
    pub async fn register_no_return_closure<F, Fut>(&self, name: &str, handler: F) -> Result<()>;
    pub async fn register_ack_closure<F, Fut>(&self, name: &str, handler: F) -> Result<()>;
    pub async fn register_response_closure<F, Fut, T>(&self, name: &str, handler: F) -> Result<()>;
}
```

## Best Practices

1. **Handler Naming**: Use descriptive, unique names (`inference`, `validate`, not `handle`)
2. **Error Handling**: Return detailed errors for debugging, use structured error types
3. **Timeouts**: Set appropriate timeouts based on expected processing time
4. **Serialization**: Use efficient formats (bincode for internal, JSON for debugging)
5. **Resource Management**: Call shutdown() on managers to clean up properly
6. **Testing**: Use the built-in examples for integration testing patterns

## Examples

See the complete examples in [`lib/runtime/examples/active_message/src/bin/`](../lib/runtime/examples/active_message/src/bin/):

- **`hello_world.rs`**: Basic request-response pattern
- **`ping_pong.rs`**: Latency measurement and ACK handling
- **`benchmarks.rs`**: Performance testing and throughput measurement
- **`leader.rs` / `worker.rs`**: Leader-worker cohort coordination
- **`cohort_parallel.rs`**: Advanced parallel operations

Run examples:
```bash
# Basic hello world
cargo run --bin hello_world

# Performance benchmarks
cargo run --bin benchmarks --release

# Leader-worker coordination
cargo run --bin leader
cargo run --bin worker  # In separate terminal
```