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
client.message("validate")?.payload(data)?.send(target_id).await?;

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

#### 4. Transport Layer

Uses high-performance ZeroMQ transport with automatic connection management and sub-millisecond latency.

### Auto-Registration

Eliminates manual bidirectional connection setup. When a client sends a message requiring a response, the server automatically learns the client's endpoint and establishes a return connection. No manual configuration needed.

```rust
// Client connects to server (one-way)
client.connect_to_peer(server_peer).await?;

// Send message - server automatically registers client for ACK/response delivery
let result = client.message("compute")?.payload(request)?.send(server_id).await?;
// No manual setup of return connection required!
```

## Performance

### Benchmark Results

*Note: Performance numbers shown are for ZMQ IPC (same-host) transport.*

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

1. **Sub-millisecond latency**: Consistent microsecond response times
2. **High throughput**: 40K+ messages per second for small payloads
3. **Zero message failures**: Reliable delivery with proper error handling
4. **Automatic connection management**: No manual setup overhead

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
// Leader setup - simplified API
let cohort = Arc::new(LeaderWorkerCohort::new(
    leader_client.clone(),
    CohortType::FixedSize(num_workers),
));

// Parallel operations
let results: Vec<WorkResponse> = cohort
    .par_map("work", |rank, worker_id| async move {
        Ok(WorkRequest { rank, task: format!("task-{}", rank) })
    }, timeout)
    .await?;
```

### Concurrency Control

```rust
// Limit concurrent messages per handler (default: unbounded)
let handler_config = HandlerConfig::default()
    .with_max_concurrent_messages(100);

manager.register_handler_typed(handler, Some(handler_config)).await?;
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
    pub async fn send(self, target: InstanceId) -> Result<MessageStatus>;
    pub async fn send_detached(self, target: InstanceId) -> Result<MessageStatus>;
}
```

### Registration

#### Typed Handler Registration

Use these ergonomic typed methods with automatic serialization:

```rust
impl ZmqActiveMessageManager {
    // Request/Response handlers - automatic serialization of request and response types
    pub async fn register_unary<Req, Res, F, Fut>(
        &self,
        name: impl Into<String>,
        closure: F
    ) -> Result<()>
    where
        Req: DeserializeOwned + Send + 'static,
        Res: Serialize + Send + 'static,
        F: Fn(Req, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Res>> + Send + 'static;

    // Void handlers - no response, automatic deserialization of input
    pub async fn register_void<Input, F, Fut>(
        &self,
        name: impl Into<String>,
        closure: F,
    ) -> Result<()>
    where
        Input: DeserializeOwned + Send + 'static,
        F: Fn(Input, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static;

    // Acknowledgment handlers - returns success/failure, automatic deserialization
    pub async fn register_typed_ack<Input, F, Fut>(
        &self,
        name: impl Into<String>,
        closure: F,
    ) -> Result<()>
    where
        Input: DeserializeOwned + Send + 'static,
        F: Fn(Input, ActiveMessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static;
}
```

#### Examples

**Request/Response Pattern:**
```rust
#[derive(Deserialize)]
struct ComputeRequest { x: i32, y: i32 }

#[derive(Serialize)]
struct ComputeResponse { result: i32 }

// Clean, typed handler with automatic serialization
manager.register_unary("compute", |req: ComputeRequest, _ctx| async move {
    Ok(ComputeResponse { result: req.x + req.y })
}).await?;
```

**Void Pattern (Fire and Forget):**
```rust
manager.register_void("log", |message: String, _ctx| async move {
    println!("Received: {}", message);
}).await?;
```

**Acknowledgment Pattern:**
```rust
manager.register_typed_ack("validate", |data: ValidationRequest, _ctx| async move {
    if data.is_valid() {
        Ok(()) // Success - ACK sent
    } else {
        anyhow::bail!("Invalid data") // Failure - NACK sent
    }
}).await?;
```

#### Advanced Registration

For advanced use cases requiring custom serialization or trait-based handlers:

```rust
impl ZmqActiveMessageManager {
    pub async fn register_handler_typed(&self, handler: HandlerType, config: Option<HandlerConfig>) -> Result<()>;
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
# Navigate to examples directory
cd lib/runtime/examples/active_message

# Basic hello world
cargo run --bin hello_world

# Performance benchmarks
cargo run --bin benchmarks --release

# Leader-worker coordination
cargo run --bin leader
cargo run --bin worker  # In separate terminal
```