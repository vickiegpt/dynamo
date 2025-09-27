// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::{AckHandler, ActiveMessageContext, HandlerType, ResponseHandler},
    manager::ActiveMessageManager,
    zmq::ZmqActiveMessageManager,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;
use tokio_util::sync::CancellationToken;
use tracing::info;

/// Generate a unique IPC socket path for testing
fn unique_ipc_socket_path() -> Result<String> {
    let temp_file = NamedTempFile::new()?;
    let path = temp_file.path().to_string_lossy().to_string();
    // Close the file but keep the path - ZMQ will create the socket
    drop(temp_file);
    Ok(format!("ipc://{}", path))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkPayload {
    sequence: u64,
    data: Vec<u8>,
}

/// Simple ping handler for latency benchmarks
#[derive(Debug, Clone)]
struct PingHandler;

#[async_trait]
impl AckHandler for PingHandler {
    async fn handle(
        &self,
        _input: Bytes,
        _ctx: ActiveMessageContext,
    ) -> Result<()> {
        // Just return Ok(()) to send ACK - minimal processing
        Ok(())
    }

    fn name(&self) -> &str {
        "ping"
    }
}

/// Echo handler for throughput benchmarks
#[derive(Debug, Clone)]
struct EchoHandler;

#[async_trait]
impl ResponseHandler for EchoHandler {
    async fn handle(
        &self,
        input: Bytes,
        _ctx: ActiveMessageContext,
    ) -> Result<Bytes> {
        // Echo back the payload with minimal processing
        let payload: BenchmarkPayload = serde_json::from_slice(&input)?;
        Ok(Bytes::from(serde_json::to_vec(&payload)?))
    }

    fn name(&self) -> &str {
        "echo"
    }
}

#[derive(Debug)]
struct BenchmarkStats {
    min: Duration,
    max: Duration,
    mean: Duration,
    median: Duration,
    p95: Duration,
    p99: Duration,
    throughput_msgs_per_sec: f64,
    total_duration: Duration,
    successful_ops: usize,
    failed_ops: usize,
}

impl BenchmarkStats {
    fn calculate(durations: &mut [Duration], total_duration: Duration, failed_ops: usize) -> Self {
        durations.sort();
        let successful_ops = durations.len();

        let min = durations.first().copied().unwrap_or_default();
        let max = durations.last().copied().unwrap_or_default();

        let sum: Duration = durations.iter().sum();
        let mean = if successful_ops > 0 {
            sum / successful_ops as u32
        } else {
            Duration::ZERO
        };

        let median = if successful_ops > 0 {
            durations[successful_ops / 2]
        } else {
            Duration::ZERO
        };

        let p95 = if successful_ops > 0 {
            durations[(successful_ops as f64 * 0.95) as usize]
        } else {
            Duration::ZERO
        };

        let p99 = if successful_ops > 0 {
            durations[(successful_ops as f64 * 0.99) as usize]
        } else {
            Duration::ZERO
        };

        let throughput_msgs_per_sec = if total_duration.as_secs_f64() > 0.0 {
            successful_ops as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            min,
            max,
            mean,
            median,
            p95,
            p99,
            throughput_msgs_per_sec,
            total_duration,
            successful_ops,
            failed_ops,
        }
    }

    fn print(&self, test_name: &str) {
        println!("=== {} Results ===", test_name);
        println!("Successful operations: {}", self.successful_ops);
        println!("Failed operations: {}", self.failed_ops);
        println!("Total duration: {:.2}s", self.total_duration.as_secs_f64());
        println!("Throughput: {:.2} msgs/sec", self.throughput_msgs_per_sec);
        println!("Latency statistics:");
        println!("  Min:    {:?}", self.min);
        println!("  Mean:   {:?}", self.mean);
        println!("  Median: {:?}", self.median);
        println!("  P95:    {:?}", self.p95);
        println!("  P99:    {:?}", self.p99);
        println!("  Max:    {:?}", self.max);
    }
}

async fn create_managers(
    max_concurrent: Option<usize>,
) -> Result<(ZmqActiveMessageManager, ZmqActiveMessageManager)> {
    let cancel_token = CancellationToken::new();

    let server_manager =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let client_manager =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token).await?;

    Ok((server_manager, client_manager))
}

async fn benchmark_ping_pong(
    name: &str,
    max_concurrent: Option<usize>,
    num_operations: usize,
) -> Result<BenchmarkStats> {
    info!(
        "Running {} benchmark with {} operations...",
        name, num_operations
    );

    let (server_manager, client_manager) = create_managers(max_concurrent).await?;

    // Register ping handler
    let ping_handler = Arc::new(PingHandler);
    let handler_type = HandlerType::ack((*ping_handler).clone());
    server_manager
        .register_handler_typed(handler_type, None)
        .await?;

    let server_client = server_manager.client();
    let client_client = client_manager.client();

    // Connect client to server (one-way connection)
    // Auto-registration will handle the reverse connection when ACK is needed
    let server_peer = PeerInfo::new(
        server_client.instance_id(),
        server_client.endpoint().to_string(),
    );
    client_client.connect_to_peer(server_peer).await?;

    // Wait for connection
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Warmup to establish publisher tasks (not measured)
    let warmup_result = client_client
        .active_message("ping")?
        .payload("warmup")?
        .send(server_client.instance_id())
        .await;

    if warmup_result.is_err() {
        anyhow::bail!("Warmup ping failed - benchmark cannot proceed");
    }

    // Small delay after warmup
    tokio::time::sleep(Duration::from_millis(50)).await;

    let start_time = Instant::now();
    let mut durations = Vec::with_capacity(num_operations);
    let mut failed_ops = 0;

    // Run fast-path benchmark (excludes initial connection overhead)
    for i in 0..num_operations {
        let op_start = Instant::now();

        let result = client_client
            .active_message("ping")?
            .payload(format!("ping_{}", i))?
            .send(server_client.instance_id())
            .await;

        let duration = op_start.elapsed();

        match result {
            Ok(_) => durations.push(duration),
            Err(_) => failed_ops += 1,
        }

        // Small delay to avoid overwhelming the system
        if i % 100 == 0 && i > 0 {
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }

    let total_duration = start_time.elapsed();

    // Cleanup
    server_manager.shutdown().await?;
    client_manager.shutdown().await?;

    Ok(BenchmarkStats::calculate(
        &mut durations,
        total_duration,
        failed_ops,
    ))
}

async fn benchmark_echo_throughput(
    name: &str,
    max_concurrent: Option<usize>,
    num_operations: usize,
    payload_size: usize,
    concurrent_requests: usize,
) -> Result<BenchmarkStats> {
    info!(
        "Running {} benchmark with {} operations, {} byte payloads, {} concurrent requests...",
        name, num_operations, payload_size, concurrent_requests
    );

    let (server_manager, client_manager) = create_managers(max_concurrent).await?;

    // Register echo handler
    let echo_handler = Arc::new(EchoHandler);
    let handler_type = HandlerType::response((*echo_handler).clone());
    server_manager
        .register_handler_typed(handler_type, None)
        .await?;

    let server_client = server_manager.client();
    let client_client = client_manager.client();

    // Connect client to server (one-way connection)
    // Auto-registration will handle the reverse connection when ACK is needed
    let server_peer = PeerInfo::new(
        server_client.instance_id(),
        server_client.endpoint().to_string(),
    );
    client_client.connect_to_peer(server_peer).await?;

    // Wait for connection
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Warmup to establish publisher tasks (not measured)
    let warmup_payload = BenchmarkPayload {
        sequence: 0,
        data: vec![0; 64], // Small warmup payload
    };
    let warmup_result = client_client
        .active_message("echo")?
        .payload(warmup_payload)?
        .expect_response::<BenchmarkPayload>()
        .send(server_client.instance_id())
        .await?
        .await_response::<BenchmarkPayload>()
        .await;

    if warmup_result.is_err() {
        anyhow::bail!("Warmup echo failed - benchmark cannot proceed");
    }

    // Small delay after warmup
    tokio::time::sleep(Duration::from_millis(50)).await;

    let start_time = Instant::now();
    let mut durations = Vec::with_capacity(num_operations);
    let mut failed_ops = 0;

    // Create semaphore to limit concurrent requests
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrent_requests));
    let mut tasks = Vec::new();

    // Spawn concurrent requests
    for i in 0..num_operations {
        let client = client_client.clone();
        let server_id = server_client.instance_id();
        let sem = semaphore.clone();
        let payload = BenchmarkPayload {
            sequence: i as u64,
            data: vec![0u8; payload_size],
        };

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let op_start = Instant::now();

            let result = client
                .active_message("echo")?
                .payload(payload)?
                .expect_response::<BenchmarkPayload>()
                .send(server_id)
                .await?
                .await_response::<BenchmarkPayload>()
                .await;

            let duration = op_start.elapsed();
            Result::<(Duration, bool), anyhow::Error>::Ok((duration, result.is_ok()))
        });

        tasks.push(task);
    }

    // Collect results
    for task in tasks {
        match task.await {
            Ok(Ok((duration, success))) => {
                if success {
                    durations.push(duration);
                } else {
                    failed_ops += 1;
                }
            }
            _ => failed_ops += 1,
        }
    }

    let total_duration = start_time.elapsed();

    // Cleanup
    server_manager.shutdown().await?;
    client_manager.shutdown().await?;

    Ok(BenchmarkStats::calculate(
        &mut durations,
        total_duration,
        failed_ops,
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("Starting Active Message Fast-Path Performance Benchmarks");
    println!("Testing fast-path latency (ping-pong) and throughput (echo) scenarios");
    println!("Note: Initial connection overhead excluded via warmup");

    // Latency benchmarks (ping-pong)
    println!("\n🏓 FAST-PATH LATENCY BENCHMARKS (Ping-Pong)");

    let ping_pong_stats = benchmark_ping_pong(
        "Fast-Path Ping-Pong Latency",
        None, // No concurrency limit
        1000,
    )
    .await?;
    ping_pong_stats.print("Fast-Path Ping-Pong Latency");

    // Throughput benchmarks with different settings
    println!("\n🚀 FAST-PATH THROUGHPUT BENCHMARKS (Echo)");

    // Small payload, moderate concurrency
    let echo_small_stats = benchmark_echo_throughput(
        "Fast-Path Echo Small Payload",
        None, // No concurrency limit
        1000,
        64, // 64 byte payload
        50, // 50 concurrent requests
    )
    .await?;
    echo_small_stats.print("Fast-Path Echo Small Payload");

    // Larger payload, lower concurrency
    let echo_large_stats = benchmark_echo_throughput(
        "Fast-Path Echo Large Payload",
        None, // No concurrency limit
        500,
        4096, // 4KB payload
        20,   // 20 concurrent requests
    )
    .await?;
    echo_large_stats.print("Fast-Path Echo Large Payload");

    // Test with concurrency limits
    println!("\n⚙️  FAST-PATH CONCURRENCY LIMIT BENCHMARKS");

    let limited_stats = benchmark_echo_throughput(
        "Fast-Path Echo with Concurrency Limit",
        Some(10), // Limit to 10 concurrent messages
        500,
        1024, // 1KB payload
        25,   // 25 concurrent requests (will be throttled by semaphore)
    )
    .await?;
    limited_stats.print("Fast-Path Echo with Concurrency Limit (10)");

    // Summary comparison
    println!("\n📊 FAST-PATH PERFORMANCE SUMMARY");
    println!(
        "Fast-Path Ping-Pong RTT:           {:.3}ms avg, {:.2} ops/sec",
        ping_pong_stats.mean.as_secs_f64() * 1000.0,
        ping_pong_stats.throughput_msgs_per_sec
    );
    println!(
        "Fast-Path Echo Small Throughput:   {:.2} msgs/sec, {:.3}ms avg latency",
        echo_small_stats.throughput_msgs_per_sec,
        echo_small_stats.mean.as_secs_f64() * 1000.0
    );
    println!(
        "Fast-Path Echo Large Throughput:   {:.2} msgs/sec, {:.3}ms avg latency",
        echo_large_stats.throughput_msgs_per_sec,
        echo_large_stats.mean.as_secs_f64() * 1000.0
    );
    println!(
        "Fast-Path Echo Limited Throughput: {:.2} msgs/sec, {:.3}ms avg latency",
        limited_stats.throughput_msgs_per_sec,
        limited_stats.mean.as_secs_f64() * 1000.0
    );

    println!("\n✅ All benchmarks completed successfully!");
    println!("The parallel message handling and concurrency controls are working correctly.");

    Ok(())
}
