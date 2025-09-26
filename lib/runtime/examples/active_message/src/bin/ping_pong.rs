// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::{ActiveMessage, AckHandler, HandlerType},
    manager::ActiveMessageManager,
    zmq::ZmqActiveMessageManager,
};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

/// Generate a unique IPC socket path for testing
fn unique_ipc_socket_path() -> Result<String> {
    let temp_file = NamedTempFile::new()?;
    let path = temp_file.path().to_string_lossy().to_string();
    // Close the file but keep the path - ZMQ will create the socket
    drop(temp_file);
    Ok(format!("ipc://{}", path))
}

/// Simple ping handler that responds with ACK
#[derive(Debug, Clone)]
struct PingHandler;

#[async_trait]
impl AckHandler for PingHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
    ) -> Result<()> {
        let payload = String::from_utf8(message.payload.to_vec())?;
        info!("Received ping: {}", payload);
        // Just return Ok(()) to send ACK
        Ok(())
    }

    fn name(&self) -> &str {
        "ping"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting ping-pong RTT measurement example");

    let cancel_token = CancellationToken::new();

    // Create two managers for ping-pong
    let server_manager = ZmqActiveMessageManager::new(
        unique_ipc_socket_path()?,
        cancel_token.clone(),
    ).await?;

    let client_manager = ZmqActiveMessageManager::new(
        unique_ipc_socket_path()?,
        cancel_token.clone(),
    ).await?;

    // Register ping handler on server
    let ping_handler = Arc::new(PingHandler);
    let handler_type = HandlerType::ack((*ping_handler).clone());
    server_manager.register_handler_typed(handler_type, None).await?;

    let server_client = server_manager.zmq_client();
    let client_client = client_manager.zmq_client();

    info!("Server listening on: {}", server_client.endpoint());
    info!("Client endpoint: {}", client_client.endpoint());

    // Connect client to server
    let server_peer = PeerInfo::new(
        server_client.instance_id(),
        server_client.endpoint().to_string(),
    );

    client_client.connect_to_peer(server_peer).await?;

    // Wait for connection to establish
    tokio::time::sleep(Duration::from_millis(100)).await;

    info!("Starting ping-pong measurements...");

    // Perform multiple ping-pong rounds for RTT measurement
    let mut rtts = Vec::new();
    let rounds = 10;

    for i in 1..=rounds {
        let start = Instant::now();

        let message = format!("ping_{}", i);

        // Send ping and wait for ACK
        let result = client_client
            .message("ping")?
            .payload(&message)?
            .send_and_confirm(server_client.instance_id())
            .await;

        match result {
            Ok(_) => {
                let rtt = start.elapsed();
                rtts.push(rtt);
                info!("Round {}: RTT = {:?}", i, rtt);
            }
            Err(e) => {
                warn!("Round {} failed: {}", i, e);
            }
        }

        // Small delay between rounds
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Calculate statistics
    if !rtts.is_empty() {
        let total: Duration = rtts.iter().sum();
        let avg = total / rtts.len() as u32;
        let min = rtts.iter().min().unwrap();
        let max = rtts.iter().max().unwrap();

        info!("=== RTT Statistics ===");
        info!("Rounds completed: {}/{}", rtts.len(), rounds);
        info!("Average RTT: {:?}", avg);
        info!("Min RTT: {:?}", min);
        info!("Max RTT: {:?}", max);

        // Calculate standard deviation
        let variance: f64 = rtts.iter()
            .map(|rtt| {
                let diff = rtt.as_nanos() as f64 - avg.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>() / rtts.len() as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);
        info!("Std deviation: {:?}", std_dev);
    } else {
        warn!("No successful ping-pong rounds completed");
    }

    info!("Shutting down...");
    server_manager.shutdown().await?;
    client_manager.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}