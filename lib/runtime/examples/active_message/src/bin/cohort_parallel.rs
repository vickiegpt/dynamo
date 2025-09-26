// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::{ActiveMessage, ResponseHandler, AckHandler, HandlerType},
    manager::ActiveMessageManager,
    zmq::{
        cohort::{LeaderWorkerCohort, LeaderWorkerCohortConfigBuilder, CohortType, CohortFailurePolicy},
        ZmqActiveMessageManager,
    },
};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::Duration;
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
struct WorkRequest {
    rank: usize,
    workload: String,
    multiplier: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkResponse {
    rank: usize,
    result: String,
    processed_length: usize,
}

/// Work handler that processes requests and returns responses
#[derive(Debug, Clone)]
struct WorkHandler {
    worker_rank: usize,
}

impl WorkHandler {
    fn new(worker_rank: usize) -> Self {
        Self { worker_rank }
    }
}

#[async_trait]
impl ResponseHandler for WorkHandler {
    type Response = WorkResponse;

    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
    ) -> Result<Self::Response> {
        let request: WorkRequest = message.deserialize()?;

        info!("Worker {} processing work: {} (multiplier: {})",
              self.worker_rank, request.workload, request.multiplier);

        // Simulate some work by repeating the string
        let result = request.workload.repeat(request.multiplier as usize);
        let processed_length = result.len();

        info!("Worker {} completed work, result length: {}",
              self.worker_rank, processed_length);

        Ok(WorkResponse {
            rank: self.worker_rank,
            result,
            processed_length,
        })
    }

    fn name(&self) -> &str {
        "work"
    }
}

/// Simple ping handler for cohort ping-pong test
#[derive(Debug, Clone)]
struct CohortPingHandler {
    worker_rank: usize,
}

impl CohortPingHandler {
    fn new(worker_rank: usize) -> Self {
        Self { worker_rank }
    }
}

#[async_trait]
impl AckHandler for CohortPingHandler {
    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
    ) -> Result<()> {
        let ping_msg = String::from_utf8(message.payload.to_vec())?;
        info!("Worker {} received ping: {}", self.worker_rank, ping_msg);
        // Just return Ok(()) to send ACK
        Ok(())
    }

    fn name(&self) -> &str {
        "cohort_ping"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting cohort parallel operations example");

    let cancel_token = CancellationToken::new();

    // Create leader manager
    let leader_manager = ZmqActiveMessageManager::new(
        unique_ipc_socket_path()?,
        cancel_token.clone(),
    ).await?;

    let leader_client = leader_manager.zmq_client();
    info!("Leader listening on: {}", leader_client.endpoint());

    // Create worker managers (3 workers with ranks 0, 1, 2)
    let mut worker_managers = Vec::new();
    let mut worker_clients = Vec::new();

    for rank in 0..3 {
        let worker_manager = ZmqActiveMessageManager::new(
            unique_ipc_socket_path()?,
            cancel_token.clone(),
        ).await?;

        // Register handlers on each worker
        let work_handler = Arc::new(WorkHandler::new(rank));
        let work_handler_type = HandlerType::response((*work_handler).clone());
        worker_manager.register_handler_typed(work_handler_type, None).await?;

        let ping_handler = Arc::new(CohortPingHandler::new(rank));
        let ping_handler_type = HandlerType::ack((*ping_handler).clone());
        worker_manager.register_handler_typed(ping_handler_type, None).await?;

        let worker_client = worker_manager.zmq_client();
        info!("Worker {} listening on: {}", rank, worker_client.endpoint());

        worker_managers.push(worker_manager);
        worker_clients.push(worker_client);
    }

    // Create cohort configuration
    let cohort_config = LeaderWorkerCohortConfigBuilder::default()
        .leader_instance(leader_client.instance_id())
        .client(leader_client.clone())
        .cohort_type(CohortType::FixedSize(3))
        .failure_policy(CohortFailurePolicy::TerminateAll)
        .build()?;

    let cohort = Arc::new(LeaderWorkerCohort::from_config(cohort_config));

    // Connect workers to leader
    let leader_peer = PeerInfo::new(
        leader_client.instance_id(),
        leader_client.endpoint().to_string(),
    );

    for worker_client in &worker_clients {
        worker_client.connect_to_peer(leader_peer.clone()).await?;
    }

    // Connect leader to workers and add them to cohort
    for (rank, worker_client) in worker_clients.iter().enumerate() {
        let worker_peer = PeerInfo::new(
            worker_client.instance_id(),
            worker_client.endpoint().to_string(),
        );
        leader_client.connect_to_peer(worker_peer).await?;

        // Add worker to cohort with its rank
        cohort.add_worker(worker_client.instance_id(), Some(rank)).await?;
    }

    // Wait for connections to establish
    tokio::time::sleep(Duration::from_millis(500)).await;

    info!("=== Cohort Parallel Ping-Pong Test ===");

    // Test 1: Parallel ping-pong with ACKs
    let ping_results = cohort.par_broadcast_acks(
        "cohort_ping",
        "Hello from leader!",
        Duration::from_secs(5),
    ).await?;

    info!("Ping-pong results:");
    for (worker_id, result) in ping_results {
        match result {
            Ok(()) => info!("Worker {} responded successfully", worker_id),
            Err(e) => info!("Worker {} failed: {}", worker_id, e),
        }
    }

    info!("=== Cohort Parallel Work Distribution Test ===");

    // Test 2: Parallel map with different work for each worker
    let work_results: Vec<WorkResponse> = cohort.par_map(
        "work",
        |rank, _worker_id| async move {
            // Create different work for each rank
            Ok(WorkRequest {
                rank,
                workload: format!("task-{}", rank),
                multiplier: (rank + 1) as i32, // rank 0 gets 1x, rank 1 gets 2x, etc.
            })
        },
        Duration::from_secs(10),
    ).await?;

    info!("Work distribution results (in rank order):");
    for (i, response) in work_results.iter().enumerate() {
        info!("Rank {}: processed '{}' -> {} chars",
              i, response.result, response.processed_length);
    }

    info!("=== Cohort Parallel Broadcast Test ===");

    // Test 3: Broadcast same work to all workers
    let broadcast_results: Vec<WorkResponse> = cohort.par_broadcast_responses(
        "work",
        WorkRequest {
            rank: 999, // Will be overridden by each worker
            workload: "broadcast-task".to_string(),
            multiplier: 3,
        },
        Duration::from_secs(10),
    ).await?;

    info!("Broadcast results (in rank order):");
    for (i, response) in broadcast_results.iter().enumerate() {
        info!("Rank {}: worker {} processed broadcast task -> {} chars",
              i, response.rank, response.processed_length);
    }

    info!("=== Cohort Indexed Parallel Map Test ===");

    // Test 4: Indexed parallel map with rank and worker_id information
    let indexed_results: Vec<(usize, WorkResponse)> = cohort.par_map_indexed(
        "work",
        |rank, worker_id| async move {
            // Use both rank and worker_id in the work
            Ok(WorkRequest {
                rank,
                workload: format!("indexed-rank-{}-worker-{}", rank, worker_id),
                multiplier: 2,
            })
        },
        Duration::from_secs(10),
    ).await?;

    info!("Indexed map results:");
    for (rank, response) in indexed_results {
        info!("Rank {}: worker processed '{}' -> {} chars",
              rank, response.result, response.processed_length);
    }

    info!("=== Performance Summary ===");
    info!("Successfully demonstrated:");
    info!("  - Parallel ping-pong with ACK collection");
    info!("  - Parallel work distribution with custom payloads per worker");
    info!("  - Parallel broadcast with rank-ordered response collection");
    info!("  - Indexed parallel map with rank and worker_id access");
    info!("  - All responses returned in proper rank order");

    info!("Shutting down cohort and workers...");

    // Cleanup
    for manager in worker_managers {
        manager.shutdown().await?;
    }
    leader_manager.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}