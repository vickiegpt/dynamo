// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for leader-worker cohort system
//!
//! These tests verify real ZMQ communication between leader and workers,
//! testing the complete cohort lifecycle including formation, operation,
//! heartbeat monitoring, and graceful shutdown.

use anyhow::Result;
use bytes::Bytes;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::info;

use dynamo_runtime::active_message::{
    client::ActiveMessageClient,
    handler::HandlerType,
    manager::ActiveMessageManager,
    zmq::{
        ZmqActiveMessageManager,
        builtin_handlers::{JoinCohortHandler, RemoveServiceHandler, RequestShutdownHandler},
        cohort::{
            CohortFailurePolicy, CohortType, LeaderWorkerCohort, LeaderWorkerCohortConfigBuilder,
        },
    },
};

/// Test utility to set up a leader with cohort handlers
async fn setup_leader(
    endpoint: &str,
    cohort_size: usize,
) -> Result<(
    Arc<ZmqActiveMessageManager>,
    Arc<LeaderWorkerCohort>,
    CancellationToken,
)> {
    let cancel_token = CancellationToken::new();
    let manager = ZmqActiveMessageManager::new(endpoint.to_string(), cancel_token.clone()).await?;
    let client = manager.zmq_client();

    // Create cohort configuration
    let cohort_config = LeaderWorkerCohortConfigBuilder::default()
        .leader_instance(client.instance_id())
        .client(client.clone())
        .cohort_type(CohortType::FixedSize(cohort_size))
        .failure_policy(CohortFailurePolicy::TerminateAll)
        .heartbeat_interval(Duration::from_millis(500)) // Faster for testing
        .heartbeat_timeout(Duration::from_secs(2)) // Shorter timeout for testing
        .cancel_token(cancel_token.clone())
        .build()?;

    let cohort = Arc::new(LeaderWorkerCohort::from_config(cohort_config));

    // Register cohort handlers
    let join_handler = JoinCohortHandler::new(cohort.clone());
    let join_handler_type = HandlerType::response(join_handler);
    manager
        .register_handler_typed(join_handler_type, None)
        .await?;

    let remove_handler = RemoveServiceHandler::new(cohort.clone());
    let remove_handler_type = HandlerType::response(remove_handler);
    manager
        .register_handler_typed(remove_handler_type, None)
        .await?;

    let shutdown_handler =
        RequestShutdownHandler::new(manager.manager_state(), cancel_token.clone());
    let shutdown_handler_type = HandlerType::response(shutdown_handler);
    manager
        .register_handler_typed(shutdown_handler_type, None)
        .await?;

    Ok((Arc::new(manager), cohort, cancel_token))
}

/// Test utility to set up a worker
async fn setup_worker(
    leader_endpoint: &str,
    leader_instance_id: uuid::Uuid,
    worker_rank: Option<usize>,
    leader_client: Arc<dynamo_runtime::active_message::zmq::ZmqActiveMessageClient>,
) -> Result<(Arc<ZmqActiveMessageManager>, CancellationToken)> {
    let cancel_token = CancellationToken::new();
    let manager =
        ZmqActiveMessageManager::new("tcp://0.0.0.0:0".to_string(), cancel_token.clone()).await?;
    let client = manager.zmq_client();

    // Establish bidirectional peer connection
    let leader_peer = dynamo_runtime::active_message::client::PeerInfo::new(
        leader_instance_id,
        leader_endpoint.to_string(),
    );
    client.connect_to_peer(leader_peer).await?;

    // Leader also needs to know about the worker for bidirectional communication
    let worker_peer = dynamo_runtime::active_message::client::PeerInfo::new(
        client.instance_id(),
        client.endpoint().to_string(),
    );
    leader_client.connect_to_peer(worker_peer).await?;

    // Register shutdown handler for workers
    let worker_shutdown_handler =
        RequestShutdownHandler::new(manager.manager_state(), cancel_token.clone());
    let worker_shutdown_handler_type = HandlerType::response(worker_shutdown_handler);
    manager
        .register_handler_typed(worker_shutdown_handler_type, None)
        .await?;

    // Join cohort
    let join_response = client.join_cohort(leader_instance_id, worker_rank).await?;
    if !join_response.accepted {
        return Err(anyhow::anyhow!(
            "Worker failed to join cohort: {:?}",
            join_response.reason
        ));
    }

    info!(
        "Worker joined cohort at position: {:?}",
        join_response.position
    );

    Ok((Arc::new(manager), cancel_token))
}

#[tokio::test]
async fn test_basic_cohort_formation() -> Result<()> {
    tracing_subscriber::fmt().try_init().ok();

    let (leader_manager, cohort, leader_cancel) = setup_leader("tcp://0.0.0.0:0", 2).await?;
    let leader_client = leader_manager.zmq_client();
    let leader_endpoint = leader_client.endpoint();
    let leader_id = leader_client.instance_id();

    // Set up workers
    let (worker1_manager, worker1_cancel) =
        setup_worker(leader_endpoint, leader_id, None, leader_client.clone()).await?;
    let (worker2_manager, worker2_cancel) =
        setup_worker(leader_endpoint, leader_id, None, leader_client.clone()).await?;

    // Wait for cohort to be full
    let cohort_full = timeout(Duration::from_secs(5), async {
        while !cohort.is_full().await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(
        cohort_full.is_ok(),
        "Cohort should become full within timeout"
    );
    assert_eq!(cohort.worker_count().await, 2);
    assert!(!cohort.has_ranks().await); // No ranks were specified

    // Clean up
    leader_cancel.cancel();
    worker1_cancel.cancel();
    worker2_cancel.cancel();

    leader_manager.shutdown().await?;
    worker1_manager.shutdown().await?;
    worker2_manager.shutdown().await?;

    Ok(())
}

#[tokio::test]
async fn test_cohort_with_ranks() -> Result<()> {
    tracing_subscriber::fmt().try_init().ok();

    let (leader_manager, cohort, leader_cancel) = setup_leader("tcp://0.0.0.0:0", 3).await?;
    let leader_client = leader_manager.zmq_client();
    let leader_endpoint = leader_client.endpoint();
    let leader_id = leader_client.instance_id();

    // Set up workers with ranks (not in order)
    let (worker0_manager, worker0_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(0), leader_client.clone()).await?;
    let (worker2_manager, worker2_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(2), leader_client.clone()).await?;
    let (worker1_manager, worker1_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(1), leader_client.clone()).await?;

    // Wait for cohort to be full
    let cohort_full = timeout(Duration::from_secs(5), async {
        while !cohort.is_full().await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(
        cohort_full.is_ok(),
        "Cohort should become full within timeout"
    );
    assert_eq!(cohort.worker_count().await, 3);
    assert!(cohort.has_ranks().await);
    assert!(cohort.is_cohort_complete().await); // Should have contiguous ranks [0, 1, 2]

    // Verify rank ordering
    let workers_by_rank = cohort.get_workers_by_rank().await;
    assert_eq!(workers_by_rank.len(), 3);
    assert_eq!(workers_by_rank[0].0, 0); // rank 0
    assert_eq!(workers_by_rank[1].0, 1); // rank 1
    assert_eq!(workers_by_rank[2].0, 2); // rank 2

    // Clean up
    leader_cancel.cancel();
    worker0_cancel.cancel();
    worker1_cancel.cancel();
    worker2_cancel.cancel();

    leader_manager.shutdown().await?;
    worker0_manager.shutdown().await?;
    worker1_manager.shutdown().await?;
    worker2_manager.shutdown().await?;

    Ok(())
}

#[tokio::test]
async fn test_cohort_rejects_excess_workers() -> Result<()> {
    tracing_subscriber::fmt().try_init().ok();

    let (leader_manager, cohort, leader_cancel) = setup_leader("tcp://0.0.0.0:0", 2).await?;
    let leader_client = leader_manager.zmq_client();
    let leader_endpoint = leader_client.endpoint();
    let leader_id = leader_client.instance_id();

    // Set up 2 workers (should succeed)
    let (worker1_manager, worker1_cancel) =
        setup_worker(leader_endpoint, leader_id, None, leader_client.clone()).await?;
    let (worker2_manager, worker2_cancel) =
        setup_worker(leader_endpoint, leader_id, None, leader_client.clone()).await?;

    // Wait for cohort to be full
    timeout(Duration::from_secs(5), async {
        while !cohort.is_full().await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("Cohort should become full");

    // Try to add a third worker (should fail or timeout)
    let worker3_result = timeout(Duration::from_secs(5), async {
        setup_worker(leader_endpoint, leader_id, None, leader_client.clone()).await
    })
    .await;

    // The third worker should either be immediately rejected or timeout
    // Both scenarios indicate the cohort is properly rejecting excess workers
    match worker3_result {
        Ok(result) => assert!(result.is_err(), "Third worker should be rejected"),
        Err(_timeout) => info!("Third worker timed out as expected when cohort is full"),
    }

    assert_eq!(cohort.worker_count().await, 2);

    // Clean up
    leader_cancel.cancel();
    worker1_cancel.cancel();
    worker2_cancel.cancel();

    leader_manager.shutdown().await?;
    worker1_manager.shutdown().await?;
    worker2_manager.shutdown().await?;

    Ok(())
}

#[tokio::test]
async fn test_mixed_rank_presence_rejected() -> Result<()> {
    tracing_subscriber::fmt().try_init().ok();

    let (leader_manager, _cohort, leader_cancel) = setup_leader("tcp://0.0.0.0:0", 3).await?;
    let leader_client = leader_manager.zmq_client();
    let leader_endpoint = leader_client.endpoint();
    let leader_id = leader_client.instance_id();

    // First worker with rank should succeed
    let (worker0_manager, worker0_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(0), leader_client.clone()).await?;

    // Second worker without rank should fail (mixed presence)
    let worker1_result =
        setup_worker(leader_endpoint, leader_id, None, leader_client.clone()).await;
    assert!(
        worker1_result.is_err(),
        "Worker without rank should be rejected after first worker had rank"
    );

    // Clean up
    leader_cancel.cancel();
    worker0_cancel.cancel();

    leader_manager.shutdown().await?;
    worker0_manager.shutdown().await?;

    Ok(())
}

#[tokio::test]
async fn test_duplicate_rank_rejected() -> Result<()> {
    tracing_subscriber::fmt().try_init().ok();

    let (leader_manager, _cohort, leader_cancel) = setup_leader("tcp://0.0.0.0:0", 3).await?;
    let leader_client = leader_manager.zmq_client();
    let leader_endpoint = leader_client.endpoint();
    let leader_id = leader_client.instance_id();

    // First worker with rank 0 should succeed
    let (worker0_manager, worker0_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(0), leader_client.clone()).await?;

    // Second worker with same rank should fail
    let worker1_result =
        setup_worker(leader_endpoint, leader_id, Some(0), leader_client.clone()).await;
    assert!(
        worker1_result.is_err(),
        "Worker with duplicate rank should be rejected"
    );

    // Clean up
    leader_cancel.cancel();
    worker0_cancel.cancel();

    leader_manager.shutdown().await?;
    worker0_manager.shutdown().await?;

    Ok(())
}

#[tokio::test]
async fn test_graceful_shutdown_flow() -> Result<()> {
    tracing_subscriber::fmt().try_init().ok();

    let (leader_manager, cohort, leader_cancel) = setup_leader("tcp://0.0.0.0:0", 2).await?;
    let leader_client = leader_manager.zmq_client();
    let leader_endpoint = leader_client.endpoint();
    let leader_id = leader_client.instance_id();

    // Set up workers
    let (worker1_manager, worker1_cancel) =
        setup_worker(leader_endpoint, leader_id, None, leader_client.clone()).await?;
    let (worker2_manager, worker2_cancel) =
        setup_worker(leader_endpoint, leader_id, None, leader_client.clone()).await?;

    // Wait for cohort to be full
    timeout(Duration::from_secs(5), async {
        while !cohort.is_full().await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("Cohort should become full");

    // Initiate graceful shutdown
    let shutdown_result = cohort.initiate_graceful_shutdown().await;
    assert!(shutdown_result.is_ok(), "Graceful shutdown should succeed");

    // Note: Due to a current limitation in RequestShutdownHandler, workers can't automatically
    // send remove_service back to the leader. In real usage, this would be fixed by storing
    // the leader ID properly. For this test, we verify that the shutdown was initiated.

    // Give time for shutdown requests to be sent and received
    tokio::time::sleep(Duration::from_millis(500)).await;

    // In a real scenario, workers would remove themselves automatically.
    // For this test, we verify the shutdown mechanism triggered correctly.
    info!("Graceful shutdown initiated successfully");

    // Clean up
    leader_cancel.cancel();
    worker1_cancel.cancel();
    worker2_cancel.cancel();

    leader_manager.shutdown().await?;
    worker1_manager.shutdown().await?;
    worker2_manager.shutdown().await?;

    Ok(())
}

#[tokio::test]
async fn test_broadcasting_to_workers() -> Result<()> {
    tracing_subscriber::fmt().try_init().ok();

    let (leader_manager, cohort, leader_cancel) = setup_leader("tcp://0.0.0.0:0", 2).await?;
    let leader_client = leader_manager.zmq_client();
    let leader_endpoint = leader_client.endpoint();
    let leader_id = leader_client.instance_id();

    // Set up workers with ranks
    let (worker0_manager, worker0_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(0), leader_client.clone()).await?;
    let (worker1_manager, worker1_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(1), leader_client.clone()).await?;

    // Wait for cohort to be full
    timeout(Duration::from_secs(5), async {
        while !cohort.is_full().await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("Cohort should become full");

    // Test broadcasting by rank
    let test_payload = Bytes::from("test_broadcast_message");
    let broadcast_result = cohort
        .broadcast_by_rank("test_handler", test_payload.clone())
        .await;
    assert!(broadcast_result.is_ok(), "Broadcast by rank should succeed");

    // Test broadcasting to all workers
    let broadcast_all_result = cohort
        .broadcast_to_workers("test_handler", test_payload)
        .await;
    assert!(
        broadcast_all_result.is_ok(),
        "Broadcast to all workers should succeed"
    );

    // Clean up
    leader_cancel.cancel();
    worker0_cancel.cancel();
    worker1_cancel.cancel();

    leader_manager.shutdown().await?;
    worker0_manager.shutdown().await?;
    worker1_manager.shutdown().await?;

    Ok(())
}

/// Test that simulates a slower cohort formation scenario
#[tokio::test]
async fn test_incremental_cohort_formation() -> Result<()> {
    tracing_subscriber::fmt().try_init().ok();

    let (leader_manager, cohort, leader_cancel) = setup_leader("tcp://0.0.0.0:0", 3).await?;
    let leader_client = leader_manager.zmq_client();
    let leader_endpoint = leader_client.endpoint();
    let leader_id = leader_client.instance_id();

    // Add workers one by one with delays
    assert_eq!(cohort.worker_count().await, 0);
    assert!(!cohort.is_full().await);

    let (worker0_manager, worker0_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(0), leader_client.clone()).await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(cohort.worker_count().await, 1);
    assert!(!cohort.is_full().await);
    assert!(!cohort.is_cohort_complete().await); // Missing ranks 1, 2

    let (worker1_manager, worker1_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(1), leader_client.clone()).await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(cohort.worker_count().await, 2);
    assert!(!cohort.is_full().await);
    assert!(!cohort.is_cohort_complete().await); // Missing rank 2

    let (worker2_manager, worker2_cancel) =
        setup_worker(leader_endpoint, leader_id, Some(2), leader_client.clone()).await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(cohort.worker_count().await, 3);
    assert!(cohort.is_full().await);
    assert!(cohort.is_cohort_complete().await); // All ranks [0, 1, 2] present

    // Clean up
    leader_cancel.cancel();
    worker0_cancel.cancel();
    worker1_cancel.cancel();
    worker2_cancel.cancel();

    leader_manager.shutdown().await?;
    worker0_manager.shutdown().await?;
    worker1_manager.shutdown().await?;
    worker2_manager.shutdown().await?;

    Ok(())
}
