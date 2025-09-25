// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use bytes::Bytes;
use dynamo_runtime::active_message::{
    manager::ActiveMessageManager,
    zmq::{LeaderWorkerCohort, ZmqActiveMessageManager},
};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::info;

use active_message_example::ComputeRequest;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting Leader");

    let cancel_token = CancellationToken::new();

    let manager =
        ZmqActiveMessageManager::new("tcp://0.0.0.0:5555".to_string(), cancel_token.clone())
            .await?;

    let client = manager.client();

    info!("Leader listening on: {}", client.endpoint());
    info!("Leader instance ID: {}", client.instance_id());

    tokio::time::sleep(Duration::from_secs(2)).await;

    let workers = vec![];

    let cohort = LeaderWorkerCohort::new(client.instance_id(), workers, manager.zmq_client());

    info!("Waiting for workers to register compute handler...");

    if let Err(e) = cohort
        .await_handler_on_all_workers("compute", Some(Duration::from_secs(30)))
        .await
    {
        info!("No workers registered: {}", e);
    }

    let request = ComputeRequest {
        x: 10,
        y: 20,
        operation: "add".to_string(),
    };

    let payload = Bytes::from(serde_json::to_vec(&request)?);

    info!("Broadcasting compute request to workers");
    cohort.broadcast_to_workers("compute", payload).await?;

    tokio::time::sleep(Duration::from_secs(2)).await;

    info!("Leader shutting down");
    manager.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}
