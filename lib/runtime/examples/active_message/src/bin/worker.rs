// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use bytes::Bytes;
use dynamo_runtime::active_message::{
    client::PeerInfo, manager::ActiveMessageManager, zmq::ZmqActiveMessageManager,
};
use std::env;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use active_message_example::ComputeHandler;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting Worker");

    let leader_endpoint =
        env::var("LEADER_ENDPOINT").unwrap_or_else(|_| "tcp://127.0.0.1:5555".to_string());

    let cancel_token = CancellationToken::new();

    let manager =
        ZmqActiveMessageManager::new("tcp://0.0.0.0:0".to_string(), cancel_token.clone()).await?;

    let client = manager.client();

    info!("Worker listening on: {}", client.endpoint());
    info!("Worker instance ID: {}", client.instance_id());

    let handler = Arc::new(ComputeHandler);
    manager.register_handler(handler, None).await?;

    info!("Registered compute handler");

    let leader_instance_id = uuid::Uuid::new_v4();
    let leader_peer = PeerInfo::new(leader_instance_id, leader_endpoint.clone());

    client.connect_to_peer(leader_peer).await?;

    info!("Connected to leader at {}", leader_endpoint);

    let register_payload = serde_json::json!({
        "instance_id": client.instance_id().to_string(),
        "endpoint": client.endpoint(),
    });

    client
        .send_message(
            leader_instance_id,
            "_register_service",
            Bytes::from(serde_json::to_vec(&register_payload)?),
        )
        .await?;

    info!("Registered with leader");

    cancel_token.cancelled().await;

    info!("Worker shutting down");
    manager.shutdown().await?;

    Ok(())
}
