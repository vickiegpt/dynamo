// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::{ActiveMessageContext, HandlerType, NoReturnHandler},
    manager::ActiveMessageManager,
    zmq::ZmqActiveMessageManager,
};
use std::sync::Arc;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Generate a unique IPC socket path for testing
fn unique_ipc_socket_path() -> Result<String> {
    let temp_file = NamedTempFile::new()?;
    let path = temp_file.path().to_string_lossy().to_string();
    // Close the file but keep the path - ZMQ will create the socket
    drop(temp_file);
    Ok(format!("ipc://{}", path))
}

#[derive(Debug, Clone)]
struct TestHandler {
    received_messages: Arc<Mutex<Vec<String>>>,
}

impl TestHandler {
    fn new() -> Self {
        Self {
            received_messages: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl NoReturnHandler for TestHandler {
    async fn handle(&self, payload: Bytes, _ctx: ActiveMessageContext) {
        // Try to deserialize as JSON string first, fallback to raw string
        let payload_str = if let Ok(json_str) = serde_json::from_slice::<String>(&payload) {
            json_str
        } else {
            match String::from_utf8(payload.to_vec()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Failed to decode message payload: {}", e);
                    return;
                }
            }
        };
        self.received_messages.lock().await.push(payload_str);
    }

    fn name(&self) -> &str {
        "test_handler"
    }
}

#[tokio::test]
async fn test_handler_type_validation_mismatch() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    // Register a NoReturnHandler but try to call it expecting a response (mismatch)
    let handler = TestHandler::new();
    let handler_type = HandlerType::no_return(handler);
    manager2.register_handler_typed(handler_type, None).await?;

    let client1 = manager1.client();
    let client2 = manager2.client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Try to call NoReturnHandler expecting a response - this should fail with handler type mismatch
    let result = client1
        .active_message("test_handler")?
        .payload("test")?
        .expect_response::<serde_json::Value>()
        .send(client2.instance_id())
        .await;

    // This should timeout because handler type validation failed and NACK was sent
    assert!(result.is_err());
    let error_msg = format!("{}", result.unwrap_err());
    assert!(
        error_msg.contains("acceptance timeout")
            || error_msg.contains("acceptance channel dropped")
    );

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_handler_type_validation_correct() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    // Register a NoReturnHandler and call it with fire_and_forget (correct match)
    let handler = TestHandler::new();
    let handler_type = HandlerType::no_return(handler);
    manager2.register_handler_typed(handler_type, None).await?;

    let client1 = manager1.client();
    let client2 = manager2.client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Call NoReturnHandler with fire_and_forget - this should work
    let result = client1
        .active_message("test_handler")?
        .payload("test message")?
        .fire_and_forget(client2.instance_id())
        .await;

    // This should succeed
    assert!(result.is_ok());

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}
