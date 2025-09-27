// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::active_message::{
    client::ActiveMessageClient, manager::ActiveMessageManager, zmq::ZmqActiveMessageManager,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Generate a unique IPC socket path for testing
fn unique_ipc_socket_path() -> Result<String> {
    let temp_file = NamedTempFile::new()?;
    let path = temp_file.path().to_string_lossy().to_string();
    std::mem::forget(temp_file); // Don't delete the file yet
    Ok(format!("ipc://{}", path))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AddRequest {
    a: i32,
    b: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AddResponse {
    sum: i32,
}

#[tokio::test]
async fn test_typed_unary_api() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    // Register typed unary handler with the clean new API
    manager2
        .register_unary("add", |req: AddRequest, _ctx| async move {
            Ok(AddResponse { sum: req.a + req.b })
        })
        .await?;

    let client1 = manager1.client();
    let client2 = manager2.client();

    let peer2 = client2.peer_info();
    let peer1 = client1.peer_info();

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Test the handler
    let request = AddRequest { a: 5, b: 10 };
    let response: AddResponse = client1
        .active_message("add")?
        .payload(request)?
        .expect_response::<AddResponse>()
        .send(client2.instance_id())
        .await?
        .await_response()
        .await?;

    assert_eq!(response.sum, 15);

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_typed_void_api() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let messages = Arc::new(Mutex::new(Vec::<String>::new()));

    // Register typed void handler with the clean new API
    {
        let messages = messages.clone();
        manager2
            .register_void("log", move |msg: String, _ctx| {
                let messages = messages.clone();
                async move {
                    messages.lock().await.push(msg);
                }
            })
            .await?;
    }

    let client1 = manager1.client();
    let client2 = manager2.client();

    let peer2 = client2.peer_info();
    let peer1 = client1.peer_info();

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Test the handler
    client1
        .active_message("log")?
        .payload("Hello from typed API!")?
        .fire_and_forget(client2.instance_id())
        .await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let logged = messages.lock().await;
    assert_eq!(logged.len(), 1);
    assert_eq!(logged[0], "Hello from typed API!");

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_typed_ack_api() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    // Register typed ack handler with the clean new API
    manager2
        .register_typed_ack("validate_length", |text: String, _ctx| async move {
            if text.len() >= 5 && text.len() <= 20 {
                Ok(())
            } else {
                anyhow::bail!("Text must be between 5 and 20 characters")
            }
        })
        .await?;

    let client1 = manager1.client();
    let client2 = manager2.client();

    let peer2 = client2.peer_info();
    let peer1 = client1.peer_info();

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Test with valid length
    let result = client1
        .active_message("validate_length")?
        .payload("Valid text")?
        .send(client2.instance_id())
        .await;
    assert!(result.is_ok(), "Valid length text should succeed");

    // Test with invalid length
    let result = client1
        .active_message("validate_length")?
        .payload("Hi")?
        .send(client2.instance_id())
        .await;
    assert!(result.is_err(), "Too short text should fail");

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}
