// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    manager::ActiveMessageManager,
    zmq::ZmqActiveMessageManager,
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
    // Close the file but keep the path - ZMQ will create the socket
    drop(temp_file);
    Ok(format!("ipc://{}", path))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComputeRequest {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ComputeResponse {
    result: i32,
}

#[tokio::test]
async fn test_closure_handlers_comprehensive() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    // Shared state for testing
    let log_messages = Arc::new(Mutex::new(Vec::<String>::new()));
    let validation_results = Arc::new(Mutex::new(Vec::<bool>::new()));

    // Register NoReturnHandler closure - logs messages
    {
        let log_messages = log_messages.clone();
        manager2
            .register_no_return_closure("log", move |msg, _client| {
                let log_messages = log_messages.clone();
                async move {
                    // Deserialize the JSON string payload
                    if let Ok(text) = msg.deserialize::<String>() {
                        log_messages.lock().await.push(text);
                    }
                }
            })
            .await?;
    }

    // Register AckHandler closure that always fails for testing
    manager2
        .register_ack_closure("always_fail", |_msg, _client| async move {
            anyhow::bail!("This handler always fails")
        })
        .await?;

    // Register AckHandler closure - validates data
    {
        let validation_results = validation_results.clone();
        manager2
            .register_ack_closure("validate", move |msg, _client| {
                let validation_results = validation_results.clone();
                async move {
                    // Deserialize the JSON string payload
                    let text: String = msg.deserialize()?;
                    let is_valid = !text.is_empty() && text.len() < 100;
                    validation_results.lock().await.push(is_valid);

                    if is_valid {
                        Ok(())
                    } else {
                        anyhow::bail!("Invalid data: too long or empty")
                    }
                }
            })
            .await?;
    }

    // Register ResponseHandler closure - computes arithmetic
    manager2
        .register_response_closure("compute", |msg, _client| async move {
            let request: ComputeRequest = msg.deserialize()?;
            let result = request.x + request.y;
            Ok(ComputeResponse { result })
        })
        .await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Test NoReturnHandler (fire_and_forget)
    client1
        .active_message("log")?
        .payload("Hello from closure!")?
        .fire_and_forget(client2.instance_id())
        .await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test AckHandler that always fails
    let fail_result = client1
        .active_message("always_fail")?
        .payload("doesn't matter")?
        .send(client2.instance_id())
        .await;
    assert!(
        fail_result.is_err(),
        "always_fail handler should return error"
    );

    // Test AckHandler (send) - valid data
    let ack_result = client1
        .active_message("validate")?
        .payload("Valid data")?
        .send(client2.instance_id())
        .await;
    assert!(ack_result.is_ok());

    // Test AckHandler (send) - invalid data
    tokio::time::sleep(Duration::from_millis(100)).await;
    let ack_result = client1
        .active_message("validate")?
        .payload("")? // Empty string should be invalid
        .send(client2.instance_id())
        .await;
    assert!(ack_result.is_err(), "Empty string validation should fail");

    // Test ResponseHandler (expect_response)
    let compute_request = ComputeRequest { x: 5, y: 10 };
    let response: ComputeResponse = client1
        .active_message("compute")?
        .payload(compute_request)?
        .expect_response::<ComputeResponse>()
        .send(client2.instance_id())
        .await?
        .await_response()
        .await?;

    assert_eq!(response.result, 15);

    // Verify all handlers worked correctly
    tokio::time::sleep(Duration::from_millis(100)).await;

    let logged_messages = log_messages.lock().await;
    assert_eq!(logged_messages.len(), 1);
    assert_eq!(logged_messages[0], "Hello from closure!");

    let validation_results = validation_results.lock().await;
    assert_eq!(validation_results.len(), 2);
    assert!(validation_results[0]); // "Valid data" should pass
    assert!(!validation_results[1]); // Empty string should fail

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_closure_handler_type_validation() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    // Register a NoReturnHandler closure
    manager2
        .register_no_return_closure("test_handler", |_msg, _client| async move {
            // Do nothing
        })
        .await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Test correct usage - NoReturnHandler with fire_and_forget
    let result = client1
        .active_message("test_handler")?
        .payload("test")?
        .fire_and_forget(client2.instance_id())
        .await;
    assert!(result.is_ok());

    // Test incorrect usage - NoReturnHandler with expect_response should fail
    let result = client1
        .active_message("test_handler")?
        .payload("test")?
        .expect_response::<serde_json::Value>()
        .send(client2.instance_id())
        .await;
    assert!(result.is_err());

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}
