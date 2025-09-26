// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use dynamo_runtime::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::{ActiveMessage, HandlerType, NoReturnHandler, ResponseHandler},
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

    async fn get_messages(&self) -> Vec<String> {
        self.received_messages.lock().await.clone()
    }
}

#[async_trait]
impl NoReturnHandler for TestHandler {
    async fn handle(&self, message: ActiveMessage, _client: &dyn ActiveMessageClient) {
        // Try to deserialize as JSON string first, fallback to raw string
        let payload = if let Ok(json_str) = serde_json::from_slice::<String>(&message.payload) {
            json_str
        } else {
            match String::from_utf8(message.payload.to_vec()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Failed to decode message payload: {}", e);
                    return;
                }
            }
        };
        self.received_messages.lock().await.push(payload);
    }

    fn name(&self) -> &str {
        "test_handler"
    }
}

#[tokio::test]
async fn test_basic_message_send_receive() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let handler = Arc::new(TestHandler::new());
    let handler_type = HandlerType::no_return((*handler).clone());
    manager2.register_handler_typed(handler_type, None).await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    let test_message = "Hello from manager1";
    client1
        .message("test_handler")?
        .payload(test_message)?
        .fire_and_forget(client2.instance_id())
        .await?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    let messages = handler.get_messages().await;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0], test_message);

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_handler_registration_events() -> Result<()> {
    let cancel_token = CancellationToken::new();

    let manager =
        ZmqActiveMessageManager::new("tcp://127.0.0.1:0".to_string(), cancel_token.clone()).await?;

    let mut events_rx = manager.handler_events();

    let handler = TestHandler::new();
    let handler_type = HandlerType::no_return(handler);
    manager.register_handler_typed(handler_type, None).await?;

    let event = tokio::time::timeout(Duration::from_secs(1), events_rx.recv()).await??;

    match event {
        dynamo_runtime::active_message::handler::HandlerEvent::Registered { name, .. } => {
            assert_eq!(name, "test_handler");
        }
        _ => panic!("Expected Registered event"),
    }

    manager.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_register_service_handler_with_response() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test register service with response using convenience method
    let service_info = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());
    let registered = client1
        .register_service(client2.instance_id(), service_info)
        .await?;

    assert!(registered);

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_list_handlers_with_response() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    // Register a test handler
    let handler = TestHandler::new();
    let handler_type = HandlerType::no_return(handler);
    manager2.register_handler_typed(handler_type, None).await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test list handlers with response using convenience method
    let handlers = client1.list_handlers(client2.instance_id()).await?;

    // Should contain our registered handlers plus built-ins
    assert!(handlers.contains(&"test_handler".to_string()));
    assert!(handlers.contains(&"_register_service".to_string()));
    assert!(handlers.contains(&"_list_handlers".to_string()));

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_wait_for_handler_with_response() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test wait for already existing handler using convenience method
    let available = client1
        .await_handler(
            client2.instance_id(),
            "_list_handlers",
            Some(Duration::from_millis(1000)),
        )
        .await?;

    assert!(available);

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_health_check_with_response() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test health check with response using convenience method
    let response = client1.health_check(client2.instance_id()).await?;

    assert_eq!(response.status, "ok");
    assert!(response.timestamp > 0);

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[tokio::test]
async fn test_message_builder_fire_and_forget() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let handler = Arc::new(TestHandler::new());
    let handler_type = HandlerType::no_return((*handler).clone());
    manager2.register_handler_typed(handler_type, None).await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test fire and forget
    let test_message = "Fire and forget message";
    client1
        .message("test_handler")?
        .payload(test_message)?
        .fire_and_forget(client2.instance_id())
        .await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let messages = handler.get_messages().await;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0], test_message);

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

#[derive(Debug, Clone)]
struct ErrorTestHandler {
    should_error: Arc<Mutex<bool>>,
}

impl ErrorTestHandler {
    fn new() -> Self {
        Self {
            should_error: Arc::new(Mutex::new(false)),
        }
    }

    async fn set_should_error(&self, value: bool) {
        *self.should_error.lock().await = value;
    }
}

#[async_trait]
impl ResponseHandler for ErrorTestHandler {
    type Response = serde_json::Value;

    async fn handle(
        &self,
        _message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
    ) -> Result<Self::Response> {
        let should_error = *self.should_error.lock().await;
        if should_error {
            anyhow::bail!("Test error from handler");
        } else {
            let test_response = serde_json::json!({"message": "success"});
            Ok(test_response)
        }
    }

    fn name(&self) -> &str {
        "error_test_handler"
    }
}

#[tokio::test]
async fn test_single_response_error_handling() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let handler = Arc::new(ErrorTestHandler::new());
    let handler_type = HandlerType::response((*handler).clone());
    manager2.register_handler_typed(handler_type, None).await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test success case
    handler.set_should_error(false).await;
    let status = client1
        .message("error_test_handler")?
        .payload("test")?
        .expect_response::<serde_json::Value>()
        .send(client2.instance_id())
        .await?;

    let response: serde_json::Value = status.await_response().await?;
    assert_eq!(response["message"], "success");

    // Test error case
    handler.set_should_error(true).await;
    let status = client1
        .message("error_test_handler")?
        .payload("test")?
        .expect_response::<serde_json::Value>()
        .send(client2.instance_id())
        .await?;

    let result: Result<serde_json::Value> = status.await_response().await;
    assert!(result.is_err());
    let error_msg = format!("{}", result.unwrap_err());
    assert!(error_msg.contains("Test error from handler"));

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

// New style handler using ResponseHandler trait
#[derive(Debug)]
struct NewStyleTestHandler {
    should_error: Arc<Mutex<bool>>,
}

impl NewStyleTestHandler {
    fn new() -> Self {
        Self {
            should_error: Arc::new(Mutex::new(false)),
        }
    }

    async fn set_should_error(&self, value: bool) {
        *self.should_error.lock().await = value;
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TestResponse {
    message: String,
}

#[async_trait]
impl ResponseHandler for NewStyleTestHandler {
    type Response = TestResponse;

    async fn handle(
        &self,
        _message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
    ) -> Result<Self::Response> {
        let should_error = *self.should_error.lock().await;
        if should_error {
            anyhow::bail!("Test error from new style handler");
        } else {
            Ok(TestResponse {
                message: "success from new style handler".to_string(),
            })
        }
    }

    fn name(&self) -> &str {
        "new_style_test_handler"
    }
}

#[tokio::test]
async fn test_new_style_response_handler() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let cancel_token = CancellationToken::new();

    let manager1 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;
    let manager2 =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let handler = Arc::new(NewStyleTestHandler::new());
    let handler_for_registration = NewStyleTestHandler {
        should_error: handler.should_error.clone(),
    };
    let handler_type = HandlerType::response(handler_for_registration);
    manager2.register_handler_typed(handler_type, None).await?;

    let client1 = manager1.zmq_client();
    let client2 = manager2.zmq_client();

    let peer2 = PeerInfo::new(client2.instance_id(), client2.endpoint().to_string());
    let peer1 = PeerInfo::new(client1.instance_id(), client1.endpoint().to_string());

    client1.connect_to_peer(peer2).await?;
    client2.connect_to_peer(peer1).await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test success case
    handler.set_should_error(false).await;
    let status = client1
        .message("new_style_test_handler")?
        .payload("test")?
        .expect_response::<TestResponse>()
        .send(client2.instance_id())
        .await?;

    let response: TestResponse = status.await_response().await?;
    assert_eq!(response.message, "success from new style handler");

    // Test error case
    handler.set_should_error(true).await;
    let status = client1
        .message("new_style_test_handler")?
        .payload("test")?
        .expect_response::<TestResponse>()
        .send(client2.instance_id())
        .await?;

    let result: Result<TestResponse> = status.await_response().await;
    assert!(result.is_err());
    let error_msg = format!("{}", result.unwrap_err());
    assert!(error_msg.contains("Test error from new style handler"));

    manager1.shutdown().await?;
    manager2.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}
