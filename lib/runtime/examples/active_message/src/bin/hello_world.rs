// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use dynamo_runtime::active_message::{
    client::ActiveMessageClient,
    handler::{ActiveMessage, HandlerType, ResponseHandler},
    manager::ActiveMessageManager,
    zmq::ZmqActiveMessageManager,
};
use serde::{Deserialize, Serialize};
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

/// Response type for hello world transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HelloResponse {
    transformed: String,
    bits_added: usize,
}

/// Hello world handler that appends bits to strings
#[derive(Debug, Clone)]
struct HelloWorldHandler;

#[async_trait]
impl ResponseHandler for HelloWorldHandler {
    type Response = HelloResponse;

    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
    ) -> Result<Self::Response> {
        let input: String = message.deserialize()?;
        info!("Received input: '{}'", input);

        // Transform the string by appending some "bits"
        let bits = " 🔥⚡🚀✨🎯";
        let transformed = format!("{}{}", input, bits);
        let bits_added = bits.len();

        info!("Transformed: '{}' -> '{}'", input, transformed);

        Ok(HelloResponse {
            transformed,
            bits_added,
        })
    }

    fn name(&self) -> &str {
        "hello_world"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("Starting hello-world transformation example");

    let cancel_token = CancellationToken::new();

    // Create two managers for request-response
    let server_manager =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    let client_manager =
        ZmqActiveMessageManager::new(unique_ipc_socket_path()?, cancel_token.clone()).await?;

    // Register hello world handler on server
    let hello_handler = Arc::new(HelloWorldHandler);
    let handler_type = HandlerType::response((*hello_handler).clone());
    server_manager
        .register_handler_typed(handler_type, None)
        .await?;

    let server_client = server_manager.client();
    let client_client = client_manager.client();

    println!("Server listening on: {}", server_client.endpoint());
    println!("Client endpoint: {}", client_client.endpoint());

    // Connect client to server
    let server_peer = server_client.peer_info();

    client_client.connect_to_peer(server_peer).await?;

    // Wait for connection to establish
    tokio::time::sleep(Duration::from_millis(100)).await;

    println!("Starting hello-world transformations...");

    // Test various inputs
    let test_inputs = [
        "Hello World",
        "Active Messages Rock",
        "Rust is Awesome",
        "ZeroMQ FTW",
        "Performance Matters",
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        println!("=== Test {} ===", i + 1);

        // Send string and wait for transformed response
        let result = client_client
            .active_message("hello_world")?
            .payload(input)?
            .expect_response::<HelloResponse>()
            .send(server_client.instance_id())
            .await?
            .await_response::<HelloResponse>()
            .await;

        match result {
            Ok(response) => {
                println!("Input: '{}'", input);
                println!("Output: '{}'", response.transformed);
                println!("Bits added: {} characters", response.bits_added);
                println!(
                    "Length: {} -> {} characters",
                    input.len(),
                    response.transformed.len()
                );
            }
            Err(e) => {
                println!("Transformation failed: {}", e);
            }
        }

        // Small delay between requests
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Test error handling with invalid JSON
    println!("=== Testing Error Handling ===");

    // This should work since we're sending a string
    let result = client_client
        .active_message("hello_world")?
        .payload("Valid String")?
        .expect_response::<HelloResponse>()
        .send(server_client.instance_id())
        .await?
        .await_response::<HelloResponse>()
        .await;

    match result {
        Ok(response) => {
            println!("Valid string handled correctly: '{}'", response.transformed);
        }
        Err(e) => {
            println!("Unexpected error with valid string: {}", e);
        }
    }

    println!("Hello-world example completed!");

    println!("Shutting down...");
    server_manager.shutdown().await?;
    client_manager.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}
