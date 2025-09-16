// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for LocalClient functionality

use dynamo_runtime::component::{LocalClient, register_local_engine};
use dynamo_runtime::engine::{AsyncEngine, Data, async_trait};
use dynamo_runtime::pipeline::{Context, Ingress, ManyOut, ResponseStream, SingleIn};
use dynamo_runtime::{DistributedRuntime, Runtime};
use futures::StreamExt;
use std::sync::Arc;

/// Simple test engine that echoes the input
struct EchoEngine;

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<String>, anyhow::Error> for EchoEngine {
    async fn generate(&self, request: SingleIn<String>) -> Result<ManyOut<String>, anyhow::Error> {
        let response = request.data().clone();
        let ctx = request.context();

        // Create a simple stream that yields the response once
        let stream = futures::stream::once(async move { response });
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

#[tokio::test]
async fn test_local_client_registration_and_retrieval() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime and DRT
    let runtime = Runtime::new("test").await?;
    let config = dynamo_runtime::distributed::DistributedConfig {
        etcd_config: Default::default(),
        nats_config: Default::default(),
        is_static: true,
    };
    let drt = DistributedRuntime::new(runtime, config).await?;

    // Create namespace, component, and endpoint
    let namespace = drt.namespace("test-ns")?;
    let component = namespace.component("test-component")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("test-endpoint");

    // Create and register an engine
    let engine: Arc<dyn AsyncEngine<SingleIn<String>, ManyOut<String>, anyhow::Error>> =
        Arc::new(EchoEngine);

    // Register the engine
    let key = register_local_engine(&endpoint, engine.clone()).await?;
    println!("Registered engine with key: {}", key);

    // Create a LocalClient and retrieve the engine
    let local_client: LocalClient<SingleIn<String>, ManyOut<String>, anyhow::Error> =
        LocalClient::from_endpoint(&endpoint).await?;

    // Test the local client
    let context = Arc::new(dynamo_runtime::pipeline::context::BaseContext::new(
        dynamo_runtime::pipeline::context::ConnectionInfo::Direct,
    ));
    let request = Context::new("Hello, LocalClient!".to_string(), context);

    let mut response_stream = local_client.generate(request).await?;
    let response = response_stream.next().await.expect("Expected response");

    assert_eq!(response, "Hello, LocalClient!");
    println!("LocalClient test passed: received '{}'", response);

    // Cleanup: unregister the engine
    drt.unregister_local_engine(&key).await;

    // Verify it's gone
    let result =
        LocalClient::<SingleIn<String>, ManyOut<String>, anyhow::Error>::from_endpoint(&endpoint)
            .await;
    assert!(
        result.is_err(),
        "Expected error when retrieving unregistered engine"
    );

    Ok(())
}

#[tokio::test]
async fn test_local_client_with_ingress() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime and DRT
    let runtime = Runtime::new("test").await?;
    let config = dynamo_runtime::distributed::DistributedConfig {
        etcd_config: Default::default(),
        nats_config: Default::default(),
        is_static: true,
    };
    let drt = DistributedRuntime::new(runtime, config).await?;

    // Create namespace, component, and endpoint
    let namespace = drt.namespace("test-ns2")?;
    let component = namespace.component("test-component2")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("test-endpoint2");

    // Create an Ingress with an engine
    let engine: Arc<dyn AsyncEngine<SingleIn<String>, ManyOut<String>, anyhow::Error>> =
        Arc::new(EchoEngine);
    let ingress = Ingress::for_engine(engine)?;

    // Note: The ingress won't automatically register its engine because
    // the trait bounds aren't satisfied at the generic level.
    // Users would need to explicitly register engines they want local access to.

    // For this test, we'll manually register the engine
    let test_engine: Arc<dyn AsyncEngine<SingleIn<String>, ManyOut<String>, anyhow::Error>> =
        Arc::new(EchoEngine);
    let key = register_local_engine(&endpoint, test_engine).await?;

    // Now we can create a LocalClient
    let local_client: LocalClient<SingleIn<String>, ManyOut<String>, anyhow::Error> =
        LocalClient::from_endpoint(&endpoint).await?;

    // Test the local client
    let context = Arc::new(dynamo_runtime::pipeline::context::BaseContext::new(
        dynamo_runtime::pipeline::context::ConnectionInfo::Direct,
    ));
    let request = Context::new("Test with Ingress".to_string(), context);

    let mut response_stream = local_client.generate(request).await?;
    let response = response_stream.next().await.expect("Expected response");

    assert_eq!(response, "Test with Ingress");
    println!(
        "LocalClient with Ingress test passed: received '{}'",
        response
    );

    // Cleanup
    drt.unregister_local_engine(&key).await;

    Ok(())
}

#[tokio::test]
async fn test_local_client_type_mismatch() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime and DRT
    let runtime = Runtime::new("test").await?;
    let config = dynamo_runtime::distributed::DistributedConfig {
        etcd_config: Default::default(),
        nats_config: Default::default(),
        is_static: true,
    };
    let drt = DistributedRuntime::new(runtime, config).await?;

    // Create namespace, component, and endpoint
    let namespace = drt.namespace("test-ns3")?;
    let component = namespace.component("test-component3")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("test-endpoint3");

    // Register an engine with String types
    let engine: Arc<dyn AsyncEngine<SingleIn<String>, ManyOut<String>, anyhow::Error>> =
        Arc::new(EchoEngine);
    let key = register_local_engine(&endpoint, engine).await?;

    // Try to create a LocalClient with different types (this should fail)
    let result =
        LocalClient::<SingleIn<i32>, ManyOut<i32>, anyhow::Error>::from_endpoint(&endpoint).await;

    assert!(result.is_err(), "Expected type mismatch error");
    if let Err(e) = result {
        println!("Got expected error for type mismatch: {}", e);
    }

    // Cleanup
    drt.unregister_local_engine(&key).await;

    Ok(())
}
