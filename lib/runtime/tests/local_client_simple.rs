// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Simple test to demonstrate LocalClient functionality

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::{LocalClient, register_local_engine};
use dynamo_runtime::engine::{AsyncEngine, async_trait};
use std::sync::Arc;

/// Simple test engine that echoes strings
struct SimpleEchoEngine;

#[async_trait]
impl AsyncEngine<String, String, String> for SimpleEchoEngine {
    async fn generate(&self, request: String) -> Result<String, String> {
        Ok(format!("Echo: {}", request))
    }
}

#[tokio::test]
async fn test_local_client_basic() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime and DRT
    let runtime = dynamo_runtime::Runtime::builder()
        .app_name("test")
        .build()
        .await?;

    let config = dynamo_runtime::DistributedConfig::builder()
        .etcd_config(Default::default())
        .nats_config(Default::default())
        .is_static(true)
        .build()?;

    let drt = DistributedRuntime::new(runtime, config).await?;

    // Create namespace, component, and endpoint
    let namespace = drt.namespace("test-ns")?;
    let component = namespace.component("test-component")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("test-endpoint");

    // Create and register an engine
    let engine: Arc<dyn AsyncEngine<String, String, String>> = Arc::new(SimpleEchoEngine);

    // Register the engine for local access
    let key = register_local_engine(&endpoint, engine.clone()).await?;
    println!("✓ Registered engine with key: {}", key);

    // Create a LocalClient using the endpoint's convenience method
    let local_client: LocalClient<String, String, String> = endpoint.local_client().await?;
    println!("✓ Created LocalClient successfully");

    // Test the local client with direct invocation
    let response = local_client
        .generate("Hello, LocalClient!".to_string())
        .await?;
    assert_eq!(response, "Echo: Hello, LocalClient!");
    println!("✓ LocalClient test passed: received '{}'", response);

    // Note: We can't unregister manually since the registry methods are now internal
    // This is fine for tests as they'll be cleaned up when the test ends

    Ok(())
}

#[tokio::test]
async fn test_local_client_type_safety() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime and DRT
    let runtime = dynamo_runtime::Runtime::builder()
        .app_name("test")
        .build()
        .await?;

    let config = dynamo_runtime::DistributedConfig::builder()
        .etcd_config(Default::default())
        .nats_config(Default::default())
        .is_static(true)
        .build()?;

    let drt = DistributedRuntime::new(runtime, config).await?;

    // Create namespace, component, and endpoint
    let namespace = drt.namespace("test-ns2")?;
    let component = namespace.component("test-component2")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("test-endpoint2");

    // Register an engine with String types
    let engine: Arc<dyn AsyncEngine<String, String, String>> = Arc::new(SimpleEchoEngine);
    let key = register_local_engine(&endpoint, engine).await?;
    println!("✓ Registered String engine");

    // Try to create a LocalClient with different types (this should fail)
    let result: Result<LocalClient<i32, i32, String>, _> = endpoint.local_client().await;
    assert!(result.is_err(), "Expected type mismatch error");

    if let Err(e) = result {
        println!("✓ Got expected error for type mismatch: {}", e);
    }

    Ok(())
}
