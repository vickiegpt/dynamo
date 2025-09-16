// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example demonstrating LocalClient functionality

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::register_local_engine;
use dynamo_runtime::engine::{AsyncEngine, async_trait};
use std::sync::Arc;

/// Simple test engine that echoes strings
struct SimpleEchoEngine;

#[async_trait]
impl AsyncEngine<String, String, String> for SimpleEchoEngine {
    async fn generate(&self, request: String) -> Result<String, String> {
        println!("Engine received: {}", request);
        Ok(format!("Echo: {}", request))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== LocalClient Demo ===\n");

    // Create runtime and DRT
    println!("1. Creating runtime...");
    let runtime = dynamo_runtime::Runtime::builder()
        .app_name("local-client-demo")
        .build()
        .await?;

    let config = dynamo_runtime::DistributedConfig::builder()
        .etcd_config(Default::default())
        .nats_config(Default::default())
        .is_static(true)
        .build()?;

    let drt = DistributedRuntime::new(runtime, config).await?;
    println!("   ✓ Runtime created\n");

    // Create namespace, component, and endpoint
    println!("2. Creating endpoint structure...");
    let namespace = drt.namespace("demo")?;
    let component = namespace.component("echo-service")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("echo");
    println!("   ✓ Created endpoint: demo/echo-service/echo\n");

    // Create and register an engine
    println!("3. Creating and registering engine...");
    let engine: Arc<dyn AsyncEngine<String, String, String>> = Arc::new(SimpleEchoEngine);

    // Register the engine for local access
    let key = register_local_engine(&endpoint, engine.clone()).await?;
    println!("   ✓ Registered engine with key: {}\n", key);

    // Demonstrate direct local invocation
    println!("4. Testing direct local invocation...");
    println!("   (This bypasses all network layers and invokes the engine directly)");

    // Direct invocation through the registered engine
    let response = engine
        .generate("Hello from direct call!".to_string())
        .await?;
    println!("   Response: {}\n", response);

    // Show what LocalClient would do
    println!("5. LocalClient usage (conceptual):");
    println!("   - LocalClient::from_endpoint(&endpoint) would retrieve the engine");
    println!("   - local_client.generate(request) would call the engine directly");
    println!("   - No network overhead, no etcd watching, no instance discovery\n");

    // Show the registered engines
    println!("6. Registry information:");
    println!("   - Key format: namespace/component/endpoint");
    println!("   - Registered key: {}", key);
    println!("   - Engine is type-erased as AnyAsyncEngine");
    println!("   - LocalClient downcasts back to specific types\n");

    // Cleanup
    println!("7. Cleanup...");
    drt.unregister_local_engine(&key).await;
    println!("   ✓ Unregistered engine\n");

    println!("=== Demo Complete ===");
    Ok(())
}
