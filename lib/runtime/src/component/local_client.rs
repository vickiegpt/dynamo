// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local client implementation for direct engine access without network overhead

use crate::engine::{
    AnyAsyncEngine, AsyncEngine, AsyncEngineContextProvider, Data, DowncastAnyAsyncEngine,
};
use crate::traits::DistributedRuntimeProvider;
use crate::v2::entity::{ComponentDescriptor, EndpointDescriptor, NamespaceDescriptor};
use crate::{Result, error};
use std::sync::Arc;

use super::Endpoint;

/// A client that directly invokes local engines without network overhead
#[derive(Clone)]
pub struct LocalClient<Req, Resp, E>
where
    Req: Data,
    Resp: Data + AsyncEngineContextProvider,
    E: Data,
{
    engine: Arc<dyn AsyncEngine<Req, Resp, E>>,
    descriptor: EndpointDescriptor,
}

impl<Req, Resp, E> LocalClient<Req, Resp, E>
where
    Req: Data,
    Resp: Data + AsyncEngineContextProvider,
    E: Data,
{
    /// Create a LocalClient from an endpoint descriptor
    pub async fn from_descriptor(
        endpoint: &Endpoint,
        descriptor: EndpointDescriptor,
    ) -> Result<Self> {
        let key = descriptor.to_string();

        // Get the engine from the local registry
        let any_engine = endpoint
            .drt()
            .get_local_engine(&key)
            .await
            .ok_or_else(|| error!("No local engine registered for endpoint: {}", key))?;

        // Downcast to the specific types
        let engine = any_engine
            .downcast::<Req, Resp, E>()
            .ok_or_else(|| error!("Type mismatch when downcasting local engine for: {}", key))?;

        Ok(Self { engine, descriptor })
    }

    /// Create a LocalClient from an endpoint
    pub async fn from_endpoint(endpoint: &Endpoint) -> Result<Self> {
        // Create the descriptor for this endpoint
        let namespace_desc =
            NamespaceDescriptor::new(&[endpoint.component.namespace.name.as_str()])
                .map_err(|e| error!("Invalid namespace: {}", e))?;
        let component_desc = namespace_desc
            .component(&endpoint.component.name)
            .map_err(|e| error!("Invalid component: {}", e))?;
        let endpoint_desc = component_desc
            .endpoint(&endpoint.name)
            .map_err(|e| error!("Invalid endpoint: {}", e))?;

        Self::from_descriptor(endpoint, endpoint_desc).await
    }

    /// Generate a response using the local engine directly
    pub async fn generate(&self, request: Req) -> Result<Resp, E> {
        self.engine.generate(request).await
    }

    /// Get the descriptor for this client
    pub fn descriptor(&self) -> &EndpointDescriptor {
        &self.descriptor
    }

    /// Get the underlying engine
    pub fn engine(&self) -> &Arc<dyn AsyncEngine<Req, Resp, E>> {
        &self.engine
    }
}

/// Helper to register a local engine with proper type erasure
pub async fn register_local_engine<Req, Resp, E>(
    endpoint: &Endpoint,
    engine: Arc<dyn AsyncEngine<Req, Resp, E>>,
) -> Result<String>
where
    Req: Data,
    Resp: Data + AsyncEngineContextProvider,
    E: Data,
{
    use crate::engine::AsAnyAsyncEngine;

    // Create the descriptor for this endpoint
    let namespace_desc = NamespaceDescriptor::new(&[endpoint.component.namespace.name.as_str()])
        .map_err(|e| error!("Invalid namespace: {}", e))?;
    let component_desc = namespace_desc
        .component(&endpoint.component.name)
        .map_err(|e| error!("Invalid component: {}", e))?;
    let endpoint_desc = component_desc
        .endpoint(&endpoint.name)
        .map_err(|e| error!("Invalid endpoint: {}", e))?;

    // Register using the path string as key
    let key = endpoint_desc.to_string();
    tracing::debug!("Registering local engine for endpoint: {}", key);

    // Type-erase and register
    let any_engine = engine.into_any_engine();
    endpoint
        .drt()
        .register_local_engine(key.clone(), any_engine)
        .await?;

    Ok(key)
}
