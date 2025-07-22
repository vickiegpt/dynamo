// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [Component] module defines the top-level API for building distributed applications.
//!
//! A distributed application consists of a set of [Component] that can host one
//! or more [Endpoint]. Each [Endpoint] is a network-accessible service
//! that can be accessed by other [Component] in the distributed application.
//!
//! A [Component] is made discoverable by registering it with the distributed runtime under
//! a [`Namespace`].
//!
//! A [`Namespace`] is a logical grouping of [Component] that are grouped together.
//!
//! We might extend namespace to include grouping behavior, which would define groups of
//! components that are tightly coupled.
//!
//! A [Component] is the core building block of a distributed application. It is a logical
//! unit of work such as a `Preprocessor` or `SmartRouter` that has a well-defined role in the
//! distributed application.
//!
//! A [Component] can present to the distributed application one or more configuration files
//! which define how that component was constructed/configured and what capabilities it can
//! provide.
//!
//! Other [Component] can write to watching locations within a [Component] etcd
//! path. This allows the [Component] to take dynamic actions depending on the watch
//! triggers.
//!
//! TODO: Top-level Overview of Endpoints/Functions

use crate::{
    discovery::Lease,
    entity::descriptor::{
        ComponentDescriptor, DescriptorError, EndpointDescriptor, InstanceDescriptor,
        NamespaceDescriptor,
    },
    service::ServiceSet,
    transports::etcd::EtcdPath,
};

use super::{
    error,
    traits::*,
    transports::etcd::{COMPONENT_KEYWORD, ENDPOINT_KEYWORD},
    transports::nats::Slug,
    utils::Duration,
    DistributedRuntime, Result, Runtime,
};

use crate::pipeline::network::{ingress::push_endpoint::PushEndpoint, PushWorkHandler};
use crate::protocols::Endpoint as EndpointId;
use async_nats::{
    rustls::quic,
    service::{Service, ServiceExt},
};
use derive_builder::Builder;
use derive_getters::Getters;
use educe::Educe;
use serde::{Deserialize, Serialize};
use service::EndpointStatsHandler;
use std::{collections::HashMap, hash::Hash, sync::Arc};
use validator::{Validate, ValidationError};

mod client;
#[allow(clippy::module_inception)]
mod component;
mod endpoint;
mod namespace;
mod registry;
pub mod service;

pub use client::{Client, InstanceSource};

/// The root etcd path where each instance registers itself in etcd.
/// An instance is namespace+component+endpoint+lease_id and must be unique.
pub const INSTANCE_ROOT_PATH: &str = "instances";

/// The root etcd path where each namespace is registered in etcd.
pub const ETCD_ROOT_PATH: &str = "dynamo://";

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransportType {
    NatsTcp(String),
}

#[derive(Default)]
pub struct RegistryInner {
    services: HashMap<String, Service>,
    stats_handlers: HashMap<String, Arc<std::sync::Mutex<HashMap<String, EndpointStatsHandler>>>>,
}

#[derive(Clone)]
pub struct Registry {
    inner: Arc<tokio::sync::Mutex<RegistryInner>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub instance_id: i64,
    pub transport: TransportType,
}

impl Instance {
    pub fn id(&self) -> i64 {
        self.instance_id
    }
}

/// A [Component] a discoverable entity in the distributed runtime.
/// You can host [Endpoint] on a [Component] by first creating
/// a [Service] then adding one or more [Endpoint] to the [Service].
///
/// You can also issue a request to a [Component]'s [Endpoint] by creating a [Client].
#[derive(Educe, Builder, Clone, Validate)]
#[educe(Debug)]
#[builder(pattern = "owned")]
pub struct Component {
    #[builder(private)]
    #[educe(Debug(ignore))]
    drt: Arc<DistributedRuntime>,

    /// Namespace reference for compatibility
    #[builder(setter(into))]
    namespace: Namespace,

    // A static component's endpoints cannot be discovered via etcd, they are
    // fixed at startup time.
    is_static: bool,

    // Component descriptor representation (primary source of truth)
    #[builder(private)]
    descriptor: ComponentDescriptor,
}

impl Hash for Component {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.descriptor.inner().namespace_segments().hash(state);
        self.descriptor.inner().component().hash(state);
        self.is_static.hash(state);
    }
}

impl PartialEq for Component {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor && self.is_static == other.is_static
    }
}

impl Eq for Component {}

impl std::fmt::Display for Component {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.namespace.name(), self.name())
    }
}

impl DistributedRuntimeProvider for Component {
    fn drt(&self) -> &DistributedRuntime {
        &self.drt
    }
}

impl RuntimeProvider for Component {
    fn rt(&self) -> &Runtime {
        self.drt.rt()
    }
}

impl Component {
    /// The component part of an instance path in etcd.
    pub fn etcd_root(&self) -> String {
        let ns = self.namespace.name();
        let cp = self.name();
        format!("{INSTANCE_ROOT_PATH}/{ns}/{cp}")
    }

    pub fn service_name(&self) -> String {
        format_service_name(&self.namespace.name(), &self.name())
    }

    pub fn path(&self) -> String {
        format_entity_path(&self.namespace.name(), &self.name())
    }

    pub fn etcd_path(&self) -> EtcdPath {
        EtcdPath::new_component(&self.namespace.name(), &self.name())
            .expect("Component name and namespace should be valid")
    }

    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    pub fn name(&self) -> String {
        extract_name_or_empty(self.descriptor.inner().component())
    }

    /// Get the component descriptor
    pub fn descriptor(&self) -> &ComponentDescriptor {
        &self.descriptor
    }

    pub fn endpoint(&self, endpoint: impl Into<String>) -> Endpoint {
        let endpoint_name = endpoint.into();

        // Create endpoint descriptor from component descriptor + endpoint name
        let descriptor = self
            .descriptor
            .clone()
            .endpoint(&endpoint_name)
            .expect("Endpoint name should be valid if component is valid");

        Endpoint {
            component: self.clone(),
            name: endpoint_name,
            is_static: self.is_static,
            descriptor,
        }
    }

    /// Create a component from a descriptor and runtime (for advanced use cases)
    pub fn from_descriptor(
        drt: Arc<DistributedRuntime>,
        descriptor: ComponentDescriptor,
        is_static: bool,
    ) -> Result<Self> {
        // Create namespace from descriptor segments
        let namespace = Namespace::from_segments(
            drt.clone(),
            &segments_to_refs(descriptor.inner().namespace_segments()),
            is_static,
        )?;

        build_component_with_descriptor(drt, namespace, descriptor, is_static)
    }

    /// Get namespace segments from the component's descriptor
    pub fn namespace_segments(&self) -> &[String] {
        self.descriptor.inner().namespace_segments()
    }

    /// Get the component name from the descriptor
    pub fn component_name(&self) -> &str {
        self.descriptor.inner().component().unwrap_or("")
    }

    /// Create component with explicit namespace segments and component name
    pub fn from_parts(
        drt: Arc<DistributedRuntime>,
        namespace_segments: &[&str],
        component_name: &str,
        is_static: bool,
    ) -> Result<Self> {
        let descriptor = ComponentDescriptor::from_component(namespace_segments, component_name)
            .map_err(|e| {
                descriptor_error(
                    "component",
                    component_name,
                    &format!("in namespace {:?}", namespace_segments),
                    e,
                )
            })?;

        Self::from_descriptor(drt, descriptor, is_static)
    }

    pub async fn list_instances(&self) -> anyhow::Result<Vec<Instance>> {
        let Some(etcd_client) = self.drt.etcd_client() else {
            return Ok(vec![]);
        };
        let mut out = vec![];
        // The extra slash is important to only list exact component matches, not substrings.
        for kv in etcd_client
            .kv_get_prefix(format!("{}/", self.etcd_root()))
            .await?
        {
            let val = match serde_json::from_slice::<Instance>(kv.value()) {
                Ok(val) => val,
                Err(err) => {
                    anyhow::bail!(
                        "Error converting etcd response to Instance: {err}. {}",
                        kv.value_str()?
                    );
                }
            };
            out.push(val);
        }
        Ok(out)
    }

    pub async fn scrape_stats(&self, timeout: Duration) -> Result<ServiceSet> {
        let service_name = self.service_name();
        let service_client = self.drt().service_client();
        service_client
            .collect_services(&service_name, timeout)
            .await
    }

    /// TODO
    ///
    /// This method will scrape the stats for all available services
    /// Returns a stream of `ServiceInfo` objects.
    /// This should be consumed by a `[tokio::time::timeout_at`] because each services
    /// will only respond once, but there is no way to know when all services have responded.
    pub async fn stats_stream(&self) -> Result<()> {
        unimplemented!("collect_stats")
    }

    pub fn service_builder(&self) -> service::ServiceConfigBuilder {
        service::ServiceConfigBuilder::from_component(self.clone())
    }
}

impl ComponentBuilder {
    pub fn from_runtime(drt: Arc<DistributedRuntime>) -> Self {
        Self::default().drt(drt)
    }
}

#[derive(Debug, Clone)]
pub struct Endpoint {
    /// Component reference for compatibility
    component: Component,

    /// Endpoint name (kept for backward compatibility)
    name: String,

    is_static: bool,

    // Endpoint descriptor representation (primary source of truth)
    descriptor: EndpointDescriptor,
}

impl Hash for Endpoint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.descriptor.hash(state);
        self.is_static.hash(state);
    }
}

impl PartialEq for Endpoint {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor && self.is_static == other.is_static
    }
}

impl Eq for Endpoint {}

impl DistributedRuntimeProvider for Endpoint {
    fn drt(&self) -> &DistributedRuntime {
        self.component.drt()
    }
}

impl RuntimeProvider for Endpoint {
    fn rt(&self) -> &Runtime {
        self.component.rt()
    }
}

impl Endpoint {
    pub fn id(&self) -> EndpointId {
        EndpointId {
            namespace: self.component.namespace().name().to_string(),
            component: self.component.name().to_string(),
            name: self.name().to_string(),
        }
    }

    pub fn name(&self) -> &str {
        // Keep backward compatibility - return the stored name
        &self.name
    }

    /// Get endpoint name from descriptor
    pub fn endpoint_name(&self) -> &str {
        self.descriptor.inner().endpoint().unwrap_or("")
    }

    /// Get the endpoint descriptor
    pub fn descriptor(&self) -> &EndpointDescriptor {
        &self.descriptor
    }

    pub fn component(&self) -> &Component {
        &self.component
    }

    // todo(ryan): deprecate this as we move to Discovery traits and Component Identifiers
    pub fn path(&self) -> String {
        format!(
            "{}/{}/{}",
            self.component.path(),
            ENDPOINT_KEYWORD,
            self.name
        )
    }

    /// The endpoint part of an instance path in etcd
    pub fn etcd_root(&self) -> String {
        let component_path = self.component.etcd_root();
        let endpoint_name = &self.name;
        format!("{component_path}/{endpoint_name}")
    }

    /// The endpoint as an EtcdPath object
    pub fn etcd_path(&self) -> EtcdPath {
        // Use descriptor for path generation, but maintain the same API
        EtcdPath::new_endpoint(
            &self.component.namespace().name(),
            &self.component.name(),
            &self.name,
        )
        .expect("Endpoint name and component name should be valid")
    }

    /// Generate etcd storage path using descriptor
    pub fn descriptor_etcd_path(&self) -> String {
        self.descriptor.inner().to_etcd_path()
    }

    /// The fully path of an instance in etcd
    pub fn etcd_path_with_lease_id(&self, lease_id: i64) -> String {
        let endpoint_root = self.etcd_root();
        if self.is_static {
            endpoint_root
        } else {
            format!("{endpoint_root}:{lease_id:x}")
        }
    }

    /// The endpoint as an EtcdPath object with lease ID
    pub fn etcd_path_object_with_lease_id(&self, lease_id: i64) -> EtcdPath {
        if self.is_static {
            self.etcd_path()
        } else {
            EtcdPath::new_endpoint_with_lease(
                &self.component.namespace().name(),
                &self.component.name(),
                &self.name,
                lease_id,
            )
            .expect("Endpoint name and component name should be valid")
        }
    }

    pub fn name_with_id(&self, lease_id: i64) -> String {
        if self.is_static {
            self.name.clone()
        } else {
            format!("{}-{:x}", self.name, lease_id)
        }
    }

    pub fn subject(&self) -> String {
        format!("{}.{}", self.component.service_name(), self.name)
    }

    /// Subject to an instance of the [Endpoint] with a specific lease id
    pub fn subject_to(&self, lease_id: i64) -> String {
        format!(
            "{}.{}",
            self.component.service_name(),
            self.name_with_id(lease_id)
        )
    }

    pub async fn client(&self) -> Result<client::Client> {
        if self.is_static {
            client::Client::new_static(self.clone()).await
        } else {
            client::Client::new_dynamic(self.clone()).await
        }
    }

    pub fn endpoint_builder(&self) -> endpoint::EndpointConfigBuilder {
        endpoint::EndpointConfigBuilder::from_endpoint(self.clone())
    }

    /// Create an endpoint from a descriptor and component (for advanced use cases)
    pub fn from_descriptor(
        component: Component,
        descriptor: EndpointDescriptor,
        is_static: bool,
    ) -> Result<Self> {
        // Extract endpoint name from descriptor for backward compatibility
        let endpoint_name = extract_name_or_empty(descriptor.inner().endpoint());

        Ok(Endpoint {
            component,
            name: endpoint_name,
            is_static,
            descriptor,
        })
    }

    /// Create an endpoint with explicit namespace, component, and endpoint names
    pub fn from_parts(
        drt: Arc<DistributedRuntime>,
        namespace_segments: &[&str],
        component_name: &str,
        endpoint_name: &str,
        is_static: bool,
    ) -> Result<Self> {
        let descriptor =
            EndpointDescriptor::from_endpoint(namespace_segments, component_name, endpoint_name)
                .map_err(|e| {
                    descriptor_error(
                        "endpoint",
                        endpoint_name,
                        &format!("in component '{}'", component_name),
                        e,
                    )
                })?;

        let component = Component::from_parts(drt, namespace_segments, component_name, is_static)?;

        Self::from_descriptor(component, descriptor, is_static)
    }

    /// Get namespace segments from the endpoint's descriptor
    pub fn namespace_segments(&self) -> &[String] {
        self.descriptor.inner().namespace_segments()
    }

    /// Get component name from the endpoint's descriptor
    pub fn component_name(&self) -> &str {
        self.descriptor.inner().component().unwrap_or("")
    }

    /// Create an instance descriptor for this endpoint with a specific lease ID
    /// This is used for dynamic endpoints that get registered with etcd
    pub fn create_instance_descriptor(&self, lease_id: i64) -> Result<InstanceDescriptor> {
        use crate::entity::descriptor::InstanceType;

        if self.is_static {
            // Static endpoints use Local instance type
            self.descriptor
                .clone()
                .instance(InstanceType::Local)
                .map_err(|e| anyhow::anyhow!("Failed to create static instance descriptor: {}", e))
        } else {
            // Dynamic endpoints use Distributed instance type with lease ID
            let instance_type = InstanceType::distributed(lease_id)
                .map_err(|e| anyhow::anyhow!("Invalid lease ID {}: {}", lease_id, e))?;
            self.descriptor
                .clone()
                .instance(instance_type)
                .map_err(|e| anyhow::anyhow!("Failed to create instance descriptor: {}", e))
        }
    }

    /// Get the URL representation of this endpoint using descriptor
    pub fn descriptor_url(&self) -> String {
        self.descriptor.inner().to_url()
    }

    /// Check if this endpoint has a valid descriptor representation
    pub fn validate_descriptor(&self) -> Result<()> {
        self.descriptor
            .inner()
            .validate()
            .map_err(|e| anyhow::anyhow!("Invalid endpoint descriptor: {}", e))
    }
}

#[derive(Builder, Clone, Validate)]
#[builder(pattern = "owned")]
pub struct Namespace {
    #[builder(private)]
    runtime: Arc<DistributedRuntime>,

    is_static: bool,

    // Entity descriptor representation (primary source of truth)
    #[builder(private)]
    descriptor: NamespaceDescriptor,
}

impl DistributedRuntimeProvider for Namespace {
    fn drt(&self) -> &DistributedRuntime {
        &self.runtime
    }
}

impl std::fmt::Debug for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Namespace {{ segments: {:?}; is_static: {} }}",
            self.descriptor.inner().namespace_segments(),
            self.is_static
        )
    }
}

impl RuntimeProvider for Namespace {
    fn rt(&self) -> &Runtime {
        self.runtime.rt()
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Namespace {
    pub(crate) fn new(runtime: DistributedRuntime, name: String, is_static: bool) -> Result<Self> {
        // Create descriptor from single segment
        let descriptor = NamespaceDescriptor::from_namespace(&[&name])
            .map_err(|e| descriptor_error("namespace", &name, "", e))?;

        build_namespace_with_descriptor(Arc::new(runtime), descriptor, is_static)
    }

    /// Get the namespace segments directly from the descriptor
    pub fn segments(&self) -> &[String] {
        self.descriptor.inner().namespace_segments()
    }

    /// Get parent namespace on-demand by creating a new namespace with all but the last segment
    pub fn parent(&self) -> Option<Namespace> {
        let segments = self.segments();
        if segments.len() <= 1 {
            return None; // Root namespace has no parent
        }

        // Create parent with all segments except the last one
        let parent_segments = &segments[..segments.len() - 1];
        let segment_refs: Vec<&str> = parent_segments.iter().map(|s| s.as_str()).collect();

        match NamespaceDescriptor::from_namespace(&segment_refs) {
            Ok(parent_descriptor) => build_namespace_with_descriptor(
                self.runtime.clone(),
                parent_descriptor,
                self.is_static,
            )
            .ok(),
            Err(_) => None, // Invalid parent segments
        }
    }

    /// Get the leaf name (last segment) of this namespace
    pub fn leaf_name(&self) -> String {
        self.segments().last().cloned().unwrap_or_default()
    }

    /// Create a [`Component`] in the namespace who's endpoints can be discovered with etcd
    pub fn component(&self, name: impl Into<String>) -> Result<Component> {
        let component_name = name.into();

        // Create component descriptor from namespace + component name
        let descriptor = ComponentDescriptor::from_component(
            &segments_to_refs(self.segments()),
            &component_name,
        )
        .map_err(|e| descriptor_error("component", &component_name, "", e))?;

        build_component_with_descriptor(
            self.runtime.clone(),
            self.clone(),
            descriptor,
            self.is_static,
        )
    }

    /// Create a child [`Namespace`] by appending a new segment
    pub fn namespace(&self, name: impl Into<String>) -> Result<Namespace> {
        let child_name = name.into();

        // Build child descriptor by combining current segments with new name
        let mut child_segments = self.segments().to_vec();
        child_segments.push(child_name.clone());
        let descriptor = NamespaceDescriptor::from_namespace(&segments_to_refs(&child_segments))
            .map_err(|e| descriptor_error("namespace", &child_name, "as child", e))?;

        build_namespace_with_descriptor(self.runtime.clone(), descriptor, self.is_static)
    }

    pub fn etcd_path(&self) -> String {
        // Use descriptor's etcd path generation
        self.descriptor.inner().to_etcd_path()
    }

    /// Get the full dotted namespace name (e.g., "prod.api.v1")
    pub fn name(&self) -> String {
        self.segments().join(".")
    }

    /// Get the namespace descriptor
    pub fn descriptor(&self) -> &NamespaceDescriptor {
        &self.descriptor
    }

    /// Create a namespace from segments (for advanced use cases)
    pub fn from_segments(
        runtime: Arc<DistributedRuntime>,
        segments: &[&str],
        is_static: bool,
    ) -> Result<Self> {
        let descriptor = NamespaceDescriptor::from_namespace(segments).map_err(|e| {
            descriptor_error("namespace", &format!("{:?}", segments), "from segments", e)
        })?;

        build_namespace_with_descriptor(runtime, descriptor, is_static)
    }
}

// Helper functions to reduce duplication (private, doesn't affect public API)

/// Convert namespace segments to string references for descriptor creation
fn segments_to_refs(segments: &[String]) -> Vec<&str> {
    segments.iter().map(|s| s.as_str()).collect()
}

/// Extract name from descriptor with empty string fallback
fn extract_name_or_empty(name_opt: Option<&str>) -> String {
    name_opt.unwrap_or("").to_string()
}

/// Create error for invalid descriptor with context
fn descriptor_error(
    entity_type: &str,
    name: &str,
    context: &str,
    error: DescriptorError,
) -> anyhow::Error {
    anyhow::anyhow!("Invalid {} '{}' {}: {}", entity_type, name, context, error)
}

/// Build namespace with common pattern
fn build_namespace_with_descriptor(
    runtime: Arc<DistributedRuntime>,
    descriptor: NamespaceDescriptor,
    is_static: bool,
) -> Result<Namespace> {
    Ok(NamespaceBuilder::default()
        .runtime(runtime)
        .is_static(is_static)
        .descriptor(descriptor)
        .build()?)
}

/// Build component with common pattern
fn build_component_with_descriptor(
    drt: Arc<DistributedRuntime>,
    namespace: Namespace,
    descriptor: ComponentDescriptor,
    is_static: bool,
) -> Result<Component> {
    Ok(ComponentBuilder::from_runtime(drt)
        .namespace(namespace)
        .is_static(is_static)
        .descriptor(descriptor)
        .build()?)
}

/// Format namespace/component path for display
fn format_entity_path(namespace_name: &str, entity_name: &str) -> String {
    format!("{}/{}", namespace_name, entity_name)
}

/// Format service name from namespace and component
fn format_service_name(namespace_name: &str, component_name: &str) -> String {
    let service_name = format!("{}_{}", namespace_name, component_name);
    Slug::slugify(&service_name).to_string()
}

// Custom validator function
fn validate_allowed_chars(input: &str) -> Result<(), ValidationError> {
    // Define the allowed character set using a regex
    let regex = regex::Regex::new(r"^[a-z0-9-_]+$").unwrap();

    if regex.is_match(input) {
        Ok(())
    } else {
        Err(ValidationError::new("invalid_characters"))
    }
}
