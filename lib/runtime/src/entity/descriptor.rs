// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure data descriptors for component identification - Version 2
//!
//! This module implements a clean descriptor system for Dynamo's distributed component
//! management. Descriptors are immutable data structures that represent component paths
//! and identities using both query parameter format (user-facing) and keyword format (etcd storage).
//!
//! # Key Features
//!
//! - **Dual Format Support**: Query parameters for users, keywords for etcd storage
//! - **Nested Namespaces**: Path-based namespace hierarchy
//! - **Clean URLs**: Shorter keywords and prettier query format
//! - **No Static Instances**: Only Distributed(i64) and Local instances
//! - **Optimized Transitions**: Minimal validation overhead when building from validated descriptors
//! - **Type-Safe Builder**: Fluent API with compile-time guarantees
//!
//! # URL Formats
//!
//! ## Query Format (User-Facing, Default)
//! ```text
//! dynamo://prod/api/v1?component=gateway&endpoint=http&instance=1234&path=config&path=v1
//! dynamo://prod/api/v1?component=gateway&endpoint=http&instance=local
//! ```
//!
//! ## Keyword Format (etcd Storage)
//! ```text
//! dynamo://prod/api/v1/_c/gateway/_e/http:1234/_path/config/v1
//! dynamo://prod/api/v1/_c/gateway/_e/http:local
//! ```

use std::fmt;
use std::str::FromStr;

/// Root path for all dynamo URLs
pub const DYNAMO_SCHEME: &str = "dynamo://";

// etcd storage keywords (short versions)
pub const COMPONENT_KEYWORD: &str = "_c";
pub const ENDPOINT_KEYWORD: &str = "_e";
pub const PATH_KEYWORD: &str = "_path";

/// Instance type for endpoints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstanceType {
    /// Distributed instance with lease ID
    Distributed(i64),
    /// Local instance (direct AsyncEngine access)
    Local,
}

impl fmt::Display for InstanceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstanceType::Distributed(id) => write!(f, "{:x}", id),
            InstanceType::Local => write!(f, "local"),
        }
    }
}

impl FromStr for InstanceType {
    type Err = DescriptorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "local" => Ok(InstanceType::Local),
            hex_str => {
                let id = i64::from_str_radix(hex_str, 16)
                    .map_err(|_| DescriptorError::InvalidInstanceId(hex_str.to_string()))?;
                Ok(InstanceType::Distributed(id))
            }
        }
    }
}

/// Errors that can occur during descriptor operations
#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum DescriptorError {
    #[error("Path must start with '{}'", DYNAMO_SCHEME)]
    InvalidPrefix,
    #[error("Invalid namespace segment: {0}")]
    InvalidNamespace(String),
    #[error("Invalid component name: {0}")]
    InvalidComponent(String),
    #[error("Invalid endpoint name: {0}")]
    InvalidEndpoint(String),
    #[error("Invalid path segment: {0}")]
    InvalidPathSegment(String),
    #[error("Empty namespace not allowed")]
    EmptyNamespace,
    #[error("Empty component name not allowed")]
    EmptyComponent,
    #[error("Empty endpoint name not allowed")]
    EmptyEndpoint,
    #[error("Invalid instance ID format: {0}")]
    InvalidInstanceId(String),
    #[error("Endpoint requires component to be present")]
    EndpointWithoutComponent,
    #[error("Instance ID can only be attached to endpoints")]
    InstanceWithoutEndpoint,
    #[error("Invalid URL format: {0}")]
    InvalidUrl(String),
    #[error("Unknown query parameter: {0}")]
    UnknownQueryParameter(String),
}

/// Core descriptor for Dynamo components
///
/// This is the main type that handles both user-facing query format
/// and etcd storage keyword format.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EntityDescriptor {
    /// Namespace segments (e.g., ["prod", "api", "v1"])
    namespace_segments: Vec<String>,
    /// Optional component name
    component: Option<String>,
    /// Optional endpoint name (requires component)
    endpoint: Option<String>,
    /// Optional instance type (requires endpoint)
    instance: Option<InstanceType>,
    /// Optional path segments for data storage
    path_segments: Vec<String>,
}

impl EntityDescriptor {
    /// Create a new builder for constructing descriptors
    pub fn builder() -> EntityDescriptorBuilder {
        EntityDescriptorBuilder::new()
    }

    /// Parse from either query format or keyword format
    pub fn parse(input: &str) -> Result<Self, DescriptorError> {
        if input.contains('?') {
            Self::parse_query_format(input)
        } else {
            Self::parse_keyword_format(input)
        }
    }

    /// Generate user-facing URL with query parameters (default format)
    pub fn to_url(&self) -> String {
        let mut url = format!("{}{}", DYNAMO_SCHEME, self.namespace_segments.join("/"));

        let mut query_parts = Vec::new();

        // Add query parameters in consistent order
        if let Some(ref component) = self.component {
            query_parts.push(format!("component={}", component));
        }

        if let Some(ref endpoint) = self.endpoint {
            query_parts.push(format!("endpoint={}", endpoint));
        }

        if let Some(ref instance) = self.instance {
            query_parts.push(format!("instance={}", instance));
        }

        for segment in &self.path_segments {
            query_parts.push(format!("path={}", segment));
        }

        if !query_parts.is_empty() {
            url.push('?');
            url.push_str(&query_parts.join("&"));
        }

        url
    }

    /// Generate etcd storage path with keywords
    pub fn to_etcd_path(&self) -> String {
        let mut path = format!("{}{}", DYNAMO_SCHEME, self.namespace_segments.join("/"));

        if let Some(ref component) = self.component {
            path.push_str(&format!("/{}/{}", COMPONENT_KEYWORD, component));

            if let Some(ref endpoint) = self.endpoint {
                path.push_str(&format!("/{}/{}", ENDPOINT_KEYWORD, endpoint));

                if let Some(ref instance) = self.instance {
                    path.push_str(&format!(":{}", instance));
                }
            }
        }

        if !self.path_segments.is_empty() {
            path.push_str(&format!("/{}", PATH_KEYWORD));
            for segment in &self.path_segments {
                path.push_str(&format!("/{}", segment));
            }
        }

        path
    }

    /// Get namespace segments
    pub fn namespace_segments(&self) -> &[String] {
        &self.namespace_segments
    }

    /// Get component name if present
    pub fn component(&self) -> Option<&str> {
        self.component.as_deref()
    }

    /// Get endpoint name if present
    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    /// Get instance type if present
    pub fn instance(&self) -> Option<&InstanceType> {
        self.instance.as_ref()
    }

    /// Get path segments
    pub fn path_segments(&self) -> &[String] {
        &self.path_segments
    }

    /// Validate the descriptor
    pub fn validate(&self) -> Result<(), DescriptorError> {
        // Validate namespace
        if self.namespace_segments.is_empty() {
            return Err(DescriptorError::EmptyNamespace);
        }

        for segment in &self.namespace_segments {
            validate_name_segment(segment, "namespace")?;
        }

        // Validate component
        if let Some(ref component) = self.component {
            validate_name(component, "component")?;
        }

        // Validate endpoint (requires component)
        if let Some(ref endpoint) = self.endpoint {
            if self.component.is_none() {
                return Err(DescriptorError::EndpointWithoutComponent);
            }
            validate_name(endpoint, "endpoint")?;
        }

        // Validate instance (requires endpoint)
        if self.instance.is_some() && self.endpoint.is_none() {
            return Err(DescriptorError::InstanceWithoutEndpoint);
        }

        // Validate path segments
        for segment in &self.path_segments {
            validate_name(segment, "path segment")?;
        }

        Ok(())
    }

    /// Parse query parameter format
    fn parse_query_format(input: &str) -> Result<Self, DescriptorError> {
        // Simple URL parsing for query parameters
        let url_parts: Vec<&str> = input.splitn(2, '?').collect();

        if url_parts.len() != 2 {
            return Err(DescriptorError::InvalidUrl(
                "Missing query parameters".to_string(),
            ));
        }

        let path_part = url_parts[0];
        let query_part = url_parts[1];

        // Validate and extract namespace from path
        if !path_part.starts_with(DYNAMO_SCHEME) {
            return Err(DescriptorError::InvalidPrefix);
        }

        let namespace_path = &path_part[DYNAMO_SCHEME.len()..];
        let namespace_segments: Vec<String> = namespace_path
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        if namespace_segments.is_empty() {
            return Err(DescriptorError::EmptyNamespace);
        }

        // Parse query parameters
        let mut component = None;
        let mut endpoint = None;
        let mut instance = None;
        let mut path_segments = Vec::new();

        for pair in query_part.split('&') {
            let kv: Vec<&str> = pair.splitn(2, '=').collect();
            if kv.len() != 2 {
                continue;
            }

            let key = kv[0];
            let value = kv[1];

            match key {
                "component" => component = Some(value.to_string()),
                "endpoint" => endpoint = Some(value.to_string()),
                "instance" => instance = Some(value.parse()?),
                "path" => path_segments.push(value.to_string()),
                _ => return Err(DescriptorError::UnknownQueryParameter(key.to_string())),
            }
        }

        let descriptor = EntityDescriptor {
            namespace_segments,
            component,
            endpoint,
            instance,
            path_segments,
        };

        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Parse keyword format (etcd storage)
    fn parse_keyword_format(input: &str) -> Result<Self, DescriptorError> {
        if !input.starts_with(DYNAMO_SCHEME) {
            return Err(DescriptorError::InvalidPrefix);
        }

        let path_without_prefix = &input[DYNAMO_SCHEME.len()..];
        let segments: Vec<&str> = path_without_prefix.split('/').collect();

        if segments.is_empty() || segments[0].is_empty() {
            return Err(DescriptorError::EmptyNamespace);
        }

        let mut namespace_segments = Vec::new();
        let mut component = None;
        let mut endpoint = None;
        let mut instance = None;
        let mut path_segments = Vec::new();

        let mut i = 0;

        // Parse namespace segments until we hit a keyword
        while i < segments.len()
            && !matches!(
                segments[i],
                COMPONENT_KEYWORD | ENDPOINT_KEYWORD | PATH_KEYWORD
            )
        {
            namespace_segments.push(segments[i].to_string());
            i += 1;
        }

        if namespace_segments.is_empty() {
            return Err(DescriptorError::EmptyNamespace);
        }

        // Parse remaining segments with keywords
        while i < segments.len() {
            match segments[i] {
                COMPONENT_KEYWORD => {
                    if i + 1 >= segments.len() {
                        return Err(DescriptorError::EmptyComponent);
                    }
                    component = Some(segments[i + 1].to_string());
                    i += 2;
                }
                ENDPOINT_KEYWORD => {
                    if component.is_none() {
                        return Err(DescriptorError::EndpointWithoutComponent);
                    }
                    if i + 1 >= segments.len() {
                        return Err(DescriptorError::EmptyEndpoint);
                    }

                    let endpoint_segment = segments[i + 1];

                    // Check for instance ID suffix (:instance)
                    if let Some(colon_pos) = endpoint_segment.find(':') {
                        let endpoint_name = &endpoint_segment[..colon_pos];
                        let instance_str = &endpoint_segment[colon_pos + 1..];

                        endpoint = Some(endpoint_name.to_string());
                        instance = Some(instance_str.parse()?);
                    } else {
                        endpoint = Some(endpoint_segment.to_string());
                    }

                    i += 2;
                }
                PATH_KEYWORD => {
                    i += 1;
                    while i < segments.len() {
                        path_segments.push(segments[i].to_string());
                        i += 1;
                    }
                }
                _ => {
                    return Err(DescriptorError::InvalidUrl(format!(
                        "Unknown keyword: {}",
                        segments[i]
                    )));
                }
            }
        }

        let descriptor = EntityDescriptor {
            namespace_segments,
            component,
            endpoint,
            instance,
            path_segments,
        };

        descriptor.validate()?;
        Ok(descriptor)
    }
}

impl fmt::Display for EntityDescriptor {
    /// Default to query format (prettier)
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_url())
    }
}

impl FromStr for EntityDescriptor {
    type Err = DescriptorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

// Specialized descriptor types with specific validation rules

/// Namespace-only descriptor (only namespace segments allowed)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamespaceDescriptor(EntityDescriptor);

impl NamespaceDescriptor {
    /// Create from EntityDescriptor with validation
    pub fn new(desc: EntityDescriptor) -> Result<Self, DescriptorError> {
        Self::validate_namespace_only(&desc)?;
        Ok(Self(desc))
    }

    /// Create from namespace segments
    pub fn from_namespace(segments: &[&str]) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::builder().namespace(segments).build()?;
        Self::new(desc)
    }

    /// Parse from string with namespace-only validation
    pub fn parse(input: &str) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::parse(input)?;
        Self::new(desc)
    }

    /// Get the inner EntityDescriptor
    pub fn inner(&self) -> &EntityDescriptor {
        &self.0
    }

    /// Add a component to create a ComponentDescriptor
    pub fn component(self, component: &str) -> Result<ComponentDescriptor, DescriptorError> {
        // Validate only the new component field
        validate_name(component, "component")?;

        let desc = EntityDescriptor::builder()
            .namespace_segments(self.0.namespace_segments())
            .component(component)
            .into(); // Skip validation since we know namespace is valid and component is validated above
        ComponentDescriptor::new(desc)
    }

    /// Add path segments to create a PathDescriptor
    pub fn path(self, path_segments: &[&str]) -> Result<PathDescriptor, DescriptorError> {
        // Validate only the new path segments
        for segment in path_segments {
            validate_name(segment, "path segment")?;
        }

        let desc = EntityDescriptor::builder()
            .namespace_segments(self.0.namespace_segments())
            .path_segments(path_segments)
            .into(); // Skip validation since we know namespace is valid and path segments are validated above
        PathDescriptor::new(desc)
    }

    /// Validate that descriptor only has namespace
    fn validate_namespace_only(desc: &EntityDescriptor) -> Result<(), DescriptorError> {
        if desc.component.is_some() {
            return Err(DescriptorError::InvalidUrl(
                "NamespaceDescriptor cannot have component".to_string(),
            ));
        }
        if desc.endpoint.is_some() {
            return Err(DescriptorError::InvalidUrl(
                "NamespaceDescriptor cannot have endpoint".to_string(),
            ));
        }
        if desc.instance.is_some() {
            return Err(DescriptorError::InvalidUrl(
                "NamespaceDescriptor cannot have instance".to_string(),
            ));
        }
        if !desc.path_segments.is_empty() {
            return Err(DescriptorError::InvalidUrl(
                "NamespaceDescriptor cannot have path segments".to_string(),
            ));
        }
        Ok(())
    }
}

/// Component descriptor (namespace + component, no endpoint/instance/path)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComponentDescriptor(EntityDescriptor);

impl ComponentDescriptor {
    /// Create from EntityDescriptor with validation
    pub fn new(desc: EntityDescriptor) -> Result<Self, DescriptorError> {
        Self::validate_component_only(&desc)?;
        Ok(Self(desc))
    }

    /// Create from namespace and component
    pub fn from_component(namespace: &[&str], component: &str) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::builder()
            .namespace(namespace)
            .component(component)
            .build()?;
        Self::new(desc)
    }

    /// Parse from string with component-only validation
    pub fn parse(input: &str) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::parse(input)?;
        Self::new(desc)
    }

    /// Get the inner EntityDescriptor
    pub fn inner(&self) -> &EntityDescriptor {
        &self.0
    }

    /// Add an endpoint to create an EndpointDescriptor
    pub fn endpoint(self, endpoint: &str) -> Result<EndpointDescriptor, DescriptorError> {
        // Validate only the new endpoint field
        validate_name(endpoint, "endpoint")?;

        let desc = EntityDescriptor::builder()
            .namespace_segments(self.0.namespace_segments())
            .component(self.0.component().unwrap()) // Safe because ComponentDescriptor always has component
            .endpoint(endpoint)
            .into(); // Skip validation since we know namespace/component are valid and endpoint is validated above
        EndpointDescriptor::new(desc)
    }

    /// Add path segments to create a PathDescriptor
    pub fn path(self, path_segments: &[&str]) -> Result<PathDescriptor, DescriptorError> {
        // Validate only the new path segments
        for segment in path_segments {
            validate_name(segment, "path segment")?;
        }

        let desc = EntityDescriptor::builder()
            .namespace_segments(self.0.namespace_segments())
            .component(self.0.component().unwrap()) // Safe because ComponentDescriptor always has component
            .path_segments(path_segments)
            .into(); // Skip validation since we know namespace/component are valid and path segments are validated above
        PathDescriptor::new(desc)
    }

    /// Validate that descriptor has namespace + component only
    fn validate_component_only(desc: &EntityDescriptor) -> Result<(), DescriptorError> {
        if desc.component.is_none() {
            return Err(DescriptorError::EmptyComponent);
        }
        if desc.endpoint.is_some() {
            return Err(DescriptorError::InvalidUrl(
                "ComponentDescriptor cannot have endpoint".to_string(),
            ));
        }
        if desc.instance.is_some() {
            return Err(DescriptorError::InvalidUrl(
                "ComponentDescriptor cannot have instance".to_string(),
            ));
        }
        if !desc.path_segments.is_empty() {
            return Err(DescriptorError::InvalidUrl(
                "ComponentDescriptor cannot have path segments".to_string(),
            ));
        }
        Ok(())
    }
}

/// Endpoint descriptor (namespace + component + endpoint, no instance/path)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EndpointDescriptor(EntityDescriptor);

impl EndpointDescriptor {
    /// Create from EntityDescriptor with validation
    pub fn new(desc: EntityDescriptor) -> Result<Self, DescriptorError> {
        Self::validate_endpoint_only(&desc)?;
        Ok(Self(desc))
    }

    /// Create from namespace, component, and endpoint
    pub fn from_endpoint(
        namespace: &[&str],
        component: &str,
        endpoint: &str,
    ) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::builder()
            .namespace(namespace)
            .component(component)
            .endpoint(endpoint)
            .build()?;
        Self::new(desc)
    }

    /// Parse from string with endpoint-only validation
    pub fn parse(input: &str) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::parse(input)?;
        Self::new(desc)
    }

    /// Get the inner EntityDescriptor
    pub fn inner(&self) -> &EntityDescriptor {
        &self.0
    }

    /// Add an instance to create an InstanceDescriptor
    pub fn instance(self, instance: InstanceType) -> Result<InstanceDescriptor, DescriptorError> {
        // No validation needed for InstanceType - it's an enum with valid variants
        let desc = EntityDescriptor::builder()
            .namespace_segments(self.0.namespace_segments())
            .component(self.0.component().unwrap()) // Safe because EndpointDescriptor always has component
            .endpoint(self.0.endpoint().unwrap()) // Safe because EndpointDescriptor always has endpoint
            .instance(instance)
            .into(); // Skip validation since we know all existing fields are valid
        InstanceDescriptor::new(desc)
    }

    /// Add path segments to create a PathDescriptor
    pub fn path(self, path_segments: &[&str]) -> Result<PathDescriptor, DescriptorError> {
        // Validate only the new path segments
        for segment in path_segments {
            validate_name(segment, "path segment")?;
        }

        let desc = EntityDescriptor::builder()
            .namespace_segments(self.0.namespace_segments())
            .component(self.0.component().unwrap()) // Safe because EndpointDescriptor always has component
            .endpoint(self.0.endpoint().unwrap()) // Safe because EndpointDescriptor always has endpoint
            .path_segments(path_segments)
            .into(); // Skip validation since we know existing fields are valid and path segments are validated above
        PathDescriptor::new(desc)
    }

    /// Validate that descriptor has namespace + component + endpoint only
    fn validate_endpoint_only(desc: &EntityDescriptor) -> Result<(), DescriptorError> {
        if desc.component.is_none() {
            return Err(DescriptorError::EmptyComponent);
        }
        if desc.endpoint.is_none() {
            return Err(DescriptorError::EmptyEndpoint);
        }
        if desc.instance.is_some() {
            return Err(DescriptorError::InvalidUrl(
                "EndpointDescriptor cannot have instance".to_string(),
            ));
        }
        if !desc.path_segments.is_empty() {
            return Err(DescriptorError::InvalidUrl(
                "EndpointDescriptor cannot have path segments".to_string(),
            ));
        }
        Ok(())
    }
}

/// Instance descriptor (namespace + component + endpoint + instance, no path)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstanceDescriptor(EntityDescriptor);

impl InstanceDescriptor {
    /// Create from EntityDescriptor with validation
    pub fn new(desc: EntityDescriptor) -> Result<Self, DescriptorError> {
        Self::validate_instance_only(&desc)?;
        Ok(Self(desc))
    }

    /// Create from namespace, component, endpoint, and instance
    pub fn from_instance(
        namespace: &[&str],
        component: &str,
        endpoint: &str,
        instance: InstanceType,
    ) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::builder()
            .namespace(namespace)
            .component(component)
            .endpoint(endpoint)
            .instance(instance)
            .build()?;
        Self::new(desc)
    }

    /// Parse from string with instance-only validation
    pub fn parse(input: &str) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::parse(input)?;
        Self::new(desc)
    }

    /// Get the inner EntityDescriptor
    pub fn inner(&self) -> &EntityDescriptor {
        &self.0
    }

    /// Add path segments to create a PathDescriptor
    pub fn path(self, path_segments: &[&str]) -> Result<PathDescriptor, DescriptorError> {
        // Validate only the new path segments
        for segment in path_segments {
            validate_name(segment, "path segment")?;
        }

        let desc = EntityDescriptor::builder()
            .namespace_segments(self.0.namespace_segments())
            .component(self.0.component().unwrap()) // Safe because InstanceDescriptor always has component
            .endpoint(self.0.endpoint().unwrap()) // Safe because InstanceDescriptor always has endpoint
            .instance(*self.0.instance().unwrap()) // Safe because InstanceDescriptor always has instance
            .path_segments(path_segments)
            .into(); // Skip validation since we know existing fields are valid and path segments are validated above
        PathDescriptor::new(desc)
    }

    /// Validate that descriptor has namespace + component + endpoint + instance only
    fn validate_instance_only(desc: &EntityDescriptor) -> Result<(), DescriptorError> {
        if desc.component.is_none() {
            return Err(DescriptorError::EmptyComponent);
        }
        if desc.endpoint.is_none() {
            return Err(DescriptorError::EmptyEndpoint);
        }
        if desc.instance.is_none() {
            return Err(DescriptorError::InvalidInstanceId(
                "Instance required for InstanceDescriptor".to_string(),
            ));
        }
        if !desc.path_segments.is_empty() {
            return Err(DescriptorError::InvalidUrl(
                "InstanceDescriptor cannot have path segments".to_string(),
            ));
        }
        Ok(())
    }
}

/// Path descriptor (can have any combination + path segments)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathDescriptor(EntityDescriptor);

impl PathDescriptor {
    /// Create from EntityDescriptor with validation
    pub fn new(desc: EntityDescriptor) -> Result<Self, DescriptorError> {
        Self::validate_path_required(&desc)?;
        Ok(Self(desc))
    }

    /// Create from namespace and path segments
    pub fn from_namespace_path(
        namespace: &[&str],
        path_segments: &[&str],
    ) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::builder()
            .namespace(namespace)
            .path_segments(path_segments)
            .build()?;
        Self::new(desc)
    }

    /// Create from component and path segments
    pub fn from_component_path(
        namespace: &[&str],
        component: &str,
        path_segments: &[&str],
    ) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::builder()
            .namespace(namespace)
            .component(component)
            .path_segments(path_segments)
            .build()?;
        Self::new(desc)
    }

    /// Create from endpoint and path segments
    pub fn from_endpoint_path(
        namespace: &[&str],
        component: &str,
        endpoint: &str,
        path_segments: &[&str],
    ) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::builder()
            .namespace(namespace)
            .component(component)
            .endpoint(endpoint)
            .path_segments(path_segments)
            .build()?;
        Self::new(desc)
    }

    /// Create from instance and path segments
    pub fn from_instance_path(
        namespace: &[&str],
        component: &str,
        endpoint: &str,
        instance: InstanceType,
        path_segments: &[&str],
    ) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::builder()
            .namespace(namespace)
            .component(component)
            .endpoint(endpoint)
            .instance(instance)
            .path_segments(path_segments)
            .build()?;
        Self::new(desc)
    }

    /// Parse from string with path validation
    pub fn parse(input: &str) -> Result<Self, DescriptorError> {
        let desc = EntityDescriptor::parse(input)?;
        Self::new(desc)
    }

    /// Get the inner EntityDescriptor
    pub fn inner(&self) -> &EntityDescriptor {
        &self.0
    }

    /// Add more path segments to create a new PathDescriptor
    pub fn path(self, additional_segments: &[&str]) -> Result<PathDescriptor, DescriptorError> {
        // Validate only the new path segments
        for segment in additional_segments {
            validate_name(segment, "path segment")?;
        }

        let mut builder = EntityDescriptor::builder()
            .namespace_segments(self.0.namespace_segments())
            .path_segments_owned(self.0.path_segments())
            .append_path(additional_segments);

        // Add component if present
        if let Some(component) = self.0.component() {
            builder = builder.component(component);
        }

        // Add endpoint if present
        if let Some(endpoint) = self.0.endpoint() {
            builder = builder.endpoint(endpoint);
        }

        // Add instance if present
        if let Some(instance) = self.0.instance() {
            builder = builder.instance(*instance);
        }

        let desc = builder.into(); // Skip validation since we know existing fields are valid and new segments are validated above
        PathDescriptor::new(desc)
    }

    /// Validate that descriptor has path segments
    fn validate_path_required(desc: &EntityDescriptor) -> Result<(), DescriptorError> {
        if desc.path_segments.is_empty() {
            return Err(DescriptorError::InvalidUrl(
                "PathDescriptor requires path segments".to_string(),
            ));
        }
        Ok(())
    }
}

// Implement Display and FromStr for all descriptor types
macro_rules! impl_descriptor_traits {
    ($type:ty) => {
        impl fmt::Display for $type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl FromStr for $type {
            type Err = DescriptorError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::parse(s)
            }
        }
    };
}

impl_descriptor_traits!(NamespaceDescriptor);
impl_descriptor_traits!(ComponentDescriptor);
impl_descriptor_traits!(EndpointDescriptor);
impl_descriptor_traits!(InstanceDescriptor);
impl_descriptor_traits!(PathDescriptor);

// Internal descriptor types for system use

/// Internal namespace descriptor specifically for "_internal" namespace
/// This provides type safety for internal system operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InternalNamespace(NamespaceDescriptor);

impl InternalNamespace {
    /// Create the internal namespace descriptor
    pub fn new() -> Result<Self, DescriptorError> {
        // Use builder with .into() to bypass validation for the special _internal namespace
        let desc = EntityDescriptor::builder().namespace(&["_internal"]).into(); // This calls build_unchecked to allow _internal namespace

        // Create NamespaceDescriptor directly since we know it's valid
        let ns_desc = NamespaceDescriptor(desc);
        Ok(Self(ns_desc))
    }

    /// Get the inner NamespaceDescriptor
    pub fn inner(&self) -> &NamespaceDescriptor {
        &self.0
    }

    /// Get the EntityDescriptor
    pub fn entity(&self) -> &EntityDescriptor {
        self.0.inner()
    }

    /// Add a component to create a ComponentDescriptor (normal validation rules apply)
    pub fn component(self, component: &str) -> Result<ComponentDescriptor, DescriptorError> {
        // Use normal validation - no underscore prefixes allowed for components
        validate_name(component, "component")?;
        let desc = EntityDescriptor::builder()
            .namespace_segments(self.0.inner().namespace_segments())
            .component(component)
            .into(); // Skip validation since we already validated the component
        Ok(ComponentDescriptor(desc))
    }

    /// Add path segments to create a PathDescriptor (normal validation rules apply)
    pub fn path(self, path_segments: &[&str]) -> Result<PathDescriptor, DescriptorError> {
        // Validate path segments using normal rules
        for segment in path_segments {
            validate_name(segment, "path segment")?;
        }

        // Use build_unchecked to bypass namespace validation (allow _internal namespace)
        let desc = EntityDescriptor::builder()
            .namespace(&["_internal"])
            .path_segments(path_segments)
            .into(); // This calls build_unchecked to allow _internal namespace
        Ok(PathDescriptor(desc))
    }
}

impl Default for InternalNamespace {
    fn default() -> Self {
        Self::new().expect("_internal namespace should always be valid")
    }
}

impl fmt::Display for InternalNamespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for InternalNamespace {
    type Err = DescriptorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Parse as EntityDescriptor first to handle _internal namespace
        let entity_desc = EntityDescriptor::parse(s)?;
        if entity_desc.namespace_segments() != ["_internal"] {
            return Err(DescriptorError::InvalidNamespace(
                "InternalNamespace must have '_internal' namespace".to_string(),
            ));
        }

        // Check that it's namespace-only (no component, endpoint, instance, or path)
        if entity_desc.component().is_some()
            || entity_desc.endpoint().is_some()
            || entity_desc.instance().is_some()
            || !entity_desc.path_segments().is_empty()
        {
            return Err(DescriptorError::InvalidUrl(
                "InternalNamespace must contain only namespace".to_string(),
            ));
        }

        // Create NamespaceDescriptor directly
        let ns_desc = NamespaceDescriptor(entity_desc);
        Ok(Self(ns_desc))
    }
}

// TryFrom implementations for conversions from EntityDescriptor

impl TryFrom<EntityDescriptor> for InternalNamespace {
    type Error = DescriptorError;

    fn try_from(desc: EntityDescriptor) -> Result<Self, Self::Error> {
        // Check that it has _internal namespace
        if desc.namespace_segments() != ["_internal"] {
            return Err(DescriptorError::InvalidNamespace(
                "InternalNamespace must have '_internal' namespace".to_string(),
            ));
        }

        // Check that it's namespace-only (no component, endpoint, instance, or path)
        if desc.component().is_some()
            || desc.endpoint().is_some()
            || desc.instance().is_some()
            || !desc.path_segments().is_empty()
        {
            return Err(DescriptorError::InvalidUrl(
                "InternalNamespace must contain only namespace".to_string(),
            ));
        }

        // Create NamespaceDescriptor directly since we know it's valid
        let ns_desc = NamespaceDescriptor(desc);
        Ok(Self(ns_desc))
    }
}

// From implementations for conversion to EntityDescriptor

impl From<InternalNamespace> for EntityDescriptor {
    fn from(desc: InternalNamespace) -> Self {
        desc.0.into()
    }
}

// From implementations for conversion to base descriptor types

impl From<InternalNamespace> for NamespaceDescriptor {
    fn from(desc: InternalNamespace) -> Self {
        desc.0
    }
}

// TryFrom implementations for convenient conversion from EntityDescriptor

impl TryFrom<EntityDescriptor> for NamespaceDescriptor {
    type Error = DescriptorError;

    fn try_from(desc: EntityDescriptor) -> Result<Self, Self::Error> {
        Self::new(desc)
    }
}

impl TryFrom<EntityDescriptor> for ComponentDescriptor {
    type Error = DescriptorError;

    fn try_from(desc: EntityDescriptor) -> Result<Self, Self::Error> {
        Self::new(desc)
    }
}

impl TryFrom<EntityDescriptor> for EndpointDescriptor {
    type Error = DescriptorError;

    fn try_from(desc: EntityDescriptor) -> Result<Self, Self::Error> {
        Self::new(desc)
    }
}

impl TryFrom<EntityDescriptor> for InstanceDescriptor {
    type Error = DescriptorError;

    fn try_from(desc: EntityDescriptor) -> Result<Self, Self::Error> {
        Self::new(desc)
    }
}

impl TryFrom<EntityDescriptor> for PathDescriptor {
    type Error = DescriptorError;

    fn try_from(desc: EntityDescriptor) -> Result<Self, Self::Error> {
        Self::new(desc)
    }
}

// From implementations for infallible conversion to EntityDescriptor

impl From<NamespaceDescriptor> for EntityDescriptor {
    fn from(desc: NamespaceDescriptor) -> Self {
        desc.0
    }
}

impl From<ComponentDescriptor> for EntityDescriptor {
    fn from(desc: ComponentDescriptor) -> Self {
        desc.0
    }
}

impl From<EndpointDescriptor> for EntityDescriptor {
    fn from(desc: EndpointDescriptor) -> Self {
        desc.0
    }
}

impl From<InstanceDescriptor> for EntityDescriptor {
    fn from(desc: InstanceDescriptor) -> Self {
        desc.0
    }
}

impl From<PathDescriptor> for EntityDescriptor {
    fn from(desc: PathDescriptor) -> Self {
        desc.0
    }
}

/// Builder for constructing EntityDescriptor instances
#[derive(Debug, Default)]
pub struct EntityDescriptorBuilder {
    namespace_segments: Vec<String>,
    component: Option<String>,
    endpoint: Option<String>,
    instance: Option<InstanceType>,
    path_segments: Vec<String>,
}

impl EntityDescriptorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set namespace segments
    pub fn namespace(mut self, segments: &[&str]) -> Self {
        self.namespace_segments = segments.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set namespace segments from Vec<String> (for internal use)
    pub fn namespace_segments(mut self, segments: &[String]) -> Self {
        self.namespace_segments = segments.to_vec();
        self
    }

    /// Set component name
    pub fn component(mut self, component: &str) -> Self {
        self.component = Some(component.to_string());
        self
    }

    /// Set endpoint name
    pub fn endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(endpoint.to_string());
        self
    }

    /// Set instance type
    pub fn instance(mut self, instance: InstanceType) -> Self {
        self.instance = Some(instance);
        self
    }

    /// Set path segments
    pub fn path_segments(mut self, segments: &[&str]) -> Self {
        self.path_segments = segments.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set path segments from Vec<String> (for internal use)
    pub fn path_segments_owned(mut self, segments: &[String]) -> Self {
        self.path_segments = segments.to_vec();
        self
    }

    /// Add a single path segment
    pub fn path_segment(mut self, segment: &str) -> Self {
        self.path_segments.push(segment.to_string());
        self
    }

    /// Append namespace segments to existing ones
    pub fn append_namespace(mut self, segments: &[&str]) -> Self {
        self.namespace_segments
            .extend(segments.iter().map(|s| s.to_string()));
        self
    }

    /// Append path segments to existing ones
    pub fn append_path(mut self, segments: &[&str]) -> Self {
        self.path_segments
            .extend(segments.iter().map(|s| s.to_string()));
        self
    }

    /// Build the descriptor with validation
    pub fn build(self) -> Result<EntityDescriptor, DescriptorError> {
        let descriptor = EntityDescriptor {
            namespace_segments: self.namespace_segments,
            component: self.component,
            endpoint: self.endpoint,
            instance: self.instance,
            path_segments: self.path_segments,
        };

        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Build the descriptor without validation (for internal use with pre-validated data)
    pub fn build_unchecked(self) -> EntityDescriptor {
        EntityDescriptor {
            namespace_segments: self.namespace_segments,
            component: self.component,
            endpoint: self.endpoint,
            instance: self.instance,
            path_segments: self.path_segments,
        }
    }
}

/// Infallible conversion from builder to descriptor (skips validation)
/// Use this when you know the source data is already validated
impl From<EntityDescriptorBuilder> for EntityDescriptor {
    fn from(builder: EntityDescriptorBuilder) -> Self {
        builder.build_unchecked()
    }
}

// Validation helpers

/// Validate a user-provided name (component, endpoint, path segment)
/// Restricted to [a-z0-9_] and cannot start with underscore
fn validate_name(name: &str, kind: &str) -> Result<(), DescriptorError> {
    if name.is_empty() {
        return Err(match kind {
            "component" => DescriptorError::EmptyComponent,
            "endpoint" => DescriptorError::EmptyEndpoint,
            _ => DescriptorError::InvalidPathSegment("Empty path segment".to_string()),
        });
    }

    // Cannot start with underscore (reserved for internal use)
    if name.starts_with('_') {
        let error_msg = format!(
            "{} '{}' cannot start with underscore (reserved for internal use)",
            kind, name
        );
        return Err(match kind {
            "component" => DescriptorError::InvalidComponent(error_msg),
            "endpoint" => DescriptorError::InvalidEndpoint(error_msg),
            _ => DescriptorError::InvalidPathSegment(error_msg),
        });
    }

    // Validate characters: [a-z0-9_]
    for ch in name.chars() {
        if !ch.is_ascii_lowercase() && !ch.is_ascii_digit() && ch != '_' {
            let error_msg = format!(
                "Invalid character '{}' in {} '{}' - only [a-z0-9_] allowed",
                ch, kind, name
            );
            return Err(match kind {
                "component" => DescriptorError::InvalidComponent(error_msg),
                "endpoint" => DescriptorError::InvalidEndpoint(error_msg),
                _ => DescriptorError::InvalidPathSegment(error_msg),
            });
        }
    }

    Ok(())
}

/// Validate a namespace segment (can be internal with leading underscore)
fn validate_name_segment(segment: &str, kind: &str) -> Result<(), DescriptorError> {
    if segment.is_empty() {
        return Err(DescriptorError::InvalidNamespace(
            "Empty namespace segment".to_string(),
        ));
    }

    // Validate characters: [a-z0-9_]
    for ch in segment.chars() {
        if !ch.is_ascii_lowercase() && !ch.is_ascii_digit() && ch != '_' {
            let error_msg = format!(
                "Invalid character '{}' in {} '{}' - only [a-z0-9_] allowed",
                ch, kind, segment
            );
            return Err(DescriptorError::InvalidNamespace(error_msg));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_format_parsing() {
        let url = "dynamo://prod/api/v1?component=gateway&endpoint=http&instance=1234&path=config&path=v1";
        let desc = EntityDescriptor::parse(url).unwrap();

        assert_eq!(desc.namespace_segments(), &["prod", "api", "v1"]);
        assert_eq!(desc.component(), Some("gateway"));
        assert_eq!(desc.endpoint(), Some("http"));
        assert_eq!(desc.instance(), Some(&InstanceType::Distributed(0x1234)));
        assert_eq!(desc.path_segments(), &["config", "v1"]);
    }

    #[test]
    fn test_keyword_format_parsing() {
        let url = "dynamo://prod/api/v1/_c/gateway/_e/http:1234/_path/config/v1";
        let desc = EntityDescriptor::parse(url).unwrap();

        assert_eq!(desc.namespace_segments(), &["prod", "api", "v1"]);
        assert_eq!(desc.component(), Some("gateway"));
        assert_eq!(desc.endpoint(), Some("http"));
        assert_eq!(desc.instance(), Some(&InstanceType::Distributed(0x1234)));
        assert_eq!(desc.path_segments(), &["config", "v1"]);
    }

    #[test]
    fn test_local_instance() {
        let url = "dynamo://ns?component=comp&endpoint=ep&instance=local";
        let desc = EntityDescriptor::parse(url).unwrap();

        assert_eq!(desc.instance(), Some(&InstanceType::Local));

        // Test round-trip
        let etcd_path = desc.to_etcd_path();
        assert_eq!(etcd_path, "dynamo://ns/_c/comp/_e/ep:local");

        let parsed_back = EntityDescriptor::parse(&etcd_path).unwrap();
        assert_eq!(desc, parsed_back);
    }

    #[test]
    fn test_builder() {
        let desc = EntityDescriptor::builder()
            .namespace(&["prod", "api"])
            .component("gateway")
            .endpoint("http")
            .instance(InstanceType::Distributed(0x1234))
            .path_segments(&["config", "v1"])
            .build()
            .unwrap();

        let expected_url =
            "dynamo://prod/api?component=gateway&endpoint=http&instance=1234&path=config&path=v1";
        assert_eq!(desc.to_url(), expected_url);

        let expected_etcd = "dynamo://prod/api/_c/gateway/_e/http:1234/_path/config/v1";
        assert_eq!(desc.to_etcd_path(), expected_etcd);
    }

    #[test]
    fn test_display_defaults_to_query_format() {
        let desc = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .build()
            .unwrap();

        // to_string() should use query format
        assert_eq!(desc.to_string(), "dynamo://ns?component=comp");
    }

    #[test]
    fn test_validation_errors() {
        // Empty namespace
        assert!(EntityDescriptor::builder()
            .component("comp")
            .build()
            .is_err());

        // Component name starting with underscore
        assert!(EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("_invalid")
            .build()
            .is_err());

        // Invalid characters (hyphens not allowed)
        assert!(EntityDescriptor::builder()
            .namespace(&["ns-invalid"])
            .build()
            .is_err());

        // Endpoint without component
        assert!(EntityDescriptor::builder()
            .namespace(&["ns"])
            .endpoint("ep")
            .build()
            .is_err());

        // Instance without endpoint
        assert!(EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .instance(InstanceType::Local)
            .build()
            .is_err());
    }

    #[test]
    fn test_round_trip_parsing() {
        let original = EntityDescriptor::builder()
            .namespace(&["prod", "api", "v1"])
            .component("gateway")
            .endpoint("http")
            .instance(InstanceType::Distributed(0x1234))
            .path_segments(&["config", "v1"])
            .build()
            .unwrap();

        // Round trip through query format
        let query_url = original.to_url();
        let from_query = EntityDescriptor::parse(&query_url).unwrap();
        assert_eq!(original, from_query);

        // Round trip through etcd format
        let etcd_path = original.to_etcd_path();
        let from_etcd = EntityDescriptor::parse(&etcd_path).unwrap();
        assert_eq!(original, from_etcd);

        // Both parsed versions should be equal
        assert_eq!(from_query, from_etcd);
    }

    #[test]
    fn test_namespace_descriptor() {
        // Valid namespace descriptor
        let desc = NamespaceDescriptor::from_namespace(&["prod", "api", "v1"]).unwrap();
        assert_eq!(desc.inner().namespace_segments(), &["prod", "api", "v1"]);
        assert_eq!(desc.inner().component(), None);
        assert_eq!(desc.inner().endpoint(), None);
        assert_eq!(desc.inner().instance(), None);
        assert_eq!(desc.inner().path_segments(), &[] as &[String]);

        // Test parsing
        let parsed = NamespaceDescriptor::parse("dynamo://prod/api/v1").unwrap();
        assert_eq!(desc, parsed);

        // Should fail with component
        let with_component = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .build()
            .unwrap();
        assert!(NamespaceDescriptor::new(with_component).is_err());

        // Should fail with endpoint
        let with_endpoint = EntityDescriptor::builder()
            .namespace(&["ns"])
            .endpoint("ep")
            .build();
        assert!(with_endpoint.is_err()); // This should already fail in EntityDescriptor validation

        // Should fail with path
        let with_path = EntityDescriptor::builder()
            .namespace(&["ns"])
            .path_segments(&["path"])
            .build()
            .unwrap();
        assert!(NamespaceDescriptor::new(with_path).is_err());
    }

    #[test]
    fn test_component_descriptor() {
        // Valid component descriptor
        let desc = ComponentDescriptor::from_component(&["prod", "api"], "gateway").unwrap();
        assert_eq!(desc.inner().namespace_segments(), &["prod", "api"]);
        assert_eq!(desc.inner().component(), Some("gateway"));
        assert_eq!(desc.inner().endpoint(), None);
        assert_eq!(desc.inner().instance(), None);
        assert_eq!(desc.inner().path_segments(), &[] as &[String]);

        // Test parsing
        let parsed = ComponentDescriptor::parse("dynamo://prod/api?component=gateway").unwrap();
        assert_eq!(desc, parsed);

        // Should fail without component
        let without_component = EntityDescriptor::builder()
            .namespace(&["ns"])
            .build()
            .unwrap();
        assert!(ComponentDescriptor::new(without_component).is_err());

        // Should fail with endpoint
        let with_endpoint = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .endpoint("ep")
            .build()
            .unwrap();
        assert!(ComponentDescriptor::new(with_endpoint).is_err());

        // Should fail with path
        let with_path = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .path_segments(&["path"])
            .build()
            .unwrap();
        assert!(ComponentDescriptor::new(with_path).is_err());
    }

    #[test]
    fn test_endpoint_descriptor() {
        // Valid endpoint descriptor
        let desc = EndpointDescriptor::from_endpoint(&["prod"], "gateway", "http").unwrap();
        assert_eq!(desc.inner().namespace_segments(), &["prod"]);
        assert_eq!(desc.inner().component(), Some("gateway"));
        assert_eq!(desc.inner().endpoint(), Some("http"));
        assert_eq!(desc.inner().instance(), None);
        assert_eq!(desc.inner().path_segments(), &[] as &[String]);

        // Test parsing
        let parsed =
            EndpointDescriptor::parse("dynamo://prod?component=gateway&endpoint=http").unwrap();
        assert_eq!(desc, parsed);

        // Should fail without component
        let without_component = EntityDescriptor::builder()
            .namespace(&["ns"])
            .endpoint("ep")
            .build();
        assert!(without_component.is_err()); // Should already fail in EntityDescriptor validation

        // Should fail without endpoint
        let without_endpoint = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .build()
            .unwrap();
        assert!(EndpointDescriptor::new(without_endpoint).is_err());

        // Should fail with instance
        let with_instance = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .endpoint("ep")
            .instance(InstanceType::Local)
            .build()
            .unwrap();
        assert!(EndpointDescriptor::new(with_instance).is_err());

        // Should fail with path
        let with_path = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .endpoint("ep")
            .path_segments(&["path"])
            .build()
            .unwrap();
        assert!(EndpointDescriptor::new(with_path).is_err());
    }

    #[test]
    fn test_instance_descriptor() {
        // Valid instance descriptor
        let desc =
            InstanceDescriptor::from_instance(&["prod"], "gateway", "http", InstanceType::Local)
                .unwrap();
        assert_eq!(desc.inner().namespace_segments(), &["prod"]);
        assert_eq!(desc.inner().component(), Some("gateway"));
        assert_eq!(desc.inner().endpoint(), Some("http"));
        assert_eq!(desc.inner().instance(), Some(&InstanceType::Local));
        assert_eq!(desc.inner().path_segments(), &[] as &[String]);

        // Test parsing
        let parsed = InstanceDescriptor::parse(
            "dynamo://prod?component=gateway&endpoint=http&instance=local",
        )
        .unwrap();
        assert_eq!(desc, parsed);

        // Test with distributed instance
        let distributed_desc = InstanceDescriptor::from_instance(
            &["ns"],
            "comp",
            "ep",
            InstanceType::Distributed(0x1234),
        )
        .unwrap();
        let distributed_parsed =
            InstanceDescriptor::parse("dynamo://ns?component=comp&endpoint=ep&instance=1234")
                .unwrap();
        assert_eq!(distributed_desc, distributed_parsed);

        // Should fail without instance
        let without_instance = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .endpoint("ep")
            .build()
            .unwrap();
        assert!(InstanceDescriptor::new(without_instance).is_err());

        // Should fail with path
        let with_path = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .endpoint("ep")
            .instance(InstanceType::Local)
            .path_segments(&["path"])
            .build()
            .unwrap();
        assert!(InstanceDescriptor::new(with_path).is_err());
    }

    #[test]
    fn test_path_descriptor() {
        // Valid path descriptor - namespace + path
        let desc = PathDescriptor::from_namespace_path(&["prod"], &["config", "v1"]).unwrap();
        assert_eq!(desc.inner().namespace_segments(), &["prod"]);
        assert_eq!(desc.inner().component(), None);
        assert_eq!(desc.inner().endpoint(), None);
        assert_eq!(desc.inner().instance(), None);
        assert_eq!(desc.inner().path_segments(), &["config", "v1"]);

        // Valid path descriptor - component + path
        let desc2 = PathDescriptor::from_component_path(&["prod"], "gateway", &["config"]).unwrap();
        assert_eq!(desc2.inner().component(), Some("gateway"));
        assert_eq!(desc2.inner().path_segments(), &["config"]);

        // Valid path descriptor - endpoint + path
        let desc3 =
            PathDescriptor::from_endpoint_path(&["prod"], "gateway", "http", &["config"]).unwrap();
        assert_eq!(desc3.inner().endpoint(), Some("http"));
        assert_eq!(desc3.inner().path_segments(), &["config"]);

        // Valid path descriptor - instance + path
        let desc4 = PathDescriptor::from_instance_path(
            &["prod"],
            "gateway",
            "http",
            InstanceType::Local,
            &["config"],
        )
        .unwrap();
        assert_eq!(desc4.inner().instance(), Some(&InstanceType::Local));
        assert_eq!(desc4.inner().path_segments(), &["config"]);

        // Test parsing
        let parsed = PathDescriptor::parse("dynamo://prod?path=config&path=v1").unwrap();
        assert_eq!(desc, parsed);

        let parsed2 = PathDescriptor::parse("dynamo://prod?component=gateway&path=config").unwrap();
        assert_eq!(desc2, parsed2);

        // Should fail without path segments
        let without_path = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .build()
            .unwrap();
        assert!(PathDescriptor::new(without_path).is_err());
    }

    #[test]
    fn test_descriptor_display_and_parsing() {
        // Test that all descriptor types can be converted to string and parsed back
        let namespace_desc = NamespaceDescriptor::from_namespace(&["prod", "api"]).unwrap();
        let namespace_str = namespace_desc.to_string();
        let namespace_parsed = NamespaceDescriptor::parse(&namespace_str).unwrap();
        assert_eq!(namespace_desc, namespace_parsed);

        let component_desc = ComponentDescriptor::from_component(&["prod"], "gateway").unwrap();
        let component_str = component_desc.to_string();
        let component_parsed = ComponentDescriptor::parse(&component_str).unwrap();
        assert_eq!(component_desc, component_parsed);

        let endpoint_desc =
            EndpointDescriptor::from_endpoint(&["prod"], "gateway", "http").unwrap();
        let endpoint_str = endpoint_desc.to_string();
        let endpoint_parsed = EndpointDescriptor::parse(&endpoint_str).unwrap();
        assert_eq!(endpoint_desc, endpoint_parsed);

        let instance_desc =
            InstanceDescriptor::from_instance(&["prod"], "gateway", "http", InstanceType::Local)
                .unwrap();
        let instance_str = instance_desc.to_string();
        let instance_parsed = InstanceDescriptor::parse(&instance_str).unwrap();
        assert_eq!(instance_desc, instance_parsed);

        let path_desc = PathDescriptor::from_namespace_path(&["prod"], &["config"]).unwrap();
        let path_str = path_desc.to_string();
        let path_parsed = PathDescriptor::parse(&path_str).unwrap();
        assert_eq!(path_desc, path_parsed);
    }

    #[test]
    fn test_try_from_entity_descriptor() {
        // Test successful conversions

        // Namespace descriptor
        let ns_entity = EntityDescriptor::builder()
            .namespace(&["prod", "api"])
            .build()
            .unwrap();
        let ns_desc: NamespaceDescriptor = ns_entity.clone().try_into().unwrap();
        assert_eq!(ns_desc.inner(), &ns_entity);

        // Component descriptor
        let comp_entity = EntityDescriptor::builder()
            .namespace(&["prod"])
            .component("gateway")
            .build()
            .unwrap();
        let comp_desc: ComponentDescriptor = comp_entity.clone().try_into().unwrap();
        assert_eq!(comp_desc.inner(), &comp_entity);

        // Endpoint descriptor
        let ep_entity = EntityDescriptor::builder()
            .namespace(&["prod"])
            .component("gateway")
            .endpoint("http")
            .build()
            .unwrap();
        let ep_desc: EndpointDescriptor = ep_entity.clone().try_into().unwrap();
        assert_eq!(ep_desc.inner(), &ep_entity);

        // Instance descriptor
        let inst_entity = EntityDescriptor::builder()
            .namespace(&["prod"])
            .component("gateway")
            .endpoint("http")
            .instance(InstanceType::Local)
            .build()
            .unwrap();
        let inst_desc: InstanceDescriptor = inst_entity.clone().try_into().unwrap();
        assert_eq!(inst_desc.inner(), &inst_entity);

        // Path descriptor
        let path_entity = EntityDescriptor::builder()
            .namespace(&["prod"])
            .path_segments(&["config", "v1"])
            .build()
            .unwrap();
        let path_desc: PathDescriptor = path_entity.clone().try_into().unwrap();
        assert_eq!(path_desc.inner(), &path_entity);
    }

    #[test]
    fn test_try_from_entity_descriptor_failures() {
        // Test failed conversions due to validation errors

        // NamespaceDescriptor should fail with component
        let with_component = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .build()
            .unwrap();
        let ns_result: Result<NamespaceDescriptor, _> = with_component.try_into();
        assert!(ns_result.is_err());

        // ComponentDescriptor should fail without component
        let without_component = EntityDescriptor::builder()
            .namespace(&["ns"])
            .build()
            .unwrap();
        let comp_result: Result<ComponentDescriptor, _> = without_component.try_into();
        assert!(comp_result.is_err());

        // ComponentDescriptor should fail with endpoint
        let with_endpoint = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .endpoint("ep")
            .build()
            .unwrap();
        let comp_result: Result<ComponentDescriptor, _> = with_endpoint.try_into();
        assert!(comp_result.is_err());

        // EndpointDescriptor should fail without endpoint
        let without_endpoint = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .build()
            .unwrap();
        let ep_result: Result<EndpointDescriptor, _> = without_endpoint.try_into();
        assert!(ep_result.is_err());

        // EndpointDescriptor should fail with instance
        let with_instance = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .endpoint("ep")
            .instance(InstanceType::Local)
            .build()
            .unwrap();
        let ep_result: Result<EndpointDescriptor, _> = with_instance.try_into();
        assert!(ep_result.is_err());

        // InstanceDescriptor should fail without instance
        let without_instance = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .endpoint("ep")
            .build()
            .unwrap();
        let inst_result: Result<InstanceDescriptor, _> = without_instance.try_into();
        assert!(inst_result.is_err());

        // InstanceDescriptor should fail with path
        let with_path = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .endpoint("ep")
            .instance(InstanceType::Local)
            .path_segments(&["path"])
            .build()
            .unwrap();
        let inst_result: Result<InstanceDescriptor, _> = with_path.try_into();
        assert!(inst_result.is_err());

        // PathDescriptor should fail without path
        let without_path = EntityDescriptor::builder()
            .namespace(&["ns"])
            .component("comp")
            .build()
            .unwrap();
        let path_result: Result<PathDescriptor, _> = without_path.try_into();
        assert!(path_result.is_err());
    }

    #[test]
    fn test_from_specialized_to_entity_descriptor() {
        // Test infallible conversions from specialized types to EntityDescriptor

        // NamespaceDescriptor to EntityDescriptor
        let ns_desc = NamespaceDescriptor::from_namespace(&["prod", "api"]).unwrap();
        let ns_entity: EntityDescriptor = ns_desc.clone().into();
        assert_eq!(ns_entity, *ns_desc.inner());

        // ComponentDescriptor to EntityDescriptor
        let comp_desc = ComponentDescriptor::from_component(&["prod"], "gateway").unwrap();
        let comp_entity: EntityDescriptor = comp_desc.clone().into();
        assert_eq!(comp_entity, *comp_desc.inner());

        // EndpointDescriptor to EntityDescriptor
        let ep_desc = EndpointDescriptor::from_endpoint(&["prod"], "gateway", "http").unwrap();
        let ep_entity: EntityDescriptor = ep_desc.clone().into();
        assert_eq!(ep_entity, *ep_desc.inner());

        // InstanceDescriptor to EntityDescriptor
        let inst_desc =
            InstanceDescriptor::from_instance(&["prod"], "gateway", "http", InstanceType::Local)
                .unwrap();
        let inst_entity: EntityDescriptor = inst_desc.clone().into();
        assert_eq!(inst_entity, *inst_desc.inner());

        // PathDescriptor to EntityDescriptor
        let path_desc = PathDescriptor::from_namespace_path(&["prod"], &["config"]).unwrap();
        let path_entity: EntityDescriptor = path_desc.clone().into();
        assert_eq!(path_entity, *path_desc.inner());
    }

    #[test]
    fn test_round_trip_conversions() {
        // Test round-trip conversions: specialized -> EntityDescriptor -> specialized

        // Create original specialized descriptors
        let original_ns = NamespaceDescriptor::from_namespace(&["prod", "api"]).unwrap();
        let original_comp = ComponentDescriptor::from_component(&["prod"], "gateway").unwrap();
        let original_ep = EndpointDescriptor::from_endpoint(&["prod"], "gateway", "http").unwrap();
        let original_inst =
            InstanceDescriptor::from_instance(&["prod"], "gateway", "http", InstanceType::Local)
                .unwrap();
        let original_path = PathDescriptor::from_namespace_path(&["prod"], &["config"]).unwrap();

        // Convert to EntityDescriptor and back
        let ns_entity: EntityDescriptor = original_ns.clone().into();
        let ns_back: NamespaceDescriptor = ns_entity.try_into().unwrap();
        assert_eq!(original_ns, ns_back);

        let comp_entity: EntityDescriptor = original_comp.clone().into();
        let comp_back: ComponentDescriptor = comp_entity.try_into().unwrap();
        assert_eq!(original_comp, comp_back);

        let ep_entity: EntityDescriptor = original_ep.clone().into();
        let ep_back: EndpointDescriptor = ep_entity.try_into().unwrap();
        assert_eq!(original_ep, ep_back);

        let inst_entity: EntityDescriptor = original_inst.clone().into();
        let inst_back: InstanceDescriptor = inst_entity.try_into().unwrap();
        assert_eq!(original_inst, inst_back);

        let path_entity: EntityDescriptor = original_path.clone().into();
        let path_back: PathDescriptor = path_entity.try_into().unwrap();
        assert_eq!(original_path, path_back);
    }

    #[test]
    fn test_descriptor_transitions() {
        // Test NamespaceDescriptor transitions
        let ns_desc = NamespaceDescriptor::from_namespace(&["prod", "api"]).unwrap();

        // Namespace -> Component
        let comp_desc = ns_desc.clone().component("gateway").unwrap();
        assert_eq!(comp_desc.inner().namespace_segments(), &["prod", "api"]);
        assert_eq!(comp_desc.inner().component(), Some("gateway"));
        assert_eq!(comp_desc.inner().endpoint(), None);
        assert_eq!(comp_desc.inner().instance(), None);
        assert_eq!(comp_desc.inner().path_segments(), &[] as &[String]);

        // Namespace -> Path
        let path_desc = ns_desc.path(&["config", "v1"]).unwrap();
        assert_eq!(path_desc.inner().namespace_segments(), &["prod", "api"]);
        assert_eq!(path_desc.inner().component(), None);
        assert_eq!(path_desc.inner().path_segments(), &["config", "v1"]);

        // Test ComponentDescriptor transitions
        let comp_desc = ComponentDescriptor::from_component(&["prod"], "gateway").unwrap();

        // Component -> Endpoint
        let ep_desc = comp_desc.clone().endpoint("http").unwrap();
        assert_eq!(ep_desc.inner().namespace_segments(), &["prod"]);
        assert_eq!(ep_desc.inner().component(), Some("gateway"));
        assert_eq!(ep_desc.inner().endpoint(), Some("http"));
        assert_eq!(ep_desc.inner().instance(), None);
        assert_eq!(ep_desc.inner().path_segments(), &[] as &[String]);

        // Component -> Path
        let path_desc = comp_desc.path(&["data"]).unwrap();
        assert_eq!(path_desc.inner().namespace_segments(), &["prod"]);
        assert_eq!(path_desc.inner().component(), Some("gateway"));
        assert_eq!(path_desc.inner().endpoint(), None);
        assert_eq!(path_desc.inner().path_segments(), &["data"]);

        // Test EndpointDescriptor transitions
        let ep_desc = EndpointDescriptor::from_endpoint(&["prod"], "gateway", "http").unwrap();

        // Endpoint -> Instance
        let inst_desc = ep_desc.clone().instance(InstanceType::Local).unwrap();
        assert_eq!(inst_desc.inner().namespace_segments(), &["prod"]);
        assert_eq!(inst_desc.inner().component(), Some("gateway"));
        assert_eq!(inst_desc.inner().endpoint(), Some("http"));
        assert_eq!(inst_desc.inner().instance(), Some(&InstanceType::Local));
        assert_eq!(inst_desc.inner().path_segments(), &[] as &[String]);

        // Endpoint -> Path
        let path_desc = ep_desc.path(&["metrics"]).unwrap();
        assert_eq!(path_desc.inner().namespace_segments(), &["prod"]);
        assert_eq!(path_desc.inner().component(), Some("gateway"));
        assert_eq!(path_desc.inner().endpoint(), Some("http"));
        assert_eq!(path_desc.inner().instance(), None);
        assert_eq!(path_desc.inner().path_segments(), &["metrics"]);

        // Test InstanceDescriptor transitions
        let inst_desc = InstanceDescriptor::from_instance(
            &["prod"],
            "gateway",
            "http",
            InstanceType::Distributed(0x1234),
        )
        .unwrap();

        // Instance -> Path
        let path_desc = inst_desc.path(&["logs", "recent"]).unwrap();
        assert_eq!(path_desc.inner().namespace_segments(), &["prod"]);
        assert_eq!(path_desc.inner().component(), Some("gateway"));
        assert_eq!(path_desc.inner().endpoint(), Some("http"));
        assert_eq!(
            path_desc.inner().instance(),
            Some(&InstanceType::Distributed(0x1234))
        );
        assert_eq!(path_desc.inner().path_segments(), &["logs", "recent"]);

        // Test PathDescriptor transitions (adding more path segments)
        let initial_path = PathDescriptor::from_namespace_path(&["dev"], &["config"]).unwrap();

        // Path -> Path (adding more segments)
        let extended_path = initial_path.path(&["database", "settings"]).unwrap();
        assert_eq!(extended_path.inner().namespace_segments(), &["dev"]);
        assert_eq!(extended_path.inner().component(), None);
        assert_eq!(
            extended_path.inner().path_segments(),
            &["config", "database", "settings"]
        );

        // Test PathDescriptor transitions with all fields present
        let full_path = PathDescriptor::from_instance_path(
            &["prod"],
            "api",
            "rest",
            InstanceType::Local,
            &["v1"],
        )
        .unwrap();
        let extended_full_path = full_path.path(&["users", "profile"]).unwrap();
        assert_eq!(extended_full_path.inner().namespace_segments(), &["prod"]);
        assert_eq!(extended_full_path.inner().component(), Some("api"));
        assert_eq!(extended_full_path.inner().endpoint(), Some("rest"));
        assert_eq!(
            extended_full_path.inner().instance(),
            Some(&InstanceType::Local)
        );
        assert_eq!(
            extended_full_path.inner().path_segments(),
            &["v1", "users", "profile"]
        );
    }

    #[test]
    fn test_chained_transitions() {
        // Test chaining multiple transitions
        let result = NamespaceDescriptor::from_namespace(&["prod", "api"])
            .unwrap()
            .component("gateway")
            .unwrap()
            .endpoint("http")
            .unwrap()
            .instance(InstanceType::Local)
            .unwrap()
            .path(&["health", "check"])
            .unwrap();

        assert_eq!(result.inner().namespace_segments(), &["prod", "api"]);
        assert_eq!(result.inner().component(), Some("gateway"));
        assert_eq!(result.inner().endpoint(), Some("http"));
        assert_eq!(result.inner().instance(), Some(&InstanceType::Local));
        assert_eq!(result.inner().path_segments(), &["health", "check"]);

        // Test URL generation for chained result
        let url = result.to_string();
        assert!(url.contains("prod/api"));
        assert!(url.contains("component=gateway"));
        assert!(url.contains("endpoint=http"));
        assert!(url.contains("instance=local"));
        assert!(url.contains("path=health"));
        assert!(url.contains("path=check"));
    }

    #[test]
    fn test_optimized_transitions_validation() {
        // Test that transitions only validate new fields, not existing ones

        // Create a valid namespace descriptor
        let ns_desc = NamespaceDescriptor::from_namespace(&["prod", "api"]).unwrap();

        // Test that component transition only validates the component field
        // (this would fail with full validation if we passed an invalid component)
        let comp_result = ns_desc.clone().component("valid_component");
        assert!(comp_result.is_ok());

        // Test that an invalid component name fails quickly
        let invalid_comp_result = ns_desc.clone().component("Invalid-Component");
        assert!(invalid_comp_result.is_err());

        // Test that path transition only validates path segments
        let path_result = ns_desc.clone().path(&["valid", "path"]);
        assert!(path_result.is_ok());

        // Test that invalid path segment fails quickly
        let invalid_path_result = ns_desc.path(&["invalid-path"]);
        assert!(invalid_path_result.is_err());

        // Test PathDescriptor append validation
        let initial_path = PathDescriptor::from_namespace_path(&["ns"], &["existing"]).unwrap();
        let append_result = initial_path.clone().path(&["new", "segments"]);
        assert!(append_result.is_ok());

        // Verify the appended path contains all segments
        let final_path = append_result.unwrap();
        assert_eq!(
            final_path.inner().path_segments(),
            &["existing", "new", "segments"]
        );

        // Test that invalid append fails
        let invalid_append = initial_path.path(&["invalid-segment"]);
        assert!(invalid_append.is_err());
    }

    #[test]
    fn test_internal_namespace() {
        // Test creating InternalNamespace
        let internal_ns = InternalNamespace::new().unwrap();
        assert_eq!(internal_ns.entity().namespace_segments(), &["_internal"]);
        assert_eq!(internal_ns.entity().component(), None);
        assert_eq!(internal_ns.entity().endpoint(), None);
        assert_eq!(internal_ns.entity().instance(), None);
        assert_eq!(internal_ns.entity().path_segments(), &[] as &[String]);

        // Test default implementation
        let default_internal = InternalNamespace::default();
        assert_eq!(internal_ns, default_internal);

        // Test Display and parsing
        let url = internal_ns.to_string();
        assert_eq!(url, "dynamo://_internal");

        let parsed = InternalNamespace::from_str(&url).unwrap();
        assert_eq!(internal_ns, parsed);

        // Test conversion to EntityDescriptor
        let entity: EntityDescriptor = internal_ns.clone().into();
        assert_eq!(entity.namespace_segments(), &["_internal"]);

        // Test TryFrom EntityDescriptor
        let back_to_internal: InternalNamespace = entity.try_into().unwrap();
        assert_eq!(internal_ns, back_to_internal);
    }

    #[test]
    fn test_internal_namespace_component_transitions() {
        let internal_ns = InternalNamespace::new().unwrap();

        // Test adding regular component (should use normal validation)
        let regular_comp = internal_ns.clone().component("regular").unwrap();
        assert_eq!(regular_comp.inner().component(), Some("regular"));

        // Test that underscore-prefixed components are rejected (normal validation rules apply)
        assert!(internal_ns
            .clone()
            .component("_internal_component")
            .is_err());
        assert!(internal_ns.clone().component("_system").is_err());
        assert!(internal_ns.clone().component("_cache").is_err());
        assert!(internal_ns.clone().component("_metrics").is_err());

        // Test invalid characters still fail
        assert!(internal_ns.clone().component("invalid-name").is_err());
        assert!(internal_ns.clone().component("Invalid").is_err());
        assert!(internal_ns.component("").is_err());
    }

    #[test]
    fn test_internal_types_validation_failures() {
        // Test that non-internal namespace fails for InternalNamespace
        let regular_entity = EntityDescriptor::builder()
            .namespace(&["regular"])
            .build()
            .unwrap();
        let internal_ns_result: Result<InternalNamespace, _> = regular_entity.try_into();
        assert!(internal_ns_result.is_err());

        // Test FromStr validation
        assert!(InternalNamespace::from_str("dynamo://regular").is_err());
    }

    #[test]
    fn test_internal_namespace_to_path() {
        // Test the transition from InternalNamespace to PathDescriptor
        let internal_ns = InternalNamespace::new().unwrap();

        // Should create PathDescriptor with regular segments only (no underscore prefixes)
        let regular_path = internal_ns.clone().path(&["config", "database"]).unwrap();
        assert_eq!(
            regular_path.inner().path_segments(),
            &["config", "database"]
        );
        assert_eq!(regular_path.inner().namespace_segments(), &["_internal"]);

        // Should reject underscore prefixed segments (normal validation rules apply)
        let invalid_path_result = internal_ns.path(&["_system", "_cache"]);
        assert!(invalid_path_result.is_err());
    }
}
