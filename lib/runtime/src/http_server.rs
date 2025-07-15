// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use axum::{body, http::StatusCode, response::IntoResponse, routing::get, Router};
use crate::profiling::{MetricsRegistry, MetricGauge};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing;

/// Runtime metrics for HTTP server
pub struct RuntimeMetrics {
    pub backend: Arc<dyn MetricsRegistry>,
    pub uptime_gauge: Box<dyn MetricGauge>,
}

impl RuntimeMetrics {
    pub fn new(backend: Arc<dyn MetricsRegistry>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let uptime_gauge = backend.create_gauge(
            "uptime_seconds",
            "Total uptime of the DistributedRuntime in seconds",
            &[("service", "dynamo"), ("subsystem", "runtime")]
        )?;
        Ok(Self {
            backend,
            uptime_gauge
        })
    }

    /// Get a reference to the backend
    pub fn get_backend(&self) -> &Arc<dyn MetricsRegistry> {
        &self.backend
    }
}

/// HTTP server state containing pre-created metrics
pub struct HttpServerState {
    drt: Arc<crate::DistributedRuntime>,
    runtime_metrics: Arc<RuntimeMetrics>,
}

impl HttpServerState {
    /// Create new HTTP server state with the provided metrics backend
    pub fn new(drt: Arc<crate::DistributedRuntime>, backend: Arc<dyn MetricsRegistry>) -> anyhow::Result<Self> {
        let runtime_metrics = match RuntimeMetrics::new(backend) {
            Ok(metrics) => Arc::new(metrics),
            Err(e) => return Err(anyhow::anyhow!("Failed to create runtime metrics: {}", e)),
        };
        Ok(Self {
            drt,
            runtime_metrics,
        })
    }

    pub fn drt(&self) -> &Arc<crate::DistributedRuntime> {
        &self.drt
    }
}

/// Start HTTP server with DistributedRuntime support
pub async fn spawn_http_server(
    host: &str,
    port: u16,
    cancel_token: CancellationToken,
    drt: Arc<crate::DistributedRuntime>,
    metrics_backend: Arc<dyn MetricsRegistry>,
) -> anyhow::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    // Create HTTP server state with the provided metrics backend
    let server_state = Arc::new(HttpServerState::new(drt, metrics_backend)?);

    let app = Router::new()
        .route(
            "/health",
            get({
                let state = Arc::clone(&server_state);
                move || health_handler(state.clone())
            }),
        )
        .route(
            "/live",
            get({
                let state = Arc::clone(&server_state);
                move || health_handler(state)
            }),
        )
        .route(
            "/metrics",
            get({
                let state = Arc::clone(&server_state);
                move || metrics_handler(state)
            }),
        )
        .fallback(|| async {
            tracing::info!("[fallback handler] called");
            (StatusCode::NOT_FOUND, "Route not found").into_response()
        });

    let address = format!("{}:{}", host, port);
    tracing::info!("[spawn_http_server] binding to: {}", address);

    let listener = match TcpListener::bind(&address).await {
        Ok(listener) => {
            // get the actual address and port, print in debug level
            let actual_address = listener.local_addr()?;
            tracing::info!(
                "[spawn_http_server] HTTP server bound to: {}",
                actual_address
            );
            (listener, actual_address)
        }
        Err(e) => {
            tracing::error!("Failed to bind to address {}: {}", address, e);
            return Err(anyhow::anyhow!("Failed to bind to address: {}", e));
        }
    };
    let (listener, actual_address) = listener;

    let observer = cancel_token.child_token();
    // Spawn the server in the background and return the handle
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(observer.cancelled_owned())
            .await
        {
            tracing::error!("HTTP server error: {}", e);
        }
    });

    Ok((actual_address, handle))
}

/// Health handler
async fn health_handler(state: Arc<HttpServerState>) -> impl IntoResponse {
    tracing::info!("[health_handler] called");
    let uptime = state.drt.uptime();
    let response = format!("OK\nUptime: {} seconds\n", uptime.as_secs());
    (StatusCode::OK, response)
}

/// Metrics handler with DistributedRuntime uptime
async fn metrics_handler(state: Arc<HttpServerState>) -> impl IntoResponse {
    // Update the uptime gauge with current value
    let uptime_seconds = state.drt.uptime().as_secs_f64();
    state.runtime_metrics.uptime_gauge.set(uptime_seconds);

    // Get metrics from the backend
    match state.runtime_metrics.backend.root_prometheus_format_str() {
        Ok(response) => (StatusCode::OK, response),
        Err(e) => {
            tracing::error!("Failed to get metrics from backend: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to get metrics".to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::PrometheusRegistry;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_http_server_lifecycle() {
        let cancel_token = CancellationToken::new();
        let cancel_token_for_server = cancel_token.clone();

        // Test basic HTTP server lifecycle without DistributedRuntime
        let app = Router::new().route("/test", get(|| async { (StatusCode::OK, "test") }));

        // start HTTP server
        let server_handle = tokio::spawn(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(cancel_token_for_server.cancelled_owned())
                .await;
        });

        // wait for a while to let the server start
        sleep(Duration::from_millis(100)).await;

        // cancel token
        cancel_token.cancel();

        // wait for the server to shut down
        let result = tokio::time::timeout(Duration::from_secs(5), server_handle).await;
        assert!(
            result.is_ok(),
            "HTTP server should shut down when cancel token is cancelled"
        );
    }

    #[tokio::test]
    async fn test_runtime_metrics_creation() {
        // Test RuntimeMetrics creation and functionality
        let backend = Arc::new(PrometheusRegistry::new("test")) as Arc<dyn MetricsRegistry>;
        let runtime_metrics = RuntimeMetrics::new(backend).unwrap();

        // Wait a bit to ensure uptime is measurable
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Test updating uptime
        let uptime_seconds = 123.456;
        runtime_metrics.uptime_gauge.set(uptime_seconds);

        // Get metrics from the backend
        let response = runtime_metrics.backend.root_prometheus_format_str().unwrap();
        println!("Full metrics response:\n{}", response);

        let expected = "\
# HELP test_uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE test_uptime_seconds gauge
test_uptime_seconds 123.456
";
        assert_eq!(response, expected);
    }

    #[tokio::test]
    async fn test_runtime_metrics_namespace() {
        // Test that metrics have correct namespace
        let backend = Arc::new(PrometheusRegistry::new("test")) as Arc<dyn MetricsRegistry>;
        let runtime_metrics = RuntimeMetrics::new(backend).unwrap();

        runtime_metrics.uptime_gauge.set(42.0);

        let response = runtime_metrics.backend.root_prometheus_format_str().unwrap();
        println!("Full metrics response:\n{}", response);

        let expected = "\
# HELP test_uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE test_uptime_seconds gauge
test_uptime_seconds 42
";
        assert_eq!(response, expected);
    }

    #[tokio::test]
    async fn test_spawn_http_server_endpoints() {
        use std::sync::Arc;
        use tokio::time::sleep;
        use tokio_util::sync::CancellationToken;
        // use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // use reqwest for HTTP requests
        let runtime = crate::Runtime::single_threaded().unwrap();
        let drt = Arc::new(
            crate::DistributedRuntime::from_settings_without_discovery(runtime)
                .await
                .unwrap(),
        );
        let cancel_token = CancellationToken::new();
        let metrics_backend = Arc::new(PrometheusRegistry::new("test")) as Arc<dyn MetricsRegistry>;
        let (addr, server_handle) = spawn_http_server("127.0.0.1", 0, cancel_token.clone(), drt, metrics_backend)
            .await
            .unwrap();
        println!("[test] Waiting for server to start...");
        sleep(std::time::Duration::from_millis(1000)).await;
        println!("[test] Server should be up, starting requests...");
        let client = reqwest::Client::new();
        for (path, expect_200, expect_body) in [
            ("/health", true, "OK"),
            ("/live", true, "OK"),
            ("/someRandomPathNotFoundHere", false, "Route not found"),
        ] {
            println!("[test] Sending request to {}", path);
            let url = format!("http://{}{}", addr, path);
            let response = client.get(&url).send().await.unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            println!(
                "[test] Response for {}: status={}, body={:?}",
                path, status, body
            );
            if expect_200 {
                assert_eq!(status, 200, "Response: status={}, body={:?}", status, body);
            } else {
                assert_eq!(status, 404, "Response: status={}, body={:?}", status, body);
            }
            assert!(
                body.contains(expect_body),
                "Response: status={}, body={:?}",
                status,
                body
            );
        }
        cancel_token.cancel();
        match server_handle.await {
            Ok(_) => println!("[test] Server shut down normally"),
            Err(e) => {
                if e.is_panic() {
                    println!("[test] Server panicked: {:?}", e);
                } else {
                    println!("[test] Server cancelled: {:?}", e);
                }
            }
        }
    }
}
