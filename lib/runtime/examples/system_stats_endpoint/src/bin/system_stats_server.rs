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

use http_server_metrics::{MyStats, DEFAULT_NAMESPACE};

use dynamo_runtime::{
    logging,
    pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    },
    profiling::{MetricCounter, MetricGauge, MetricHistogram, MetricsRegistry},
    protocols::annotated::Annotated,
    stream, DistributedRuntime, Result, Runtime, Worker,
};
use std::sync::Arc;

/// Service metrics struct using the metric classes from profiling.rs
// TODO(keiven): implement register trait
pub struct ExampleHTTPMetrics {
    registry: Arc<dyn MetricsRegistry>,
    pub request_counter: Box<dyn MetricCounter>,
    pub active_requests_gauge: Box<dyn MetricGauge>,
    pub request_duration_histogram: Box<dyn MetricHistogram>,
}

impl ExampleHTTPMetrics {
    /// Create a new ServiceMetrics instance using the metric backend
    pub fn new(
        backend: Arc<dyn MetricsRegistry>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Create request counter
        // TODO: namespace - component - name
        let request_counter = backend.create_counter(
            "service_requests_total",
            "Total number of requests processed",
            &[("service", "backend")],
        )?;

        // Create active requests gauge
        let active_requests_gauge = backend.create_gauge(
            "service_active_requests",
            "Number of requests currently being processed",
            &[("service", "backend")],
        )?;

        // Create request duration histogram
        let request_duration_histogram = backend.create_histogram(
            "service_request_duration_seconds",
            "Request duration in seconds",
            &[("service", "backend")],
        )?;

        Ok(ExampleHTTPMetrics {
            registry: backend,
            request_counter,
            active_requests_gauge,
            request_duration_histogram,
        })
    }

    /// Get a read-only reference to the backend
    pub fn backend(&self) -> &Arc<dyn MetricsRegistry> {
        &self.registry
    }
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    backend(distributed).await
}

struct RequestHandler {
    metrics: Arc<ExampleHTTPMetrics>,
}

impl RequestHandler {
    fn new(metrics: Arc<ExampleHTTPMetrics>) -> Arc<Self> {
        Arc::new(Self { metrics })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
    async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let start_time = std::time::Instant::now();

        // Record request start
        self.metrics.request_counter.inc();
        self.metrics.active_requests_gauge.inc(1.0);

        let (data, ctx) = input.into_parts();

        let chars = data
            .chars()
            .map(|c| Annotated::from_data(c.to_string()))
            .collect::<Vec<_>>();

        let stream = stream::iter(chars);

        // Calculate duration
        let duration = start_time.elapsed().as_secs_f64();

        // Record request end
        self.metrics.active_requests_gauge.dec(1.0);
        self.metrics.request_duration_histogram.observe(duration);
        // self.metrics.response_size_histogram.observe(response_size); // This line was removed

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

async fn backend(runtime: DistributedRuntime) -> Result<()> {
    // Get the metrics backend from the runtime (async, lazy-initialized)
    let registry = runtime.metrics_registry().await?;
    // Initialize metrics using the profiling-based struct
    let metrics =
        Arc::new(ExampleHTTPMetrics::new(registry.clone()).map_err(|e| Error::msg(e.to_string()))?);

    // attach an ingress to an engine, with the RequestHandler using the metrics struct
    let ingress = Ingress::for_engine(RequestHandler::new(metrics.clone()))?;

    // make the ingress discoverable via a component service
    // we must first create a service, then we can attach one more more endpoints
    /*
    service, <namespace>_<service>__<metric_name>
    component, <namespace>_<component>__<metric_name>
    endpoint, <namespace>_<service>_<component>_<endpoint>__<metric_name>
        */
    runtime
        .namespace(DEFAULT_NAMESPACE)?
        .component("backend")?
        .service_builder()
        .create()
        .await?
        .endpoint("generate")
        .endpoint_builder()
        .stats_handler(|stats| {
            println!("stats: {:?}", stats);
            let stats = MyStats { val: 10 };
            serde_json::to_value(stats).unwrap()
        })
        // TODO(keiven): add metrics and healthcheck handlers
        //.metrics_handler(metrics.backend().prometheus_format_str().to_string())
        //.healthcheck_handler(some_function_here, metrics)
        .handler(ingress)
        .start()
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::profiling::PrometheusRegistry;

    #[test]
    fn test_service_metrics_with_profiling_backend() {
        println!("=== ServiceMetrics with Profiling Backend Test ===");

        // Create a Prometheus backend using the profiling module
        let metrics_backend =
            Arc::new(PrometheusRegistry::new("test_service")) as Arc<dyn MetricsRegistry>;

        // Create ServiceMetrics using the new struct
        let service_metrics = ExampleHTTPMetrics::new(metrics_backend.clone()).unwrap();

        println!("Created ServiceMetrics with profiling backend");

        // Test the metrics functionality
        service_metrics.request_counter.inc();
        service_metrics.request_counter.inc_by(2);
        service_metrics.active_requests_gauge.set(5.0);
        service_metrics.active_requests_gauge.inc(1.0);
        service_metrics.request_duration_histogram.observe(0.1);
        service_metrics.request_duration_histogram.observe(0.25);

        // Verify the metrics values
        assert_eq!(
            service_metrics.request_counter.get_value(),
            3,
            "Request counter should be 3"
        );
        assert_eq!(
            service_metrics.active_requests_gauge.get_value(),
            6.0,
            "Active requests should be 6.0"
        );
        assert_eq!(
            service_metrics.request_duration_histogram.get_count(),
            2,
            "Should have 2 duration observations"
        );
        assert!(
            (service_metrics.request_duration_histogram.get_sum() - 0.35).abs() < f64::EPSILON,
            "Sum should be 0.35"
        );

        // Get the Prometheus metrics output
        match service_metrics.backend().root_prometheus_format_str() {
            Ok(metrics) => {
                println!("Prometheus metrics output:");
                println!("{}", metrics);

                // Verify the output contains expected metric names
                assert!(
                    metrics.contains("test_service_service_requests_total"),
                    "Should contain request counter"
                );
                assert!(
                    metrics.contains("test_service_service_active_requests"),
                    "Should contain active requests gauge"
                );
                assert!(
                    metrics.contains("test_service_service_request_duration_seconds"),
                    "Should contain duration histogram"
                );
            }
            Err(e) => {
                panic!("Failed to get metrics: {}", e);
            }
        }

        println!("=== ServiceMetrics Test Complete ===");
    }
}
