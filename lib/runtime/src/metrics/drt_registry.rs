// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use prometheus::Encoder;

use crate::{metrics::{build_prometheus_metric, prometheus_names::{build_component_metric_name, sanitize_prometheus_label}, MetricFactory, PrometheusMetric}, traits::DistributedRuntimeProvider};

// If set to true, then metrics will be labeled with the namespace, component, and endpoint labels.
// These labels are prefixed with "dynamo_" to avoid collisions with Kubernetes and other monitoring system labels.
pub const USE_AUTO_LABELS: bool = true;

pub mod drt_labels {
    pub const NAMESPACE: &str = "dynamo_namespace";
    pub const COMPONENT: &str = "dynamo_component";
    pub const ENDPOINT:  &str = "dynamo_endpoint";
}

pub trait DistributedRuntimeMetricsRegistry: Send + Sync + DistributedRuntimeProvider {
    // Get the name of this registry (without any hierarchy prefix)
    fn basename(&self) -> String;

    /// Retrieve the complete hierarchy and basename for this registry. Currently, the hierarchy for drt is an empty string,
    /// so we must account for the leading underscore. The existing code remains unchanged to accommodate any future
    /// scenarios where drt's prefix might be assigned a value.
    fn hierarchy(&self) -> String {
        [self.parent_hierarchy(), vec![self.basename()]]
            .concat()
            .join("_")
            .trim_start_matches('_')
            .to_string()
    }

    // Get the parent hierarchy for this registry (just the base names, NOT the flattened hierarchy key)
    fn parent_hierarchy(&self) -> Vec<String>;

    // ---- high-level generic builder (DRT-aware) ----
    fn create_metric<M: PrometheusMetric>(
        &self,
        metric_name: &str,
        metric_desc: &str,
        labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
        const_labels: Option<&[&str]>,
    ) -> anyhow::Result<M> {
        // forbid overriding the auto-injected labels
        super::validate_no_duplicate_label_keys(labels)?;

        let basename = self.basename();
        let parent_hierarchy = self.parent_hierarchy();

        let hierarchy = [parent_hierarchy.clone(), vec![basename.clone()]].concat();

        let metric_name = build_component_metric_name(metric_name);

        // Build updated_labels: auto-labels first, then `labels` + stored labels
        let mut updated_labels: Vec<(String, String)> = Vec::new();

        if USE_AUTO_LABELS {
            // Validate that user-provided labels don't conflict with auto-generated labels
            for (key, _) in labels {
                if *key == drt_labels::NAMESPACE || *key == drt_labels::COMPONENT || *key == drt_labels::ENDPOINT {
                    return Err(anyhow::anyhow!(
                        "Label '{}' is automatically added by auto_label feature and cannot be manually set",
                        key
                    ));
                }
            }

            // Add auto-generated labels with sanitized values
            if hierarchy.len() > 1 {
                let namespace = &hierarchy[1];
                if !namespace.is_empty() {
                    let valid_namespace = sanitize_prometheus_label(namespace)?;
                    if !valid_namespace.is_empty() {
                        updated_labels.push((drt_labels::NAMESPACE.to_string(), valid_namespace));
                    }
                }
            }
            if hierarchy.len() > 2 {
                let component = &hierarchy[2];
                if !component.is_empty() {
                    let valid_component = sanitize_prometheus_label(component)?;
                    if !valid_component.is_empty() {
                        updated_labels.push((drt_labels::COMPONENT.to_string(), valid_component));
                    }
                }
            }
            if hierarchy.len() > 3 {
                let endpoint = &hierarchy[3];
                if !endpoint.is_empty() {
                    let valid_endpoint = sanitize_prometheus_label(endpoint)?;
                    if !valid_endpoint.is_empty() {
                        updated_labels.push((drt_labels::ENDPOINT.to_string(), valid_endpoint));
                    }
                }
            }
        }

        // Add user labels
        updated_labels.extend(
            labels
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string())),
        );

        // Build Metrics (pure Prometheus)
        let prometheus_metric = build_prometheus_metric::<M>(
            &metric_name, metric_desc, updated_labels, buckets, const_labels
        )?;

        // Iterate over the DRT's registry and register this metric across all hierarchical levels.
        // The accumulated hierarchy is structured as: ["", "testnamespace", "testnamespace_testcomponent", "testnamespace_testcomponent_testendpoint"]
        // This accumulation is essential to differentiate between the names of children and grandchildren.
        // Build accumulated hierarchy and register metrics in a single loop
        // current_prefix accumulates the hierarchical path as we iterate through hierarchy
        // For example, if hierarchy = ["", "testnamespace", "testcomponent"], then:
        // - Iteration 1: current_prefix = "" (empty string from DRT)
        // - Iteration 2: current_prefix = "testnamespace"
        // - Iteration 3: current_prefix = "testnamespace_testcomponent"

        // Fan-out register across accumulated DRT keys
        let mut current_hierarchy = String::new();
        for name in &hierarchy {
            if !current_hierarchy.is_empty() && !name.is_empty() {
                current_hierarchy.push('_');
            }
            current_hierarchy.push_str(name);

            // Register metric at this hierarchical level using the new helper function
            let collector: Box<dyn prometheus::core::Collector> = Box::new(prometheus_metric.clone());
            self
                .drt()
                .add_prometheus_metric(&current_hierarchy, collector)?;
        }

        Ok(prometheus_metric)
    }

    fn prometheus_metrics_fmt(&self) -> anyhow::Result<String> {
        // Execute callbacks first to ensure any new metrics are added to the registry
        let callback_results = self.drt().execute_metrics_callbacks(&self.hierarchy());

        // Log any callback errors but continue
        for result in callback_results {
            if let Err(e) = result {
                tracing::error!("Error executing metrics callback: {}", e);
            }
        }

        // Get the Prometheus registry for this hierarchy
        let prometheus_registry = {
            let mut registry_entry = self.drt().hierarchy_to_metricsregistry.write().unwrap();
            registry_entry
                .entry(self.hierarchy())
                .or_default()
                .prometheus_registry
                .clone()
        };
        let metric_families = prometheus_registry.gather();
        let encoder = prometheus::TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}


impl<T: DistributedRuntimeMetricsRegistry> MetricFactory for T {
    fn create_metric<M: PrometheusMetric>(
        &self,
        metric_name: &str,
        metric_desc: &str,
        labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
        const_labels: Option<&[&str]>,
    ) -> anyhow::Result<M> {
        DistributedRuntimeMetricsRegistry::create_metric(self, metric_name, metric_desc, labels, buckets, const_labels)
    }

    fn prometheus_metrics_fmt(&self) -> anyhow::Result<String> {
        DistributedRuntimeMetricsRegistry::prometheus_metrics_fmt(self)
    }
}
