// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use prometheus::Encoder;

use crate::metrics::{build_prometheus_metric, prometheus_names::{sanitize_prometheus_label, sanitize_prometheus_name}, MetricFactory, PrometheusMetric};

pub trait StandaloneRegistry: Send + Sync {
    /// Register the collector in a single in-process registry (no hierarchy).
    fn register_collector(
        &self,
        collector: Box<dyn prometheus::core::Collector>,
    ) -> anyhow::Result<()>;

    fn gather(&self) -> Vec<prometheus::proto::MetricFamily>;

    fn metrics_prefix(&self) -> &str;

    fn get_metrics_name(&self, metric_name: &str) -> String;

    fn create_metric<M: PrometheusMetric>(
        &self,
        metric_name: &str,
        metric_desc: &str,
        labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
        const_labels: Option<&[&str]>,
    ) -> anyhow::Result<M> {
        let metric_name_sanitized = sanitize_prometheus_name(&self.get_metrics_name(metric_name))?;
        let labels: Vec<(String, String)> = labels
            .iter()
            .map(|(k, v)| Ok(((*k).to_string(), sanitize_prometheus_label(v)?)))
            .collect::<anyhow::Result<_>>()?;

        let metric = build_prometheus_metric::<M>(
            &metric_name_sanitized, metric_desc, labels, buckets, const_labels
        )?;

        let coll: Box<dyn prometheus::core::Collector> = Box::new(metric.clone());
        self.register_collector(coll)?;
        Ok(metric)
    }

    fn prometheus_metrics_fmt(&self) -> anyhow::Result<String> {
        let encoder = prometheus::TextEncoder::new();
        let families = self.gather();
        let mut buf = Vec::new();
        encoder.encode(&families, &mut buf)?;
        Ok(String::from_utf8(buf)?)
    }
}

/// A tiny wrapper around a single `prometheus::Registry`.
pub struct StandaloneMetrics {
    metrics_prefix: String,
    registry: prometheus::Registry,
}
impl StandaloneMetrics {
    pub fn new(metrics_prefix: &str) -> Self {
        Self {
            metrics_prefix: metrics_prefix.to_string(),
            registry: prometheus::Registry::new()
        }
    }
}

impl StandaloneRegistry for StandaloneMetrics {
    fn register_collector(
        &self,
        collector: Box<dyn prometheus::core::Collector>,
    ) -> anyhow::Result<()> {
        self.registry.register(collector).map_err(|e| anyhow::anyhow!(e))
    }

    fn gather(&self) -> Vec<prometheus::proto::MetricFamily> {
        self.registry.gather()
    }

    fn metrics_prefix(&self) -> &str {
        &self.metrics_prefix
    }

    fn get_metrics_name(&self, metric_name: &str) -> String {
        format!("{}_{}", &self.metrics_prefix, metric_name)
    }
}

impl MetricFactory for StandaloneMetrics {
    fn create_metric<M: PrometheusMetric>(
        &self,
        name: &str,
        description: &str,
        user_labels: &[(&str, &str)],
        buckets: Option<Vec<f64>>,
        vec_label_names: Option<&[&str]>,
    ) -> anyhow::Result<M> {
        StandaloneRegistry::create_metric(self, name, description, user_labels, buckets, vec_label_names)
    }

    fn prometheus_metrics_fmt(&self) -> anyhow::Result<String> {
        StandaloneRegistry::prometheus_metrics_fmt(self)
    }
}
