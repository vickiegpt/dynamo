// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Advanced Metrics Collection 📊
//!
//! This module provides advanced metrics collection utilities including
//! sampling, aggregation, and background collection for production environments.

use super::{TransferBenchmark, TransferPath};
use crate::block_manager::layout::LayoutType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Configuration for the metrics collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorConfig {
    /// Sample rate (0.0 to 1.0) - fraction of transfers to record
    pub sample_rate: f64,
    /// Maximum number of metrics entries to keep in memory
    pub max_entries: usize,
    /// How often to flush metrics to storage/logs
    pub flush_interval: Duration,
    /// Whether to enable background collection
    pub background_collection: bool,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 1.0, // Record all transfers by default
            max_entries: 10_000,
            flush_interval: Duration::from_secs(60),
            background_collection: false,
        }
    }
}

/// A transfer event for background processing
#[derive(Debug, Clone)]
pub struct TransferEvent {
    pub path: TransferPath,
    pub layout: LayoutType,
    pub bytes: u64,
    pub blocks: usize,
    pub timestamp: Instant,
}

/// Advanced metrics collector with sampling and background processing
#[derive(Debug)]
pub struct MetricsCollector {
    config: CollectorConfig,
    benchmark: Arc<TransferBenchmark>,
    event_sender: Option<mpsc::UnboundedSender<TransferEvent>>,
    sample_counter: std::sync::atomic::AtomicU64,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(benchmark: Arc<TransferBenchmark>, config: CollectorConfig) -> Self {
        let event_sender = if config.background_collection {
            let (sender, receiver) = mpsc::unbounded_channel();

            // Spawn background processor
            let benchmark_clone = benchmark.clone();
            let config_clone = config.clone();
            tokio::spawn(async move {
                Self::background_processor(receiver, benchmark_clone, config_clone).await;
            });

            Some(sender)
        } else {
            None
        };

        Self {
            config,
            benchmark,
            event_sender,
            sample_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Record a transfer event (with sampling)
    pub fn record_transfer(
        &self,
        path: TransferPath,
        layout: LayoutType,
        bytes: u64,
        blocks: usize,
    ) {
        // Apply sampling
        if !self.should_sample() {
            return;
        }

        let event = TransferEvent {
            path,
            layout,
            bytes,
            blocks,
            timestamp: Instant::now(),
        };

        if let Some(sender) = &self.event_sender {
            // Background processing
            let _ = sender.send(event);
        } else {
            // Direct processing
            self.benchmark.record_transfer(path, layout, bytes, blocks);
        }
    }

    /// Check if we should sample this transfer
    fn should_sample(&self) -> bool {
        if self.config.sample_rate >= 1.0 {
            return true;
        }
        if self.config.sample_rate <= 0.0 {
            return false;
        }

        let counter = self
            .sample_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let sample_threshold = (1.0 / self.config.sample_rate) as u64;
        counter % sample_threshold == 0
    }

    /// Background processor for handling transfer events
    async fn background_processor(
        mut receiver: mpsc::UnboundedReceiver<TransferEvent>,
        benchmark: Arc<TransferBenchmark>,
        config: CollectorConfig,
    ) {
        let mut batch = Vec::new();
        let mut last_flush = Instant::now();

        while let Some(event) = receiver.recv().await {
            batch.push(event);

            // Flush if we've hit the batch size or time limit
            if batch.len() >= 100 || last_flush.elapsed() >= config.flush_interval {
                Self::process_batch(&batch, &benchmark);
                batch.clear();
                last_flush = Instant::now();
            }
        }

        // Process remaining events
        if !batch.is_empty() {
            Self::process_batch(&batch, &benchmark);
        }
    }

    /// Process a batch of transfer events
    fn process_batch(events: &[TransferEvent], benchmark: &TransferBenchmark) {
        for event in events {
            benchmark.record_transfer(event.path, event.layout, event.bytes, event.blocks);
        }
    }

    /// Get collector statistics
    pub fn get_stats(&self) -> CollectorStats {
        let metrics = self.benchmark.get_metrics();

        CollectorStats {
            total_events: self
                .sample_counter
                .load(std::sync::atomic::Ordering::Relaxed),
            unique_paths: metrics.len(),
            sample_rate: self.config.sample_rate,
            background_collection: self.config.background_collection,
            session_duration: self.benchmark.session_duration(),
        }
    }
}

/// Statistics about the collector itself
#[derive(Debug, Serialize, Deserialize)]
pub struct CollectorStats {
    pub total_events: u64,
    pub unique_paths: usize,
    pub sample_rate: f64,
    pub background_collection: bool,
    pub session_duration: Duration,
}

/// Aggregated metrics for analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub by_path: HashMap<TransferPath, PathMetrics>,
    pub by_layout: HashMap<LayoutType, LayoutMetrics>,
    pub overall: OverallMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PathMetrics {
    pub total_transfers: usize,
    pub total_bytes: u64,
    pub avg_transfer_size: f64,
    pub transfer_rate_bps: f64,
    pub layouts_used: Vec<LayoutType>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LayoutMetrics {
    pub total_transfers: usize,
    pub total_bytes: u64,
    pub avg_transfer_size: f64,
    pub paths_used: Vec<TransferPath>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OverallMetrics {
    pub total_transfers: usize,
    pub total_bytes: u64,
    pub unique_paths: usize,
    pub unique_layouts: usize,
    pub session_duration: Duration,
    pub overall_rate_bps: f64,
}

impl AggregatedMetrics {
    /// Create aggregated metrics from benchmark data
    pub fn from_benchmark(benchmark: &TransferBenchmark) -> Self {
        let metrics = benchmark.get_metrics();
        let session_duration = benchmark.session_duration();

        let mut by_path: HashMap<TransferPath, PathMetrics> = HashMap::new();
        let mut by_layout: HashMap<LayoutType, LayoutMetrics> = HashMap::new();

        let mut total_transfers = 0;
        let mut total_bytes = 0u64;

        for (key, metric) in &metrics {
            let transfers = metric
                .transfer_count
                .load(std::sync::atomic::Ordering::Relaxed);
            let bytes = metric
                .total_bytes
                .load(std::sync::atomic::Ordering::Relaxed);

            total_transfers += transfers;
            total_bytes += bytes;

            // Aggregate by path
            let path_metric = by_path.entry(key.path).or_insert_with(|| PathMetrics {
                total_transfers: 0,
                total_bytes: 0,
                avg_transfer_size: 0.0,
                transfer_rate_bps: 0.0,
                layouts_used: Vec::new(),
            });

            path_metric.total_transfers += transfers;
            path_metric.total_bytes += bytes;
            if !path_metric.layouts_used.contains(&key.layout) {
                path_metric.layouts_used.push(key.layout);
            }

            // Aggregate by layout
            let layout_metric = by_layout
                .entry(key.layout)
                .or_insert_with(|| LayoutMetrics {
                    total_transfers: 0,
                    total_bytes: 0,
                    avg_transfer_size: 0.0,
                    paths_used: Vec::new(),
                });

            layout_metric.total_transfers += transfers;
            layout_metric.total_bytes += bytes;
            if !layout_metric.paths_used.contains(&key.path) {
                layout_metric.paths_used.push(key.path);
            }
        }

        // Calculate averages and rates
        for metric in by_path.values_mut() {
            metric.avg_transfer_size = if metric.total_transfers > 0 {
                metric.total_bytes as f64 / metric.total_transfers as f64
            } else {
                0.0
            };
            metric.transfer_rate_bps = metric.total_bytes as f64 / session_duration.as_secs_f64();
        }

        for metric in by_layout.values_mut() {
            metric.avg_transfer_size = if metric.total_transfers > 0 {
                metric.total_bytes as f64 / metric.total_transfers as f64
            } else {
                0.0
            };
        }

        let overall = OverallMetrics {
            total_transfers,
            total_bytes,
            unique_paths: by_path.len(),
            unique_layouts: by_layout.len(),
            session_duration,
            overall_rate_bps: total_bytes as f64 / session_duration.as_secs_f64(),
        };

        Self {
            by_path,
            by_layout,
            overall,
        }
    }

    /// Generate a summary report
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("📈 Aggregated Metrics Summary\n");
        report.push_str("═══════════════════════════════════════\n");

        report.push_str(&format!(
            "Total Transfers: {}\n",
            self.overall.total_transfers
        ));
        report.push_str(&format!(
            "Total Data: {:.2} MB\n",
            self.overall.total_bytes as f64 / 1_048_576.0
        ));
        report.push_str(&format!("Unique Paths: {}\n", self.overall.unique_paths));
        report.push_str(&format!(
            "Unique Layouts: {}\n",
            self.overall.unique_layouts
        ));
        report.push_str(&format!(
            "Session Duration: {:.2}s\n",
            self.overall.session_duration.as_secs_f64()
        ));
        report.push_str(&format!(
            "Overall Rate: {:.2} MB/s\n",
            self.overall.overall_rate_bps / 1_048_576.0
        ));

        report.push_str("\n📊 By Transfer Path:\n");
        for (path, metrics) in &self.by_path {
            report.push_str(&format!(
                "  {}: {} transfers, {:.2} MB, {:.2} MB/s\n",
                path.description(),
                metrics.total_transfers,
                metrics.total_bytes as f64 / 1_048_576.0,
                metrics.transfer_rate_bps / 1_048_576.0
            ));
        }

        report.push_str("\n🏗️ By Layout Type:\n");
        for (layout, metrics) in &self.by_layout {
            report.push_str(&format!(
                "  {:?}: {} transfers, {:.2} MB\n",
                layout,
                metrics.total_transfers,
                metrics.total_bytes as f64 / 1_048_576.0
            ));
        }

        report
    }
}

/// Utility for creating production-ready collectors
pub struct CollectorBuilder {
    config: CollectorConfig,
}

impl CollectorBuilder {
    pub fn new() -> Self {
        Self {
            config: CollectorConfig::default(),
        }
    }

    pub fn sample_rate(mut self, rate: f64) -> Self {
        self.config.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn max_entries(mut self, max: usize) -> Self {
        self.config.max_entries = max;
        self
    }

    pub fn flush_interval(mut self, interval: Duration) -> Self {
        self.config.flush_interval = interval;
        self
    }

    pub fn background_collection(mut self, enabled: bool) -> Self {
        self.config.background_collection = enabled;
        self
    }

    pub fn build(self, benchmark: Arc<TransferBenchmark>) -> MetricsCollector {
        MetricsCollector::new(benchmark, self.config)
    }
}

impl Default for CollectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::layout::LayoutType;

    #[test]
    fn test_collector_config() {
        let config = CollectorConfig::default();
        assert_eq!(config.sample_rate, 1.0);
        assert_eq!(config.max_entries, 10_000);
        assert!(!config.background_collection);
    }

    #[test]
    fn test_collector_builder() {
        let builder = CollectorBuilder::new()
            .sample_rate(0.5)
            .max_entries(5000)
            .background_collection(true);

        assert_eq!(builder.config.sample_rate, 0.5);
        assert_eq!(builder.config.max_entries, 5000);
        assert!(builder.config.background_collection);
    }
}
