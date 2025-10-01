// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Block Transfer Benchmarking System 📊
//!
//! This module provides comprehensive benchmarking and metrics collection for block transfers,
//! specifically designed to measure the impact of layout changes on transfer efficiency.

pub mod collector;
pub mod hooks;
pub mod reporter;

pub use collector::{AggregatedMetrics, CollectorBuilder, MetricsCollector};
pub use hooks::*;
pub use reporter::{BenchmarkReport, ComparisonMetrics};

use anyhow;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime};

use crate::block_manager::layout::LayoutType;
use crate::block_manager::storage::StorageType;

/// Transfer paths that we benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferPath {
    /// Host to Disk (GDS) transfers
    HostToDisk,
    /// Host to Device (G4) transfers
    HostToDevice,
    /// Device to Host transfers
    DeviceToHost,
    /// Disk to Device transfers
    DiskToDevice,
    /// Host to Host (memcpy) transfers
    HostToHost,
    /// Device to Device transfers
    DeviceToDevice,
}

impl TransferPath {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            TransferPath::HostToDisk => "Host → Disk (GDS)",
            TransferPath::HostToDevice => "Host → Device (G4)",
            TransferPath::DeviceToHost => "Device → Host",
            TransferPath::DiskToDevice => "Disk → Device",
            TransferPath::HostToHost => "Host → Host (memcpy)",
            TransferPath::DeviceToDevice => "Device → Device",
        }
    }

    /// Determine transfer path from storage types
    pub fn from_storage_types(from: &StorageType, to: &StorageType) -> Self {
        match (from, to) {
            (StorageType::System | StorageType::Pinned, StorageType::Disk(_)) => {
                TransferPath::HostToDisk
            }
            (StorageType::System | StorageType::Pinned, StorageType::Device(_)) => {
                TransferPath::HostToDevice
            }
            (StorageType::Device(_), StorageType::System | StorageType::Pinned) => {
                TransferPath::DeviceToHost
            }
            (StorageType::Disk(_), StorageType::Device(_)) => TransferPath::DiskToDevice,
            (
                StorageType::System | StorageType::Pinned,
                StorageType::System | StorageType::Pinned,
            ) => TransferPath::HostToHost,
            (StorageType::Device(_), StorageType::Device(_)) => TransferPath::DeviceToDevice,
            _ => TransferPath::HostToHost, // Default fallback
        }
    }
}

/// Metrics for a specific transfer path and layout combination
#[derive(Debug, Serialize, Deserialize)]
pub struct TransferMetrics {
    /// Total number of transfers
    pub transfer_count: AtomicUsize,
    /// Total bytes transferred
    pub total_bytes: AtomicU64,
    /// Total number of blocks transferred
    pub total_blocks: AtomicUsize,
    /// Minimum transfer size observed
    pub min_transfer_size: AtomicU64,
    /// Maximum transfer size observed
    pub max_transfer_size: AtomicU64,
    /// Start time for rate calculations
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
}

impl TransferMetrics {
    pub fn new() -> Self {
        Self {
            transfer_count: AtomicUsize::new(0),
            total_bytes: AtomicU64::new(0),
            total_blocks: AtomicUsize::new(0),
            min_transfer_size: AtomicU64::new(u64::MAX),
            max_transfer_size: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record a transfer
    pub fn record_transfer(&self, bytes: u64, blocks: usize) {
        self.transfer_count.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.total_blocks.fetch_add(blocks, Ordering::Relaxed);

        // Update min/max
        let current_min = self.min_transfer_size.load(Ordering::Relaxed);
        if bytes < current_min {
            self.min_transfer_size.store(bytes, Ordering::Relaxed);
        }

        let current_max = self.max_transfer_size.load(Ordering::Relaxed);
        if bytes > current_max {
            self.max_transfer_size.store(bytes, Ordering::Relaxed);
        }
    }

    /// Get average transfer size
    pub fn avg_transfer_size(&self) -> f64 {
        let count = self.transfer_count.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            self.total_bytes.load(Ordering::Relaxed) as f64 / count as f64
        }
    }

    /// Get transfer rate in bytes per second
    pub fn transfer_rate_bps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            0.0
        } else {
            self.total_bytes.load(Ordering::Relaxed) as f64 / elapsed
        }
    }

    /// Get block transfer rate (blocks per second)
    pub fn block_rate_bps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            0.0
        } else {
            self.total_blocks.load(Ordering::Relaxed) as f64 / elapsed
        }
    }
}

/// Key for indexing metrics by path and layout
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetricsKey {
    pub path: TransferPath,
    pub layout: LayoutType,
}

/// Main benchmarking collector
#[derive(Debug, Clone)]
pub struct TransferBenchmark {
    /// Benchmark session name
    pub session_name: String,
    /// Metrics by transfer path and layout type
    pub metrics: Arc<RwLock<HashMap<MetricsKey, TransferMetrics>>>,
    /// Session start time
    pub session_start: SystemTime,
    /// Whether benchmarking is enabled (can be toggled)
    pub enabled: Arc<std::sync::atomic::AtomicBool>,
}

impl TransferBenchmark {
    /// Create new benchmark session
    pub fn new(session_name: impl Into<String>) -> Self {
        Self {
            session_name: session_name.into(),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            session_start: SystemTime::now(),
            enabled: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Enable/disable benchmarking
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if benchmarking is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Record a transfer
    pub fn record_transfer(
        &self,
        path: TransferPath,
        layout: LayoutType,
        bytes: u64,
        blocks: usize,
    ) {
        if !self.is_enabled() {
            return;
        }

        let key = MetricsKey { path, layout };

        // Get or create metrics for this key
        {
            let metrics_map = self.metrics.read();
            if let Some(metrics) = metrics_map.get(&key) {
                metrics.record_transfer(bytes, blocks);
                return;
            }
        }

        // Need to create new metrics entry
        let mut metrics_map = self.metrics.write();
        let metrics = metrics_map.entry(key).or_insert_with(TransferMetrics::new);
        metrics.record_transfer(bytes, blocks);
    }

    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> HashMap<MetricsKey, TransferMetrics> {
        let metrics_map = self.metrics.read();

        // Create a snapshot with current values
        metrics_map
            .iter()
            .map(|(key, metrics)| {
                let snapshot = TransferMetrics {
                    transfer_count: AtomicUsize::new(
                        metrics.transfer_count.load(Ordering::Relaxed),
                    ),
                    total_bytes: AtomicU64::new(metrics.total_bytes.load(Ordering::Relaxed)),
                    total_blocks: AtomicUsize::new(metrics.total_blocks.load(Ordering::Relaxed)),
                    min_transfer_size: AtomicU64::new(
                        metrics.min_transfer_size.load(Ordering::Relaxed),
                    ),
                    max_transfer_size: AtomicU64::new(
                        metrics.max_transfer_size.load(Ordering::Relaxed),
                    ),
                    start_time: metrics.start_time,
                };
                (key.clone(), snapshot)
            })
            .collect()
    }

    /// Reset all metrics
    pub fn reset(&self) {
        let mut metrics_map = self.metrics.write();
        metrics_map.clear();
    }

    /// Get session duration
    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed().unwrap_or(Duration::ZERO)
    }

    /// Get a summary string of benchmark status
    pub fn summary(&self) -> String {
        let metrics = self.get_metrics();
        let total_transfers: usize = metrics
            .values()
            .map(|m| m.transfer_count.load(std::sync::atomic::Ordering::Relaxed))
            .sum();
        let total_bytes: u64 = metrics
            .values()
            .map(|m| m.total_bytes.load(std::sync::atomic::Ordering::Relaxed))
            .sum();

        format!(
            "Benchmark '{}': {} transfers, {} total bytes across {} paths",
            self.session_name,
            total_transfers,
            total_bytes,
            metrics.len()
        )
    }

    /// Generate a formatted report
    pub fn generate_report(&self) -> String {
        let report = BenchmarkReport::from_benchmark(self);
        report.generate_report()
    }

    /// Save benchmark data to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let report = BenchmarkReport::from_benchmark(self);
        let json_data = serde_json::to_string_pretty(&report)?;
        fs::write(path, json_data)?;
        Ok(())
    }

    /// Load benchmark data from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let json_data = fs::read_to_string(path)?;
        let report: BenchmarkReport = serde_json::from_str(&json_data)?;

        // Convert BenchmarkReport back to TransferBenchmark
        let benchmark = TransferBenchmark {
            session_name: report.session_name,
            session_start: SystemTime::now() - report.session_duration,
            metrics: Arc::new(RwLock::new(report.metrics)),
            enabled: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        };

        Ok(benchmark)
    }

    /// Get default save path for benchmark data
    pub fn default_save_path() -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push("dynamo_benchmark_session.json");
        path
    }

    /// Get named save path for specific session
    pub fn named_save_path(session_name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("dynamo_benchmark_{}.json", session_name));
        path
    }

    /// Get path for active session tracker file
    pub fn active_session_path() -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push("dynamo_benchmark_active_session.txt");
        path
    }

    /// Save active session name to file and set environment variable
    pub fn save_active_session(session_name: &str) -> anyhow::Result<()> {
        fs::write(Self::active_session_path(), session_name)?;
        // Also set environment variable for child processes
        unsafe {
            std::env::set_var("DYNAMO_BENCHMARK_SESSION", session_name);
        }
        Ok(())
    }

    /// Load active session name from file or environment variable
    pub fn load_active_session() -> Option<String> {
        // First check environment variable
        if let Ok(session_name) = std::env::var("DYNAMO_BENCHMARK_SESSION") {
            return Some(session_name);
        }
        // Then try file
        fs::read_to_string(Self::active_session_path()).ok()
    }
}

/// Global benchmark instance
static GLOBAL_BENCHMARK: parking_lot::RwLock<Option<TransferBenchmark>> =
    parking_lot::RwLock::new(None);

/// Initialize global benchmark (creates new or loads existing)
pub fn init_benchmark(session_name: impl Into<String>) -> anyhow::Result<()> {
    let session_name = session_name.into();
    let save_path = TransferBenchmark::named_save_path(&session_name);

    let benchmark = if save_path.exists() {
        // Try to load existing benchmark, fallback to new if load fails
        TransferBenchmark::load_from_file(&save_path)
            .unwrap_or_else(|_| TransferBenchmark::new(&session_name))
    } else {
        TransferBenchmark::new(&session_name)
    };

    // Save immediately to establish the file
    benchmark.save_to_file(&save_path)?;

    // Mark this session as active
    TransferBenchmark::save_active_session(&session_name)?;

    let mut global = GLOBAL_BENCHMARK.write();
    *global = Some(benchmark);
    Ok(())
}

/// Get global benchmark instance (loads from active session if needed)
pub fn global_benchmark() -> Option<TransferBenchmark> {
    {
        let global = GLOBAL_BENCHMARK.read();
        if global.is_some() {
            return global.as_ref().cloned();
        }
    }

    // If no in-memory instance, try to load the active session
    if let Some(active_session) = TransferBenchmark::load_active_session() {
        if let Ok(benchmark) = load_benchmark(&active_session) {
            let mut global = GLOBAL_BENCHMARK.write();
            *global = Some(benchmark.clone());
            return Some(benchmark);
        }
    }

    None
}

/// Load a specific benchmark session by name
pub fn load_benchmark(session_name: &str) -> anyhow::Result<TransferBenchmark> {
    let save_path = TransferBenchmark::named_save_path(session_name);
    TransferBenchmark::load_from_file(&save_path)
}

/// Save global benchmark to file
pub fn save_global_benchmark() -> anyhow::Result<()> {
    if let Some(benchmark) = global_benchmark() {
        let save_path = TransferBenchmark::named_save_path(&benchmark.session_name);
        benchmark.save_to_file(save_path)?;
    }
    Ok(())
}

/// Record transfer in global benchmark (convenience function)
pub fn record_transfer(path: TransferPath, layout: LayoutType, bytes: u64, blocks: usize) {
    if let Some(benchmark) = global_benchmark() {
        benchmark.record_transfer(path, layout, bytes, blocks);
        // Auto-save after recording
        let _ = save_global_benchmark();
    }
}

/// Reset global benchmark
pub fn reset_global_benchmark() -> anyhow::Result<()> {
    if let Some(benchmark) = global_benchmark() {
        let save_path = TransferBenchmark::named_save_path(&benchmark.session_name);
        if save_path.exists() {
            fs::remove_file(save_path)?;
        }
    }

    let mut global = GLOBAL_BENCHMARK.write();
    *global = None;
    Ok(())
}
