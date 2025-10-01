// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Benchmark Reporting 📈
//!
//! This module provides comprehensive reporting capabilities for transfer benchmarks,
//! including comparative analysis, performance insights, and formatted output.

use super::{MetricsKey, TransferBenchmark, TransferMetrics, TransferPath};
use crate::block_manager::layout::LayoutType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write;

/// Comparative metrics between two benchmark runs
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub transfer_count_change: f64,
    pub total_bytes_change: f64,
    pub avg_transfer_size_change: f64,
    pub transfer_rate_change: f64,
    pub block_rate_change: f64,
}

impl ComparisonMetrics {
    pub fn compare(before: &TransferMetrics, after: &TransferMetrics) -> Self {
        let before_count = before
            .transfer_count
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let after_count = after
            .transfer_count
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let before_bytes = before
            .total_bytes
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let after_bytes = after.total_bytes.load(std::sync::atomic::Ordering::Relaxed) as f64;

        Self {
            transfer_count_change: Self::percent_change(before_count, after_count),
            total_bytes_change: Self::percent_change(before_bytes, after_bytes),
            avg_transfer_size_change: Self::percent_change(
                before.avg_transfer_size(),
                after.avg_transfer_size(),
            ),
            transfer_rate_change: Self::percent_change(
                before.transfer_rate_bps(),
                after.transfer_rate_bps(),
            ),
            block_rate_change: Self::percent_change(
                before.block_rate_bps(),
                after.block_rate_bps(),
            ),
        }
    }

    fn percent_change(before: f64, after: f64) -> f64 {
        if before == 0.0 {
            if after == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            ((after - before) / before) * 100.0
        }
    }
}

/// Benchmark report with formatted output
#[derive(Debug)]
pub struct BenchmarkReport {
    pub session_name: String,
    pub metrics: HashMap<MetricsKey, TransferMetrics>,
    pub session_duration: std::time::Duration,
}

impl BenchmarkReport {
    pub fn from_benchmark(benchmark: &TransferBenchmark) -> Self {
        Self {
            session_name: benchmark.session_name.clone(),
            metrics: benchmark.get_metrics(),
            session_duration: benchmark.session_duration(),
        }
    }

    /// Generate formatted report
    pub fn generate_report(&self) -> String {
        let mut output = String::new();

        writeln!(output, "🚀 Transfer Benchmark Report").unwrap();
        writeln!(output, "═══════════════════════════════════════").unwrap();
        writeln!(output, "Session: {}", self.session_name).unwrap();
        writeln!(
            output,
            "Duration: {:.2}s",
            self.session_duration.as_secs_f64()
        )
        .unwrap();
        writeln!(output).unwrap();

        if self.metrics.is_empty() {
            writeln!(output, "❌ No transfer data collected").unwrap();
            return output;
        }

        // Group by transfer path for better organization
        let mut by_path: HashMap<TransferPath, Vec<(&MetricsKey, &TransferMetrics)>> =
            HashMap::new();
        for (key, metrics) in &self.metrics {
            by_path.entry(key.path).or_default().push((key, metrics));
        }

        for (path, entries) in by_path {
            writeln!(output, "📊 {}", path.description()).unwrap();
            writeln!(output, "───────────────────────────────────────").unwrap();

            for (_key, metrics) in entries {
                let count = metrics
                    .transfer_count
                    .load(std::sync::atomic::Ordering::Relaxed);
                let total_bytes = metrics
                    .total_bytes
                    .load(std::sync::atomic::Ordering::Relaxed);
                let total_blocks = metrics
                    .total_blocks
                    .load(std::sync::atomic::Ordering::Relaxed);
                let min_size = metrics
                    .min_transfer_size
                    .load(std::sync::atomic::Ordering::Relaxed);
                let max_size = metrics
                    .max_transfer_size
                    .load(std::sync::atomic::Ordering::Relaxed);

                if count == 0 {
                    continue;
                }

                writeln!(output, "    Transfers:     {:>12}", format_number(count)).unwrap();
                writeln!(
                    output,
                    "    Total Bytes:   {:>12}",
                    format_bytes(total_bytes)
                )
                .unwrap();
                writeln!(
                    output,
                    "    Total Blocks:  {:>12}",
                    format_number(total_blocks)
                )
                .unwrap();
                writeln!(
                    output,
                    "    Avg Size:      {:>12}",
                    format_bytes(metrics.avg_transfer_size() as u64)
                )
                .unwrap();
                writeln!(
                    output,
                    "    Min Size:      {:>12}",
                    format_bytes(if min_size == u64::MAX { 0 } else { min_size })
                )
                .unwrap();
                writeln!(output, "    Max Size:      {:>12}", format_bytes(max_size)).unwrap();
                writeln!(
                    output,
                    "    Transfer Rate: {:>12}/s",
                    format_bytes(metrics.transfer_rate_bps() as u64)
                )
                .unwrap();
                writeln!(
                    output,
                    "    Block Rate:    {:>12.1} blocks/s",
                    metrics.block_rate_bps()
                )
                .unwrap();
                writeln!(output).unwrap();
            }
        }

        // Summary section
        writeln!(output, "📈 Summary").unwrap();
        writeln!(output, "───────────────────────────────────────").unwrap();

        let total_transfers: usize = self
            .metrics
            .values()
            .map(|m| m.transfer_count.load(std::sync::atomic::Ordering::Relaxed))
            .sum();
        let total_bytes: u64 = self
            .metrics
            .values()
            .map(|m| m.total_bytes.load(std::sync::atomic::Ordering::Relaxed))
            .sum();
        let total_blocks: usize = self
            .metrics
            .values()
            .map(|m| m.total_blocks.load(std::sync::atomic::Ordering::Relaxed))
            .sum();

        writeln!(
            output,
            "Total Transfers: {}",
            format_number(total_transfers)
        )
        .unwrap();
        writeln!(output, "Total Data:      {}", format_bytes(total_bytes)).unwrap();
        writeln!(output, "Total Blocks:    {}", format_number(total_blocks)).unwrap();
        writeln!(
            output,
            "Overall Rate:    {}/s",
            format_bytes((total_bytes as f64 / self.session_duration.as_secs_f64()) as u64)
        )
        .unwrap();

        output
    }

    /// Generate comparison report between two benchmark runs
    pub fn compare_with(&self, other: &BenchmarkReport) -> String {
        let mut output = String::new();

        writeln!(output, "🔄 Benchmark Comparison Report").unwrap();
        writeln!(output, "═══════════════════════════════════════").unwrap();
        writeln!(
            output,
            "Before: {} ({:.2}s)",
            other.session_name,
            other.session_duration.as_secs_f64()
        )
        .unwrap();
        writeln!(
            output,
            "After:  {} ({:.2}s)",
            self.session_name,
            self.session_duration.as_secs_f64()
        )
        .unwrap();
        writeln!(output).unwrap();

        // Find common keys
        let mut common_keys: Vec<_> = self
            .metrics
            .keys()
            .filter(|key| other.metrics.contains_key(key))
            .collect();
        common_keys.sort_by_key(|k| {
            let path_order = match k.path {
                TransferPath::HostToDisk => 0,
                TransferPath::HostToDevice => 1,
                TransferPath::DeviceToHost => 2,
                TransferPath::DeviceToDevice => 3,
                TransferPath::DiskToDevice => 4,
                TransferPath::HostToHost => 5,
            };
            let layout_order = match k.layout {
                LayoutType::FullyContiguous => 0,
                LayoutType::LayerSeparate {
                    outer_contiguous: false,
                } => 1,
                LayoutType::LayerSeparate {
                    outer_contiguous: true,
                } => 2,
            };
            (path_order, layout_order)
        });

        if common_keys.is_empty() {
            writeln!(output, "❌ No common transfer paths found for comparison").unwrap();
            return output;
        }

        for key in &common_keys {
            let before_metrics = &other.metrics[key];
            let after_metrics = &self.metrics[key];
            let comparison = ComparisonMetrics::compare(before_metrics, after_metrics);

            writeln!(output, "📊 {} - {:?}", key.path.description(), key.layout).unwrap();
            writeln!(output, "───────────────────────────────────────").unwrap();

            writeln!(
                output,
                "  Transfer Count: {} → {} ({:+.1}%)",
                format_number(
                    before_metrics
                        .transfer_count
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
                format_number(
                    after_metrics
                        .transfer_count
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
                comparison.transfer_count_change
            )
            .unwrap();

            writeln!(
                output,
                "  Total Bytes:    {} → {} ({:+.1}%)",
                format_bytes(
                    before_metrics
                        .total_bytes
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
                format_bytes(
                    after_metrics
                        .total_bytes
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
                comparison.total_bytes_change
            )
            .unwrap();

            writeln!(
                output,
                "  Avg Size:       {} → {} ({:+.1}%)",
                format_bytes(before_metrics.avg_transfer_size() as u64),
                format_bytes(after_metrics.avg_transfer_size() as u64),
                comparison.avg_transfer_size_change
            )
            .unwrap();

            writeln!(
                output,
                "  Transfer Rate:  {}/s → {}/s ({:+.1}%)",
                format_bytes(before_metrics.transfer_rate_bps() as u64),
                format_bytes(after_metrics.transfer_rate_bps() as u64),
                comparison.transfer_rate_change
            )
            .unwrap();

            writeln!(output).unwrap();
        }

        // Highlight significant improvements
        writeln!(output, "🎯 Key Insights").unwrap();
        writeln!(output, "───────────────────────────────────────").unwrap();

        let mut insights = Vec::new();

        for key in &common_keys {
            let before_metrics = &other.metrics[key];
            let after_metrics = &self.metrics[key];
            let comparison = ComparisonMetrics::compare(before_metrics, after_metrics);

            if comparison.avg_transfer_size_change > 10.0 {
                insights.push(format!("✅ {}: Average transfer size increased by {:.1}% (larger transfers = better efficiency)",
                    key.path.description(), comparison.avg_transfer_size_change));
            }

            if comparison.transfer_count_change < -10.0 {
                insights.push(format!(
                    "✅ {}: Transfer count reduced by {:.1}% (fewer transfers = less overhead)",
                    key.path.description(),
                    -comparison.transfer_count_change
                ));
            }

            if comparison.transfer_rate_change > 5.0 {
                insights.push(format!(
                    "✅ {}: Transfer rate improved by {:.1}%",
                    key.path.description(),
                    comparison.transfer_rate_change
                ));
            }
        }

        if insights.is_empty() {
            writeln!(output, "No significant performance changes detected.").unwrap();
        } else {
            for insight in insights {
                writeln!(output, "{}", insight).unwrap();
            }
        }

        output
    }

    /// Export metrics to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export metrics to CSV
    pub fn to_csv(&self) -> String {
        let mut output = String::new();

        // CSV header
        writeln!(output, "transfer_path,layout_type,transfer_count,total_bytes,total_blocks,avg_transfer_size,min_transfer_size,max_transfer_size,transfer_rate_bps,block_rate_bps").unwrap();

        for (key, metrics) in &self.metrics {
            let count = metrics
                .transfer_count
                .load(std::sync::atomic::Ordering::Relaxed);
            let total_bytes = metrics
                .total_bytes
                .load(std::sync::atomic::Ordering::Relaxed);
            let total_blocks = metrics
                .total_blocks
                .load(std::sync::atomic::Ordering::Relaxed);
            let min_size = metrics
                .min_transfer_size
                .load(std::sync::atomic::Ordering::Relaxed);
            let max_size = metrics
                .max_transfer_size
                .load(std::sync::atomic::Ordering::Relaxed);

            writeln!(
                output,
                "{:?},{:?},{},{},{},{:.2},{},{},{:.2},{:.2}",
                key.path,
                key.layout,
                count,
                total_bytes,
                total_blocks,
                metrics.avg_transfer_size(),
                if min_size == u64::MAX { 0 } else { min_size },
                max_size,
                metrics.transfer_rate_bps(),
                metrics.block_rate_bps()
            )
            .unwrap();
        }

        output
    }
}

impl Serialize for BenchmarkReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("BenchmarkReport", 3)?;
        state.serialize_field("session_name", &self.session_name)?;
        state.serialize_field(
            "session_duration_secs",
            &self.session_duration.as_secs_f64(),
        )?;

        // Convert metrics to serializable format
        let serializable_metrics: HashMap<String, HashMap<String, u64>> = self
            .metrics
            .iter()
            .map(|(key, metrics)| {
                let key_str = format!("{:?}_{:?}", key.path, key.layout);
                let mut values = HashMap::new();
                values.insert(
                    "transfer_count".to_string(),
                    metrics
                        .transfer_count
                        .load(std::sync::atomic::Ordering::Relaxed) as u64,
                );
                values.insert(
                    "total_bytes".to_string(),
                    metrics
                        .total_bytes
                        .load(std::sync::atomic::Ordering::Relaxed),
                );
                values.insert(
                    "total_blocks".to_string(),
                    metrics
                        .total_blocks
                        .load(std::sync::atomic::Ordering::Relaxed) as u64,
                );
                values.insert(
                    "min_transfer_size".to_string(),
                    metrics
                        .min_transfer_size
                        .load(std::sync::atomic::Ordering::Relaxed),
                );
                values.insert(
                    "max_transfer_size".to_string(),
                    metrics
                        .max_transfer_size
                        .load(std::sync::atomic::Ordering::Relaxed),
                );
                values.insert(
                    "avg_transfer_size".to_string(),
                    metrics.avg_transfer_size() as u64,
                );
                values.insert(
                    "transfer_rate_bps".to_string(),
                    metrics.transfer_rate_bps() as u64,
                );
                values.insert(
                    "block_rate_bps".to_string(),
                    metrics.block_rate_bps() as u64,
                );
                (key_str, values)
            })
            .collect();

        state.serialize_field("metrics", &serializable_metrics)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for BenchmarkReport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;
        use std::time::Duration;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            SessionName,
            SessionDurationSecs,
            Metrics,
        }

        struct BenchmarkReportVisitor;

        impl<'de> Visitor<'de> for BenchmarkReportVisitor {
            type Value = BenchmarkReport;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct BenchmarkReport")
            }

            fn visit_map<V>(self, mut map: V) -> Result<BenchmarkReport, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut session_name = None;
                let mut session_duration_secs = None;
                let mut serializable_metrics = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::SessionName => {
                            if session_name.is_some() {
                                return Err(de::Error::duplicate_field("session_name"));
                            }
                            session_name = Some(map.next_value()?);
                        }
                        Field::SessionDurationSecs => {
                            if session_duration_secs.is_some() {
                                return Err(de::Error::duplicate_field("session_duration_secs"));
                            }
                            session_duration_secs = Some(map.next_value()?);
                        }
                        Field::Metrics => {
                            if serializable_metrics.is_some() {
                                return Err(de::Error::duplicate_field("metrics"));
                            }
                            serializable_metrics =
                                Some(map.next_value::<HashMap<String, HashMap<String, u64>>>()?);
                        }
                    }
                }

                let session_name =
                    session_name.ok_or_else(|| de::Error::missing_field("session_name"))?;
                let session_duration_secs: f64 = session_duration_secs
                    .ok_or_else(|| de::Error::missing_field("session_duration_secs"))?;
                let serializable_metrics =
                    serializable_metrics.ok_or_else(|| de::Error::missing_field("metrics"))?;

                // Convert back to proper metrics format
                let mut metrics = HashMap::new();
                for (key_str, values) in serializable_metrics {
                    // Parse the key back - this is a simplified approach
                    // In practice, you might want a more robust parsing method
                    if let Some((_path_str, _layout_str)) = key_str.split_once('_') {
                        // For now, create dummy metrics since we can't perfectly reconstruct
                        // the atomic values and MetricsKey from the serialized format
                        // This is a limitation of the current serialization approach
                        let transfer_metrics = TransferMetrics::new();
                        transfer_metrics.transfer_count.store(
                            values.get("transfer_count").copied().unwrap_or(0) as usize,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        transfer_metrics.total_bytes.store(
                            values.get("total_bytes").copied().unwrap_or(0),
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        transfer_metrics.total_blocks.store(
                            values.get("total_blocks").copied().unwrap_or(0) as usize,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        transfer_metrics.min_transfer_size.store(
                            values.get("min_transfer_size").copied().unwrap_or(u64::MAX),
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        transfer_metrics.max_transfer_size.store(
                            values.get("max_transfer_size").copied().unwrap_or(0),
                            std::sync::atomic::Ordering::Relaxed,
                        );

                        // Create a dummy MetricsKey - this is a limitation
                        // In practice, you'd need to store the key in a more parseable format
                        let dummy_key = MetricsKey {
                            path: TransferPath::HostToDisk,      // Default value
                            layout: LayoutType::FullyContiguous, // Default value
                        };
                        metrics.insert(dummy_key, transfer_metrics);
                    }
                }

                Ok(BenchmarkReport {
                    session_name,
                    metrics,
                    session_duration: Duration::from_secs_f64(session_duration_secs),
                })
            }
        }

        const FIELDS: &'static [&'static str] =
            &["session_name", "session_duration_secs", "metrics"];
        deserializer.deserialize_struct("BenchmarkReport", FIELDS, BenchmarkReportVisitor)
    }
}

/// Format large numbers with commas
fn format_number<T: std::fmt::Display>(n: T) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();

    for (i, &ch) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }

    result
}

/// Format bytes with appropriate units
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(123), "123");
        assert_eq!(format_number(1234), "1,234");
        assert_eq!(format_number(1234567), "1,234,567");
    }
}
