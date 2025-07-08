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

use once_cell::sync::OnceCell;
use prometheus::{IntCounterVec, Opts, Registry, register_int_counter_vec};
use std::sync::Arc;

static GLOBAL_METRICS: OnceCell<Arc<GlobalMetrics>> = OnceCell::new();

#[derive(Debug)]
pub struct GlobalMetrics {
    operation_timing_counter: IntCounterVec,
}

impl GlobalMetrics {
    pub fn new() -> Self {
        let operation_timing_counter = register_int_counter_vec!(
            "dynamo_operation_timing_microseconds",
            "Cumulative time spent in operations in microseconds",
            &["operation", "component"]
        )
        .unwrap();

        Self {
            operation_timing_counter,
        }
    }

    pub fn record_operation_timing(&self, operation: &str, component: &str, duration_micros: u64) {
        self.operation_timing_counter
            .with_label_values(&[operation, component])
            .inc_by(duration_micros);
    }


}

/// Initialize global metrics
pub fn init_global_metrics() -> Arc<GlobalMetrics> {
    let metrics = Arc::new(GlobalMetrics::new());
    GLOBAL_METRICS
        .set(metrics.clone())
        .expect("Global metrics already initialized");
    tracing::info!("===== Global timing metrics initialized and registered with default Prometheus registry =====");
    metrics
}

/// Get global metrics
pub fn get_global_metrics() -> Option<&'static Arc<GlobalMetrics>> {
    GLOBAL_METRICS.get()
}

/// Gather all metrics from the default registry
/// This function can be called from other crates to get all registered metrics
pub fn gather_all_metrics() -> Vec<prometheus::proto::MetricFamily> {
    prometheus::gather()
}

/// Macro to time an operation and record it in global metrics
#[macro_export]
macro_rules! time_global_operation {
    ($operation:expr, $component:expr, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();
        let duration_micros = duration.as_micros() as u64;
        
        if let Some(metrics) = $crate::global_metrics::get_global_metrics() {
            metrics.record_operation_timing($operation, $component, duration_micros);
        }
        
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::Registry;

    #[test]
    fn test_global_metrics() {
        // Initialize metrics
        let metrics = init_global_metrics();
        
        // Test timing an operation
        time_global_operation!("test_op", "test_component", {
            std::thread::sleep(std::time::Duration::from_millis(10));
        });
        
        // Verify the metric was recorded by checking the default registry
        let metric_families = prometheus::gather();
        assert!(!metric_families.is_empty());
    }
} 