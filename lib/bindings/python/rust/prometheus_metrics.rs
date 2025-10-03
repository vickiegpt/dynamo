// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// NOTE: This file implements Python bindings for Prometheus metric types.
// It should be kept in sync with:
// - lib/bindings/python/src/dynamo/_metrics.pyi (Python type stubs - method signatures must match)
// - lib/runtime/src/metrics.rs (MetricsRegistry trait - metric types should align)
//
// When adding/modifying metric methods:
// 1. Update the Rust implementation here (#[pymethods])
// 2. Update the Python type stub in _metrics.pyi
// 3. Follow standard Prometheus API conventions (e.g., Counter.inc(), Gauge.set(), etc.)

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::rs;

/// Base struct for common metric fields shared across all metric types
#[derive(Clone)]
struct MetricBase {
    /// Metric name
    name: String,
    /// Help text / description
    help: String,
    /// Constant label values (key-value pairs that are always set for this metric)
    label_values: HashMap<String, String>,
    /// Whether this is a vector metric (has dynamic labels)
    is_vec: bool,
    /// Whether this is an integer-based metric (IntCounter, IntGauge, IntCounterVec, IntGaugeVec)
    is_int: bool,
}

impl MetricBase {
    fn new(name: String, help: String, is_vec: bool, is_int: bool) -> Self {
        Self {
            name,
            help,
            label_values: HashMap::new(),
            is_vec,
            is_int,
        }
    }

    // TODO: Implement with_label_values when needed for setting constant labels
    // fn with_label_values(mut self, labels: HashMap<String, String>) -> Self {
    //     self.label_values = labels;
    //     self
    // }
}

/// Python wrapper for Counter metric
#[pyclass]
pub struct Counter {
    base: MetricBase,
    counter: Arc<Mutex<Option<prometheus::Counter>>>,
}

/// Python wrapper for IntCounter metric
#[pyclass]
pub struct IntCounter {
    base: MetricBase,
    counter: Arc<Mutex<Option<prometheus::IntCounter>>>,
}

/// Python wrapper for CounterVec metric
#[pyclass]
pub struct CounterVec {
    base: MetricBase,
    label_names: Vec<String>,
    counter: Arc<Mutex<Option<prometheus::CounterVec>>>,
}

/// Python wrapper for IntCounterVec metric
#[pyclass]
pub struct IntCounterVec {
    base: MetricBase,
    label_names: Vec<String>,
    counter: Arc<Mutex<Option<prometheus::IntCounterVec>>>,
}

/// Python wrapper for Gauge metric
#[pyclass]
pub struct Gauge {
    base: MetricBase,
    gauge: Arc<Mutex<Option<prometheus::Gauge>>>,
}

/// Python wrapper for IntGauge metric
#[pyclass]
pub struct IntGauge {
    base: MetricBase,
    gauge: Arc<Mutex<Option<prometheus::IntGauge>>>,
}

/// Python wrapper for GaugeVec metric
#[pyclass]
pub struct GaugeVec {
    base: MetricBase,
    label_names: Vec<String>,
    gauge: Arc<Mutex<Option<prometheus::GaugeVec>>>,
}

/// Python wrapper for IntGaugeVec metric
#[pyclass]
pub struct IntGaugeVec {
    base: MetricBase,
    label_names: Vec<String>,
    gauge: Arc<Mutex<Option<prometheus::IntGaugeVec>>>,
}

/// Python wrapper for Histogram metric
#[pyclass]
pub struct Histogram {
    base: MetricBase,
    histogram: Arc<Mutex<Option<prometheus::Histogram>>>,
}

// ============================================================================
// Counter implementations
// ============================================================================

#[pymethods]
impl Counter {
    #[new]
    fn new(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, false),
            counter: Arc::new(Mutex::new(None)),
        }
    }

    /// Increment counter by 1
    fn inc(&self) -> PyResult<()> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            counter.inc();
        }
        Ok(())
    }

    /// Increment counter by value
    fn inc_by(&self, value: f64) -> PyResult<()> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            counter.inc_by(value);
        }
        Ok(())
    }

    /// Get counter value
    fn get(&self) -> PyResult<f64> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            Ok(counter.get())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Counter not yet registered",
            ))
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.base.name.clone()
    }

    #[getter]
    fn help(&self) -> String {
        self.base.help.clone()
    }

    #[getter]
    fn label_values(&self) -> HashMap<String, String> {
        self.base.label_values.clone()
    }

    #[getter]
    fn is_vec(&self) -> bool {
        self.base.is_vec
    }

    #[getter]
    fn is_int(&self) -> bool {
        self.base.is_int
    }
}

impl Counter {
    pub(crate) fn new_internal(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, false),
            counter: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_counter(&self, counter: prometheus::Counter) {
        let mut counter_opt = self.counter.lock().unwrap();
        *counter_opt = Some(counter);
    }
}

#[pymethods]
impl IntCounter {
    #[new]
    fn new(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, true),
            counter: Arc::new(Mutex::new(None)),
        }
    }

    /// Increment counter by 1
    fn inc(&self) -> PyResult<()> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            counter.inc();
        }
        Ok(())
    }

    /// Increment counter by value
    fn inc_by(&self, value: u64) -> PyResult<()> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            counter.inc_by(value);
        }
        Ok(())
    }

    /// Get counter value
    fn get(&self) -> PyResult<u64> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            Ok(counter.get())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "IntCounter not yet registered",
            ))
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.base.name.clone()
    }

    #[getter]
    fn help(&self) -> String {
        self.base.help.clone()
    }

    #[getter]
    fn label_values(&self) -> HashMap<String, String> {
        self.base.label_values.clone()
    }

    #[getter]
    fn is_vec(&self) -> bool {
        self.base.is_vec
    }

    #[getter]
    fn is_int(&self) -> bool {
        self.base.is_int
    }
}

impl IntCounter {
    pub(crate) fn new_internal(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, true),
            counter: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_counter(&self, counter: prometheus::IntCounter) {
        let mut counter_opt = self.counter.lock().unwrap();
        *counter_opt = Some(counter);
    }
}

#[pymethods]
impl CounterVec {
    #[new]
    fn new(name: String, label_names: Vec<String>) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), true, false),
            label_names,
            counter: Arc::new(Mutex::new(None)),
        }
    }

    /// Increment counter by 1 with labels
    fn inc(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            counter.with_label_values(&label_values).inc();
        }
        Ok(())
    }

    /// Increment counter by value with labels
    fn inc_by(&self, labels: HashMap<String, String>, value: f64) -> PyResult<()> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            counter.with_label_values(&label_values).inc_by(value);
        }
        Ok(())
    }

    /// Get counter value with labels
    fn get(&self, labels: HashMap<String, String>) -> PyResult<f64> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            Ok(counter.with_label_values(&label_values).get())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "CounterVec not yet registered",
            ))
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.base.name.clone()
    }

    #[getter]
    fn label_names(&self) -> Vec<String> {
        self.label_names.clone()
    }
}

impl CounterVec {
    pub(crate) fn new_internal(name: String, label_names: Vec<String>) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), true, false),
            label_names,
            counter: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_counter(&self, counter: prometheus::CounterVec) {
        let mut counter_opt = self.counter.lock().unwrap();
        *counter_opt = Some(counter);
    }
}

#[pymethods]
impl IntCounterVec {
    #[new]
    fn new(name: String, label_names: Vec<String>) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), true, true),
            label_names,
            counter: Arc::new(Mutex::new(None)),
        }
    }

    /// Increment counter by 1 with labels
    fn inc(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            counter.with_label_values(&label_values).inc();
        }
        Ok(())
    }

    /// Increment counter by value with labels
    fn inc_by(&self, labels: HashMap<String, String>, value: u64) -> PyResult<()> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            counter.with_label_values(&label_values).inc_by(value);
        }
        Ok(())
    }

    /// Get counter value with labels
    fn get(&self, labels: HashMap<String, String>) -> PyResult<u64> {
        let counter_opt = self.counter.lock().unwrap();
        if let Some(ref counter) = *counter_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            Ok(counter.with_label_values(&label_values).get())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "IntCounterVec not yet registered",
            ))
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.base.name.clone()
    }

    #[getter]
    fn label_names(&self) -> Vec<String> {
        self.label_names.clone()
    }
}

impl IntCounterVec {
    pub(crate) fn new_internal(name: String, label_names: Vec<String>) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), true, true),
            label_names,
            counter: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_counter(&self, counter: prometheus::IntCounterVec) {
        let mut counter_opt = self.counter.lock().unwrap();
        *counter_opt = Some(counter);
    }
}

// ============================================================================
// Gauge implementations
// ============================================================================

#[pymethods]
impl Gauge {
    #[new]
    fn new(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, false),
            gauge: Arc::new(Mutex::new(None)),
        }
    }

    /// Set gauge value
    fn set(&self, value: f64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.set(value);
        }
        Ok(())
    }

    /// Get gauge value
    fn get(&self) -> PyResult<f64> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            Ok(gauge.get())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Gauge not yet registered",
            ))
        }
    }

    /// Increment gauge by 1
    fn inc(&self) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.inc();
        }
        Ok(())
    }

    /// Increment gauge by value
    fn inc_by(&self, value: f64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.add(value);
        }
        Ok(())
    }

    /// Decrement gauge by 1
    fn dec(&self) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.dec();
        }
        Ok(())
    }

    /// Decrement gauge by value
    fn dec_by(&self, value: f64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.sub(value);
        }
        Ok(())
    }

    /// Add value to gauge
    fn add(&self, value: f64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.add(value);
        }
        Ok(())
    }

    /// Subtract value from gauge
    fn sub(&self, value: f64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.sub(value);
        }
        Ok(())
    }

    #[getter]
    fn name(&self) -> String {
        self.base.name.clone()
    }
}

impl Gauge {
    pub(crate) fn new_internal(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, false),
            gauge: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_gauge(&self, gauge: prometheus::Gauge) {
        let mut gauge_opt = self.gauge.lock().unwrap();
        *gauge_opt = Some(gauge);
    }
}

#[pymethods]
impl IntGauge {
    #[new]
    fn new(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, true),
            gauge: Arc::new(Mutex::new(None)),
        }
    }

    /// Set gauge value
    fn set(&self, value: i64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.set(value);
        }
        Ok(())
    }

    /// Get gauge value
    fn get(&self) -> PyResult<i64> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            Ok(gauge.get())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "IntGauge not yet registered",
            ))
        }
    }

    /// Increment gauge by 1
    fn inc(&self) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.inc();
        }
        Ok(())
    }

    /// Decrement gauge by 1
    fn dec(&self) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.dec();
        }
        Ok(())
    }

    /// Add value to gauge
    fn add(&self, value: i64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.add(value);
        }
        Ok(())
    }

    /// Subtract value from gauge
    fn sub(&self, value: i64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            gauge.sub(value);
        }
        Ok(())
    }

    #[getter]
    fn name(&self) -> String {
        self.base.name.clone()
    }
}

impl IntGauge {
    pub(crate) fn new_internal(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, true),
            gauge: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_gauge(&self, gauge: prometheus::IntGauge) {
        let mut gauge_opt = self.gauge.lock().unwrap();
        *gauge_opt = Some(gauge);
    }
}

#[pymethods]
impl GaugeVec {
    #[new]
    fn new(name: String, label_names: Vec<String>) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), true, false),
            label_names,
            gauge: Arc::new(Mutex::new(None)),
        }
    }

    /// Set gauge value with labels
    fn set(&self, value: f64, labels: HashMap<String, String>) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).set(value);
        }
        Ok(())
    }

    /// Get gauge value with labels
    fn get(&self, labels: HashMap<String, String>) -> PyResult<f64> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            Ok(gauge.with_label_values(&label_values).get())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "GaugeVec not yet registered",
            ))
        }
    }

    /// Increment gauge by 1 with labels
    fn inc(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).inc();
        }
        Ok(())
    }

    /// Decrement gauge by 1 with labels
    fn dec(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).dec();
        }
        Ok(())
    }

    /// Add value to gauge with labels
    fn add(&self, labels: HashMap<String, String>, value: f64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).add(value);
        }
        Ok(())
    }

    /// Subtract value from gauge with labels
    fn sub(&self, labels: HashMap<String, String>, value: f64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).sub(value);
        }
        Ok(())
    }

    #[getter]
    fn name(&self) -> String {
        self.base.name.clone()
    }

    #[getter]
    fn label_names(&self) -> Vec<String> {
        self.label_names.clone()
    }
}

impl GaugeVec {
    pub(crate) fn new_internal(name: String, label_names: Vec<String>) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), true, false),
            label_names,
            gauge: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_gauge(&self, gauge: prometheus::GaugeVec) {
        let mut gauge_opt = self.gauge.lock().unwrap();
        *gauge_opt = Some(gauge);
    }
}

#[pymethods]
impl IntGaugeVec {
    #[new]
    fn new(name: String, label_names: Vec<String>) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), true, true),
            label_names,
            gauge: Arc::new(Mutex::new(None)),
        }
    }

    /// Set gauge value with labels
    fn set(&self, value: i64, labels: HashMap<String, String>) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).set(value);
        }
        Ok(())
    }

    /// Get gauge value with labels
    fn get(&self, labels: HashMap<String, String>) -> PyResult<i64> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            Ok(gauge.with_label_values(&label_values).get())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "IntGaugeVec not yet registered",
            ))
        }
    }

    /// Increment gauge by 1 with labels
    fn inc(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).inc();
        }
        Ok(())
    }

    /// Decrement gauge by 1 with labels
    fn dec(&self, labels: HashMap<String, String>) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).dec();
        }
        Ok(())
    }

    /// Add value to gauge with labels
    fn add(&self, labels: HashMap<String, String>, value: i64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).add(value);
        }
        Ok(())
    }

    /// Subtract value from gauge with labels
    fn sub(&self, labels: HashMap<String, String>, value: i64) -> PyResult<()> {
        let gauge_opt = self.gauge.lock().unwrap();
        if let Some(ref gauge) = *gauge_opt {
            let label_values: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
            gauge.with_label_values(&label_values).sub(value);
        }
        Ok(())
    }

    #[getter]
    fn name(&self) -> String {
        self.base.name.clone()
    }

    #[getter]
    fn label_names(&self) -> Vec<String> {
        self.label_names.clone()
    }
}

impl IntGaugeVec {
    pub(crate) fn new_internal(name: String, label_names: Vec<String>) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), true, true),
            label_names,
            gauge: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_gauge(&self, gauge: prometheus::IntGaugeVec) {
        let mut gauge_opt = self.gauge.lock().unwrap();
        *gauge_opt = Some(gauge);
    }
}

// ============================================================================
// Histogram implementation
// ============================================================================

#[pymethods]
impl Histogram {
    #[new]
    fn new(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, false),
            histogram: Arc::new(Mutex::new(None)),
        }
    }

    /// Observe a value
    fn observe(&self, value: f64) -> PyResult<()> {
        let histogram_opt = self.histogram.lock().unwrap();
        if let Some(ref histogram) = *histogram_opt {
            histogram.observe(value);
        }
        Ok(())
    }

    #[getter]
    fn name(&self) -> String {
        self.base.name.clone()
    }
}

impl Histogram {
    pub(crate) fn new_internal(name: String) -> Self {
        Self {
            base: MetricBase::new(name, String::new(), false, false),
            histogram: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn set_histogram(&self, histogram: prometheus::Histogram) {
        let mut histogram_opt = self.histogram.lock().unwrap();
        *histogram_opt = Some(histogram);
    }
}

/// RuntimeMetrics provides factory methods for creating typed Prometheus metrics
/// and utilities for registering metrics callbacks.
/// Exposed as endpoint.metrics in Python.
#[pyclass]
#[derive(Clone)]
pub struct RuntimeMetrics {
    endpoint: dynamo_runtime::component::Endpoint,
}

impl RuntimeMetrics {
    pub fn new(endpoint: dynamo_runtime::component::Endpoint) -> Self {
        Self { endpoint }
    }

    /// Generic helper to register metrics callbacks for any type implementing MetricsRegistry
    /// This allows Endpoint, Component, and Namespace to share the same callback registration logic
    pub fn register_callback_for<T>(
        registry_item: &T,
        callback: PyObject,
    ) -> PyResult<()>
    where
        T: rs::metrics::MetricsRegistry + rs::traits::DistributedRuntimeProvider,
    {
        let hierarchy = registry_item.hierarchy();

        // Store the callback in the DRT's metrics callback registry using the registry_item's hierarchy
        registry_item.drt().register_metrics_callback(
            vec![hierarchy.clone()],
            Arc::new(move || {
                // Execute the Python callback in the Python event loop
                Python::with_gil(|py| {
                    if let Err(e) = callback.call0(py) {
                        tracing::error!("Metrics callback failed: {}", e);
                    }
                });
                Ok(())
            }),
        );

        Ok(())
    }
}

#[pymethods]
impl RuntimeMetrics {
    /// Register a Python callback to be invoked before metrics are scraped
    /// This callback will be called for this endpoint's metrics hierarchy
    fn register_update_callback(&self, callback: PyObject, _py: Python) -> PyResult<()> {
        Self::register_callback_for(&self.endpoint, callback)
    }

    // NOTE: The order of create_* methods below matches lib/runtime/src/metrics.rs::MetricsRegistry trait
    // Keep them synchronized when adding new metric types

    /// Create a Counter metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_counter(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<Counter>> {
        use dynamo_runtime::metrics::MetricsRegistry;

        let labels_vec: Vec<(&str, &str)> = labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();

        let counter = self
            .endpoint
            .create_counter(&name, &description, &labels_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = Counter::new_internal(name.clone());
        metric.set_counter(counter);
        Py::new(py, metric)
    }

    /// Create a CounterVec metric
    fn create_countervec(
        &self,
        name: String,
        description: String,
        label_names: Vec<String>,
        py: Python,
    ) -> PyResult<Py<CounterVec>> {
        use dynamo_runtime::metrics::MetricsRegistry;

        let label_names_str: Vec<&str> = label_names.iter().map(|s| s.as_str()).collect();
        let counter_vec = self
            .endpoint
            .create_countervec(&name, &description, &label_names_str, &[])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = CounterVec::new_internal(name.clone(), label_names);
        metric.set_counter(counter_vec);
        Py::new(py, metric)
    }

    /// Create a Gauge metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_gauge(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<Gauge>> {
        use dynamo_runtime::metrics::MetricsRegistry;

        let labels_vec: Vec<(&str, &str)> = labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();

        let gauge = self
            .endpoint
            .create_gauge(&name, &description, &labels_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = Gauge::new_internal(name.clone());
        metric.set_gauge(gauge);
        Py::new(py, metric)
    }

    /// Create a Histogram metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_histogram(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<Histogram>> {
        use dynamo_runtime::metrics::MetricsRegistry;

        let labels_vec: Vec<(&str, &str)> = labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();

        let histogram = self
            .endpoint
            .create_histogram(&name, &description, &labels_vec, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = Histogram::new_internal(name.clone());
        metric.set_histogram(histogram);
        Py::new(py, metric)
    }

    /// Create an IntCounter metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_intcounter(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<IntCounter>> {
        use dynamo_runtime::metrics::MetricsRegistry;

        let labels_vec: Vec<(&str, &str)> = labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();

        let counter = self
            .endpoint
            .create_intcounter(&name, &description, &labels_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = IntCounter::new_internal(name.clone());
        metric.set_counter(counter);
        Py::new(py, metric)
    }

    /// Create an IntCounterVec metric
    fn create_intcountervec(
        &self,
        name: String,
        description: String,
        label_names: Vec<String>,
        py: Python,
    ) -> PyResult<Py<IntCounterVec>> {
        use dynamo_runtime::metrics::MetricsRegistry;

        let label_names_str: Vec<&str> = label_names.iter().map(|s| s.as_str()).collect();
        let counter_vec = self
            .endpoint
            .create_intcountervec(&name, &description, &label_names_str, &[])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = IntCounterVec::new_internal(name.clone(), label_names);
        metric.set_counter(counter_vec);
        Py::new(py, metric)
    }

    /// Create an IntGauge metric
    #[pyo3(signature = (name, description, labels=None))]
    fn create_intgauge(
        &self,
        name: String,
        description: String,
        labels: Option<Vec<(String, String)>>,
        py: Python,
    ) -> PyResult<Py<IntGauge>> {
        use dynamo_runtime::metrics::MetricsRegistry;

        let labels_vec: Vec<(&str, &str)> = labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();

        let gauge = self
            .endpoint
            .create_intgauge(&name, &description, &labels_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = IntGauge::new_internal(name.clone());
        metric.set_gauge(gauge);
        Py::new(py, metric)
    }

    /// Create a GaugeVec metric
    fn create_gaugevec(
        &self,
        name: String,
        description: String,
        label_names: Vec<String>,
        py: Python,
    ) -> PyResult<Py<GaugeVec>> {
        use dynamo_runtime::metrics::MetricsRegistry;

        let label_names_str: Vec<&str> = label_names.iter().map(|s| s.as_str()).collect();
        let gauge_vec = self
            .endpoint
            .create_gaugevec(&name, &description, &label_names_str, &[])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = GaugeVec::new_internal(name.clone(), label_names);
        metric.set_gauge(gauge_vec);
        Py::new(py, metric)
    }

    /// Create an IntGaugeVec metric
    fn create_intgaugevec(
        &self,
        name: String,
        description: String,
        label_names: Vec<String>,
        py: Python,
    ) -> PyResult<Py<IntGaugeVec>> {
        use dynamo_runtime::metrics::MetricsRegistry;

        let label_names_str: Vec<&str> = label_names.iter().map(|s| s.as_str()).collect();
        let gauge_vec = self
            .endpoint
            .create_intgaugevec(&name, &description, &label_names_str, &[])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let metric = IntGaugeVec::new_internal(name.clone(), label_names);
        metric.set_gauge(gauge_vec);
        Py::new(py, metric)
    }
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add specific metric type classes
    m.add_class::<Counter>()?;
    m.add_class::<IntCounter>()?;
    m.add_class::<CounterVec>()?;
    m.add_class::<IntCounterVec>()?;
    m.add_class::<Gauge>()?;
    m.add_class::<IntGauge>()?;
    m.add_class::<GaugeVec>()?;
    m.add_class::<IntGaugeVec>()?;
    m.add_class::<Histogram>()?;

    Ok(())
}
