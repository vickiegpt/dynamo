// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the CXL Expert Manager and CXL Checkpoint Manager.
//!
//! This module exposes the Rust CXL Expert Manager and CXL Checkpoint Manager to Python,
//! providing high-performance expert weight management and checkpoint/restore for MoE models
//! with CXL memory tiering.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use dynamo_llm::kv_router::cxl_checkpoint::CxlCheckpointManager as RustCxlCheckpointManager;
use dynamo_llm::kv_router::cxl_expert_manager::{
    CxlExpertManager as RustCxlExpertManager, CxlExpertManagerConfig, ExpertLocation,
};
use dynamo_llm::kv_router::cxl_multi_host_expert_manager::{
    ExpertPartitionStrategy,
    MultiHostExpertManager as RustMultiHostExpertManager,
    MultiHostExpertManagerConfig as RustMultiHostExpertManagerConfig,
};
use dynamo_llm::kv_router::cxl_p2p::{CxlP2pConfig, CxlP2pContext, ExpertMemoryPool};
use dynamo_llm::kv_router::protocols::{
    CxlFabricTopology, CxlHostInfo, CxlSharedPoolInfo, CxlTransferPath,
    ExpertPlacementEvent, ExpertPlacementEventType, ExternalSequenceBlockHash, LocalBlockHash,
};

/// Python wrapper for CXL Expert Manager configuration
#[pyclass(name = "CxlExpertManagerConfig")]
#[derive(Clone)]
pub struct PyCxlExpertManagerConfig {
    #[pyo3(get, set)]
    pub num_experts: u32,
    #[pyo3(get, set)]
    pub num_moe_layers: u32,
    #[pyo3(get, set)]
    pub expert_weight_size: u64,
    #[pyo3(get, set)]
    pub gpu_expert_capacity: u64,
    #[pyo3(get, set)]
    pub cxl_expert_capacity: u64,
    #[pyo3(get, set)]
    pub window_size: u32,
    #[pyo3(get, set)]
    pub cxl_bandwidth_gbps: f64,
    #[pyo3(get, set)]
    pub max_gpu_experts: usize,
    #[pyo3(get, set)]
    pub enable_p2p_dma: bool,
    #[pyo3(get, set)]
    pub qos_priority_levels: u32,
}

#[pymethods]
impl PyCxlExpertManagerConfig {
    #[new]
    #[pyo3(signature = (
        num_experts = 128,
        num_moe_layers = 32,
        expert_weight_size = 268435456,
        gpu_expert_capacity = 85899345920,
        cxl_expert_capacity = 549755813888,
        window_size = 16,
        cxl_bandwidth_gbps = 128.0,
        max_gpu_experts = 32,
        enable_p2p_dma = true,
        qos_priority_levels = 4
    ))]
    pub fn new(
        num_experts: u32,
        num_moe_layers: u32,
        expert_weight_size: u64,
        gpu_expert_capacity: u64,
        cxl_expert_capacity: u64,
        window_size: u32,
        cxl_bandwidth_gbps: f64,
        max_gpu_experts: usize,
        enable_p2p_dma: bool,
        qos_priority_levels: u32,
    ) -> Self {
        Self {
            num_experts,
            num_moe_layers,
            expert_weight_size,
            gpu_expert_capacity,
            cxl_expert_capacity,
            window_size,
            cxl_bandwidth_gbps,
            max_gpu_experts,
            enable_p2p_dma,
            qos_priority_levels,
        }
    }

    #[staticmethod]
    pub fn default() -> Self {
        Self {
            num_experts: 128,
            num_moe_layers: 32,
            expert_weight_size: 256 * 1024 * 1024,
            gpu_expert_capacity: 80 * 1024 * 1024 * 1024,
            cxl_expert_capacity: 512 * 1024 * 1024 * 1024,
            window_size: 16,
            cxl_bandwidth_gbps: 128.0,
            max_gpu_experts: 32,
            enable_p2p_dma: true,
            qos_priority_levels: 4,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CxlExpertManagerConfig(num_experts={}, num_moe_layers={}, max_gpu_experts={}, window_size={})",
            self.num_experts, self.num_moe_layers, self.max_gpu_experts, self.window_size
        )
    }
}

impl From<&PyCxlExpertManagerConfig> for CxlExpertManagerConfig {
    fn from(config: &PyCxlExpertManagerConfig) -> Self {
        CxlExpertManagerConfig {
            num_experts: config.num_experts,
            num_moe_layers: config.num_moe_layers,
            expert_weight_size: config.expert_weight_size,
            gpu_expert_capacity: config.gpu_expert_capacity,
            cxl_expert_capacity: config.cxl_expert_capacity,
            window_size: config.window_size,
            cxl_bandwidth_gbps: config.cxl_bandwidth_gbps,
            max_gpu_experts: config.max_gpu_experts,
            enable_p2p_dma: config.enable_p2p_dma,
            qos_priority_levels: config.qos_priority_levels,
        }
    }
}

/// Python wrapper for CXL Expert Manager
#[pyclass(name = "CxlExpertManager")]
pub struct PyCxlExpertManager {
    inner: Arc<RustCxlExpertManager>,
    cancel_token: CancellationToken,
}

#[pymethods]
impl PyCxlExpertManager {
    /// Create a new CXL Expert Manager
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<PyCxlExpertManagerConfig>) -> PyResult<Self> {
        let cancel_token = CancellationToken::new();
        let rust_config = config
            .as_ref()
            .map(|c| c.into())
            .unwrap_or_default();

        let manager = RustCxlExpertManager::new(rust_config, cancel_token.clone());

        Ok(Self {
            inner: Arc::new(manager),
            cancel_token,
        })
    }

    /// Register an expert in the system
    #[pyo3(signature = (expert_id, layer_id, weight_size_bytes, initial_location = "gpu"))]
    pub fn register_expert(
        &self,
        expert_id: u32,
        layer_id: u32,
        weight_size_bytes: u64,
        initial_location: &str,
    ) -> PyResult<()> {
        let location = match initial_location {
            "gpu" => ExpertLocation::GpuHbm,
            "cxl" => ExpertLocation::CxlMemory,
            "evicted" => ExpertLocation::Evicted,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid location: {}. Expected 'gpu', 'cxl', or 'evicted'",
                    initial_location
                )))
            }
        };

        self.inner
            .register_expert(expert_id, layer_id, weight_size_bytes, location);
        Ok(())
    }

    /// Request an expert for inference (async)
    pub fn request_expert<'py>(
        &self,
        py: Python<'py>,
        expert_id: u32,
        layer_id: u32,
        priority: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            manager
                .request_expert(expert_id, layer_id, priority)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })
    }

    /// Record a KV delta for windowed checkpointing
    pub fn record_kv_delta(
        &self,
        token_offset: u32,
        layer_id: u32,
        expert_id: u32,
        topk_experts: Vec<u32>,
        gating_scores: Vec<f32>,
        kv_block_hash: u64,
        tokens_hash: u64,
    ) -> PyResult<()> {
        self.inner.record_kv_delta(
            token_offset,
            layer_id,
            expert_id,
            topk_experts,
            gating_scores,
            ExternalSequenceBlockHash(kv_block_hash),
            LocalBlockHash(tokens_hash),
        );
        Ok(())
    }

    /// Force checkpoint commit
    pub fn force_checkpoint(&self) -> PyResult<()> {
        self.inner.force_checkpoint();
        Ok(())
    }

    /// Perform fast recovery (async)
    pub fn fast_recovery<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let recovery_data = self.inner.get_recovery_data();
        let manager = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            manager
                .fast_recovery(recovery_data)
                .await
                .map(|d| d.as_millis() as f64)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })
    }

    /// Get current metrics as a dictionary
    pub fn get_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self.inner.get_metrics();
        let dict = PyDict::new(py);

        dict.set_item(
            "total_prefetches",
            metrics
                .total_prefetches
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "total_evictions",
            metrics
                .total_evictions
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "cache_hits",
            metrics
                .cache_hits
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "cache_misses",
            metrics
                .cache_misses
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "total_bytes_transferred",
            metrics
                .total_bytes_transferred
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "avg_transfer_latency_us",
            metrics
                .avg_transfer_latency_us
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "recovery_count",
            metrics
                .recovery_count
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "avg_recovery_time_ms",
            metrics
                .avg_recovery_time_ms
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;

        // Calculate hit rate
        let hits = metrics
            .cache_hits
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let misses = metrics
            .cache_misses
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let hit_rate = if hits + misses > 0.0 {
            hits / (hits + misses)
        } else {
            0.0
        };
        dict.set_item("cache_hit_rate", hit_rate)?;

        Ok(dict)
    }

    /// Shutdown the expert manager
    pub fn shutdown(&self) -> PyResult<()> {
        self.cancel_token.cancel();
        Ok(())
    }

    fn __repr__(&self) -> String {
        let metrics = self.inner.get_metrics();
        let hits = metrics
            .cache_hits
            .load(std::sync::atomic::Ordering::Relaxed);
        let misses = metrics
            .cache_misses
            .load(std::sync::atomic::Ordering::Relaxed);
        format!(
            "CxlExpertManager(cache_hits={}, cache_misses={}, hit_rate={:.2}%)",
            hits,
            misses,
            if hits + misses > 0 {
                (hits as f64 / (hits + misses) as f64) * 100.0
            } else {
                0.0
            }
        )
    }
}

/// Python wrapper for CXL Checkpoint Manager
///
/// This provides checkpoint/restore functionality using CXL memory with P2P DMA.
/// It records token-to-expert mappings and stores them in CXL memory for fast recovery.
#[pyclass(name = "CxlCheckpointManager")]
pub struct PyCxlCheckpointManager {
    inner: Arc<RustCxlCheckpointManager>,
}

#[pymethods]
impl PyCxlCheckpointManager {
    /// Create a new CXL Checkpoint Manager
    ///
    /// Args:
    ///     window_size: Number of tokens per checkpoint window (default: 16)
    ///     num_layers: Number of MoE layers in the model (default: 32)
    ///     buffer_size_mb: Size of checkpoint ring buffer in MB (default: 256)
    #[new]
    #[pyo3(signature = (window_size = 16, num_layers = 32, buffer_size_mb = 256))]
    pub fn new(window_size: u32, num_layers: u32, buffer_size_mb: usize) -> PyResult<Self> {
        let ctx = Arc::new(
            CxlP2pContext::new()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        let buffer_size = buffer_size_mb * 1024 * 1024;
        let manager = RustCxlCheckpointManager::new(ctx, window_size, num_layers, Some(buffer_size))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(manager),
        })
    }

    /// Set the current sequence ID for checkpointing
    pub fn set_sequence_id(&self, sequence_id: u64) {
        self.inner.set_sequence_id(sequence_id);
    }

    /// Record a token-to-expert mapping
    ///
    /// This records the routing decision for a single token at a specific layer.
    /// When window_size tokens are recorded, a checkpoint is automatically prepared.
    ///
    /// Args:
    ///     token_position: Position of the token in the sequence
    ///     layer_id: MoE layer ID
    ///     expert_id: Selected expert ID
    ///     topk_experts: List of top-k expert IDs
    ///     gating_scores: Gating scores for top-k experts
    ///     kv_block_hash: Hash of the KV block (reference, not data)
    pub fn record_mapping(
        &self,
        token_position: u32,
        layer_id: u32,
        expert_id: u32,
        topk_experts: Vec<u32>,
        gating_scores: Vec<f32>,
        kv_block_hash: u64,
    ) {
        self.inner.record_mapping(
            token_position,
            layer_id,
            expert_id,
            topk_experts,
            gating_scores,
            kv_block_hash,
        );
    }

    /// Write a checkpoint to CXL memory
    ///
    /// This serializes the current window's mappings and writes them to CXL memory.
    /// Uses P2P DMA if gpu_ptr is provided, otherwise uses CPU path.
    ///
    /// Args:
    ///     expert_locations: Dict mapping (layer_id, expert_id) to location (0=GPU, 1=CXL)
    ///     hot_set: List of (layer_id, expert_id) tuples for hot experts
    ///     gpu_ptr: Optional GPU pointer for P2P DMA (None for CPU path)
    ///
    /// Returns:
    ///     Checkpoint ID
    #[pyo3(signature = (expert_locations = None, hot_set = None, gpu_ptr = None))]
    pub fn write_checkpoint(
        &self,
        expert_locations: Option<HashMap<(u32, u32), u8>>,
        hot_set: Option<Vec<(u32, u32)>>,
        gpu_ptr: Option<u64>,
    ) -> PyResult<u64> {
        let expert_locs = expert_locations.unwrap_or_default();
        let hot = hot_set.unwrap_or_default();

        self.inner
            .write_checkpoint(expert_locs, hot, gpu_ptr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Force commit any pending mappings to a checkpoint
    ///
    /// Returns:
    ///     Optional checkpoint ID if there were pending mappings
    #[pyo3(signature = (expert_locations = None, hot_set = None))]
    pub fn force_commit(
        &self,
        expert_locations: Option<HashMap<(u32, u32), u8>>,
        hot_set: Option<Vec<(u32, u32)>>,
    ) -> PyResult<Option<u64>> {
        let expert_locs = expert_locations.unwrap_or_default();
        let hot = hot_set.unwrap_or_default();

        self.inner
            .force_commit(expert_locs, hot)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Perform fast recovery from the latest checkpoint
    ///
    /// This reads the checkpoint from CXL memory and returns the routing decisions
    /// that need to be replayed. Uses P2P DMA if gpu_ptr is provided.
    ///
    /// Args:
    ///     gpu_ptr: Optional GPU pointer for P2P DMA (None for CPU path)
    ///
    /// Returns:
    ///     Dict with recovery information
    #[pyo3(signature = (gpu_ptr=None))]
    pub fn fast_recovery<'py>(
        &self,
        py: Python<'py>,
        gpu_ptr: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .inner
            .fast_recovery(gpu_ptr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("checkpoint_id", result.checkpoint_id)?;
        dict.set_item("window_start", result.window_start)?;
        dict.set_item("window_len", result.window_len)?;
        dict.set_item("recovery_time_us", result.recovery_time.as_micros() as u64)?;

        // Convert replay instructions to list of dicts
        let replay_list = PyList::empty(py);
        for instr in &result.replay_instructions {
            let instr_dict = PyDict::new(py);
            instr_dict.set_item("sequence_id", instr.sequence_id)?;
            instr_dict.set_item("token_position", instr.token_position)?;
            instr_dict.set_item("layer_id", instr.layer_id)?;
            instr_dict.set_item("expert_id", instr.expert_id)?;
            instr_dict.set_item("topk_experts", instr.topk_experts.clone())?;
            instr_dict.set_item("gating_scores", instr.gating_scores.clone())?;
            replay_list.append(instr_dict)?;
        }
        dict.set_item("replay_instructions", replay_list)?;

        // Convert expert locations
        let expert_locs = PyDict::new(py);
        for ((layer_id, expert_id), loc) in &result.expert_locations {
            let key = format!("{}_{}", layer_id, expert_id);
            expert_locs.set_item(key, *loc)?;
        }
        dict.set_item("expert_locations", expert_locs)?;

        // Convert hot set
        let hot_list = PyList::empty(py);
        for (layer_id, expert_id) in &result.hot_set {
            let tuple = (*layer_id, *expert_id);
            hot_list.append(tuple)?;
        }
        dict.set_item("hot_set", hot_list)?;

        Ok(dict)
    }

    /// Get checkpoint manager metrics
    pub fn get_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self.inner.get_metrics();
        let dict = PyDict::new(py);

        dict.set_item(
            "checkpoints_written",
            metrics
                .checkpoints_written
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "checkpoints_read",
            metrics
                .checkpoints_read
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "bytes_written",
            metrics
                .bytes_written
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "bytes_read",
            metrics
                .bytes_read
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item("avg_write_latency_us", metrics.avg_write_latency_us())?;
        dict.set_item("avg_read_latency_us", metrics.avg_read_latency_us())?;
        dict.set_item("write_bandwidth_gbps", metrics.write_bandwidth_gbps())?;
        dict.set_item(
            "recovery_count",
            metrics
                .recovery_count
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "last_recovery_time_us",
            metrics
                .last_recovery_time_us
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;

        Ok(dict)
    }

    /// Get number of valid checkpoints stored
    pub fn checkpoint_count(&self) -> usize {
        self.inner.checkpoint_count()
    }

    /// Get total size of checkpoint data in bytes
    pub fn total_checkpoint_size(&self) -> usize {
        self.inner.total_checkpoint_size()
    }

    fn __repr__(&self) -> String {
        let metrics = self.inner.get_metrics();
        let written = metrics
            .checkpoints_written
            .load(std::sync::atomic::Ordering::Relaxed);
        let read = metrics
            .checkpoints_read
            .load(std::sync::atomic::Ordering::Relaxed);
        format!(
            "CxlCheckpointManager(checkpoints_written={}, checkpoints_read={}, total_size={})",
            written,
            read,
            self.inner.total_checkpoint_size()
        )
    }
}

// ---------------------------------------------------------------------------
// Multi-Host Expert Manager PyO3 Bindings
// ---------------------------------------------------------------------------

/// Python wrapper for ExpertPartitionStrategy enum
#[pyclass(name = "ExpertPartitionStrategy", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyExpertPartitionStrategy {
    RoundRobin = 0,
    LayerPartition = 1,
    AffinityCluster = 2,
    Dynamic = 3,
}

impl From<PyExpertPartitionStrategy> for ExpertPartitionStrategy {
    fn from(val: PyExpertPartitionStrategy) -> Self {
        match val {
            PyExpertPartitionStrategy::RoundRobin => ExpertPartitionStrategy::RoundRobin,
            PyExpertPartitionStrategy::LayerPartition => ExpertPartitionStrategy::LayerPartition,
            PyExpertPartitionStrategy::AffinityCluster => ExpertPartitionStrategy::AffinityCluster,
            PyExpertPartitionStrategy::Dynamic => ExpertPartitionStrategy::Dynamic,
        }
    }
}

impl From<ExpertPartitionStrategy> for PyExpertPartitionStrategy {
    fn from(val: ExpertPartitionStrategy) -> Self {
        match val {
            ExpertPartitionStrategy::RoundRobin => PyExpertPartitionStrategy::RoundRobin,
            ExpertPartitionStrategy::LayerPartition => PyExpertPartitionStrategy::LayerPartition,
            ExpertPartitionStrategy::AffinityCluster => PyExpertPartitionStrategy::AffinityCluster,
            ExpertPartitionStrategy::Dynamic => PyExpertPartitionStrategy::Dynamic,
        }
    }
}

/// Python wrapper for MultiHostExpertManagerConfig
#[pyclass(name = "MultiHostExpertManagerConfig")]
#[derive(Clone)]
pub struct PyMultiHostExpertManagerConfig {
    #[pyo3(get, set)]
    pub host_id: u32,
    #[pyo3(get, set)]
    pub num_layers: u32,
    #[pyo3(get, set)]
    pub num_experts: u32,
    #[pyo3(get, set)]
    pub partition_strategy: PyExpertPartitionStrategy,
    #[pyo3(get, set)]
    pub enable_replication: bool,
    #[pyo3(get, set)]
    pub max_replicas: u32,
    #[pyo3(get, set)]
    pub replication_threshold: u64,
    #[pyo3(get, set)]
    pub rebalance_interval: u64,

    // Topology fields stored for building the Rust config
    pub(crate) hosts: Vec<CxlHostInfo>,
    pub(crate) shared_pools: Vec<CxlSharedPoolInfo>,
    pub(crate) latency_matrix_ns: Vec<Vec<u64>>,
    pub(crate) bandwidth_matrix_gbps: Vec<Vec<f64>>,
}

#[pymethods]
impl PyMultiHostExpertManagerConfig {
    #[new]
    #[pyo3(signature = (
        host_id = 0,
        num_hosts = 2,
        num_layers = 32,
        num_experts = 128,
        partition_strategy = PyExpertPartitionStrategy::RoundRobin,
        enable_replication = false,
        max_replicas = 2,
        replication_threshold = 100,
        rebalance_interval = 1000,
        gpu_count_per_host = 8,
        local_cxl_capacity_gb = 512,
        inter_host_latency_ns = 200,
        inter_host_bandwidth_gbps = 64.0,
        shared_pool_capacity_gb = 1024,
        shared_pool_bandwidth_gbps = 128.0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host_id: u32,
        num_hosts: u32,
        num_layers: u32,
        num_experts: u32,
        partition_strategy: PyExpertPartitionStrategy,
        enable_replication: bool,
        max_replicas: u32,
        replication_threshold: u64,
        rebalance_interval: u64,
        gpu_count_per_host: u32,
        local_cxl_capacity_gb: u64,
        inter_host_latency_ns: u64,
        inter_host_bandwidth_gbps: f64,
        shared_pool_capacity_gb: u64,
        shared_pool_bandwidth_gbps: f64,
    ) -> Self {
        let hosts: Vec<CxlHostInfo> = (0..num_hosts)
            .map(|i| CxlHostInfo {
                host_id: i,
                hostname: format!("host{}", i),
                gpu_count: gpu_count_per_host,
                local_cxl_capacity: local_cxl_capacity_gb * 1024 * 1024 * 1024,
                worker_ids: vec![i as i64],
            })
            .collect();

        let latency_matrix_ns: Vec<Vec<u64>> = (0..num_hosts)
            .map(|i| {
                (0..num_hosts)
                    .map(|j| {
                        if i == j {
                            0
                        } else {
                            inter_host_latency_ns
                        }
                    })
                    .collect()
            })
            .collect();

        let bandwidth_matrix_gbps: Vec<Vec<f64>> = (0..num_hosts)
            .map(|i| {
                (0..num_hosts)
                    .map(|j| {
                        if i == j {
                            0.0
                        } else {
                            inter_host_bandwidth_gbps
                        }
                    })
                    .collect()
            })
            .collect();

        let shared_pools = vec![CxlSharedPoolInfo {
            pool_id: 1,
            capacity: shared_pool_capacity_gb * 1024 * 1024 * 1024,
            connected_hosts: (0..num_hosts).collect(),
            bandwidth_gbps: shared_pool_bandwidth_gbps,
        }];

        Self {
            host_id,
            num_layers,
            num_experts,
            partition_strategy,
            enable_replication,
            max_replicas,
            replication_threshold,
            rebalance_interval,
            hosts,
            shared_pools,
            latency_matrix_ns,
            bandwidth_matrix_gbps,
        }
    }

    /// Set custom topology from Python dicts
    #[pyo3(signature = (hosts, latency_matrix_ns, bandwidth_matrix_gbps, shared_pools = None))]
    pub fn set_topology(
        &mut self,
        hosts: Vec<(u32, String, u32, u64, Vec<i64>)>,
        latency_matrix_ns: Vec<Vec<u64>>,
        bandwidth_matrix_gbps: Vec<Vec<f64>>,
        shared_pools: Option<Vec<(u64, u64, Vec<u32>, f64)>>,
    ) {
        self.hosts = hosts
            .into_iter()
            .map(|(host_id, hostname, gpu_count, cxl_cap, worker_ids)| CxlHostInfo {
                host_id,
                hostname,
                gpu_count,
                local_cxl_capacity: cxl_cap,
                worker_ids,
            })
            .collect();

        self.latency_matrix_ns = latency_matrix_ns;
        self.bandwidth_matrix_gbps = bandwidth_matrix_gbps;

        if let Some(pools) = shared_pools {
            self.shared_pools = pools
                .into_iter()
                .map(|(pool_id, capacity, connected, bw)| CxlSharedPoolInfo {
                    pool_id,
                    capacity,
                    connected_hosts: connected,
                    bandwidth_gbps: bw,
                })
                .collect();
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiHostExpertManagerConfig(host_id={}, num_hosts={}, num_layers={}, num_experts={}, strategy={:?})",
            self.host_id,
            self.hosts.len(),
            self.num_layers,
            self.num_experts,
            self.partition_strategy,
        )
    }
}

/// Python wrapper for MultiHostExpertManager
///
/// Manages expert parallelism across multiple hosts connected via CXL 3.0 fabric.
/// Each host has GPU HBM (hot tier) + local CXL (cold tier) + access to shared GFAM.
#[pyclass(name = "MultiHostExpertManager")]
pub struct PyMultiHostExpertManager {
    inner: Arc<RustMultiHostExpertManager>,
}

#[pymethods]
impl PyMultiHostExpertManager {
    /// Create a new MultiHostExpertManager
    #[new]
    #[pyo3(signature = (config))]
    pub fn new(config: PyMultiHostExpertManagerConfig) -> PyResult<Self> {
        let topology = CxlFabricTopology {
            hosts: config.hosts.clone(),
            shared_pools: config.shared_pools.clone(),
            latency_matrix_ns: config.latency_matrix_ns.clone(),
            bandwidth_matrix_gbps: config.bandwidth_matrix_gbps.clone(),
        };

        let rust_config = RustMultiHostExpertManagerConfig {
            host_id: config.host_id,
            topology,
            partition_strategy: config.partition_strategy.into(),
            num_layers: config.num_layers,
            num_experts: config.num_experts,
            enable_replication: config.enable_replication,
            max_replicas: config.max_replicas,
            replication_threshold: config.replication_threshold,
            rebalance_interval: config.rebalance_interval,
        };

        let p2p_config = CxlP2pConfig {
            default_buffer_size: 1024 * 1024, // 1MB per buffer for multi-host
            use_huge_pages: false,
            host_id: config.host_id,
            enable_fabric: true,
            ..Default::default()
        };

        let ctx = Arc::new(
            CxlP2pContext::with_config(p2p_config)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        let pool = Arc::new(
            ExpertMemoryPool::new(ctx.clone(), 1)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        let manager = RustMultiHostExpertManager::new(rust_config, ctx, pool);

        Ok(Self {
            inner: Arc::new(manager),
        })
    }

    /// Initialize partitioning and populate the global placement table
    pub fn initialize(&self) {
        self.inner.initialize();
    }

    /// Request an expert, following the multi-host resolution flow
    ///
    /// Returns a dict with:
    ///   - transfer_path: str ("local_p2p", "shared_pool", or "cross_host_fabric")
    ///   - transfer_id: int
    ///   - latency_ns: int
    ///   - bandwidth_gbps: float
    ///   - bytes_transferred: int
    pub fn request_expert<'py>(
        &self,
        py: Python<'py>,
        layer_id: u32,
        expert_id: u32,
        gpu_ptr: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .inner
            .request_expert(layer_id, expert_id, gpu_ptr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let (path, transfer) = result;
        let dict = PyDict::new(py);

        let path_str = match path {
            CxlTransferPath::LocalP2p => "local_p2p",
            CxlTransferPath::SharedPool => "shared_pool",
            CxlTransferPath::CrossHostFabric => "cross_host_fabric",
        };
        dict.set_item("transfer_path", path_str)?;
        dict.set_item("transfer_id", transfer.transfer_id)?;
        dict.set_item("latency_ns", transfer.latency_ns)?;
        dict.set_item("bandwidth_gbps", transfer.bandwidth_gbps)?;
        dict.set_item("bytes_transferred", transfer.bytes_transferred)?;

        Ok(dict)
    }

    /// Proactive cross-host prefetch based on expert predictions
    ///
    /// Args:
    ///     predictions: List of (layer_id, expert_id) tuples to prefetch
    ///
    /// Returns:
    ///     List of (layer_id, expert_id, transfer_path_str) for experts that need prefetching
    pub fn cross_host_prefetch(
        &self,
        predictions: Vec<(u32, u32)>,
    ) -> Vec<(u32, u32, String)> {
        let actions = self.inner.cross_host_prefetch(&predictions);
        actions
            .into_iter()
            .map(|(l, e, path)| {
                let path_str = match path {
                    CxlTransferPath::LocalP2p => "local_p2p".to_string(),
                    CxlTransferPath::SharedPool => "shared_pool".to_string(),
                    CxlTransferPath::CrossHostFabric => "cross_host_fabric".to_string(),
                };
                (l, e, path_str)
            })
            .collect()
    }

    /// Check if an expert should be replicated based on access patterns
    ///
    /// Returns a dict with event info if replication happened, None otherwise
    pub fn maybe_replicate_expert<'py>(
        &self,
        py: Python<'py>,
        layer_id: u32,
        expert_id: u32,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.maybe_replicate_expert(layer_id, expert_id) {
            Some(event) => {
                let dict = placement_event_to_dict(py, &event)?;
                Ok(Some(dict))
            }
            None => Ok(None),
        }
    }

    /// Handle a host failure: reassign orphaned experts to surviving hosts
    ///
    /// Returns list of placement event dicts
    pub fn handle_host_failure<'py>(
        &self,
        py: Python<'py>,
        failed_host: u32,
    ) -> PyResult<Bound<'py, PyList>> {
        let events = self.inner.handle_host_failure(failed_host);
        let list = PyList::empty(py);
        for event in &events {
            let dict = placement_event_to_dict(py, event)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Handle a new host joining the fabric
    pub fn handle_host_join<'py>(
        &self,
        py: Python<'py>,
        host_id: u32,
        hostname: String,
        gpu_count: u32,
        local_cxl_capacity: u64,
        worker_ids: Vec<i64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let host_info = CxlHostInfo {
            host_id,
            hostname,
            gpu_count,
            local_cxl_capacity,
            worker_ids,
        };
        let events = self.inner.handle_host_join(host_info);
        let list = PyList::empty(py);
        for event in &events {
            let dict = placement_event_to_dict(py, event)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Rebalance expert placements based on access patterns
    pub fn rebalance<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyList>> {
        let events = self.inner.rebalance();
        let list = PyList::empty(py);
        for event in &events {
            let dict = placement_event_to_dict(py, event)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Apply an expert placement event (received via NATS)
    #[pyo3(signature = (event_type, version, source_host, timestamp_us, layer_id=None, expert_id=None, from_host=None, to_host=None, source_host_repl=None, replica_host=None, failed_host_id=None))]
    pub fn apply_placement_event(
        &self,
        event_type: String,
        version: u64,
        source_host: u32,
        timestamp_us: u64,
        layer_id: Option<u32>,
        expert_id: Option<u32>,
        from_host: Option<u32>,
        to_host: Option<u32>,
        source_host_repl: Option<u32>,
        replica_host: Option<u32>,
        failed_host_id: Option<u32>,
    ) -> PyResult<()> {
        let evt_type = match event_type.as_str() {
            "expert_moved" => ExpertPlacementEventType::ExpertMoved {
                layer_id: layer_id.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("layer_id required for expert_moved")
                })?,
                expert_id: expert_id.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("expert_id required for expert_moved")
                })?,
                from_host: from_host.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("from_host required for expert_moved")
                })?,
                to_host: to_host.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("to_host required for expert_moved")
                })?,
            },
            "expert_replicated" => ExpertPlacementEventType::ExpertReplicated {
                layer_id: layer_id.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "layer_id required for expert_replicated",
                    )
                })?,
                expert_id: expert_id.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "expert_id required for expert_replicated",
                    )
                })?,
                source_host: source_host_repl.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "source_host_repl required for expert_replicated",
                    )
                })?,
                replica_host: replica_host.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "replica_host required for expert_replicated",
                    )
                })?,
            },
            "host_left" => ExpertPlacementEventType::HostLeft {
                host_id: failed_host_id.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "failed_host_id required for host_left",
                    )
                })?,
                graceful: false,
            },
            "host_joined" => ExpertPlacementEventType::HostJoined {
                host_info: CxlHostInfo {
                    host_id: to_host.unwrap_or(0),
                    hostname: format!("host{}", to_host.unwrap_or(0)),
                    gpu_count: 8,
                    local_cxl_capacity: 512 * 1024 * 1024 * 1024,
                    worker_ids: vec![to_host.unwrap_or(0) as i64],
                },
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown event type: {}",
                    event_type
                )))
            }
        };

        let event = ExpertPlacementEvent {
            event_type: evt_type,
            version,
            source_host,
            timestamp_us,
        };
        self.inner.apply_placement_event(&event);
        Ok(())
    }

    /// Get placements for a specific expert
    ///
    /// Returns list of dicts with host_id, memory_tier, is_replica
    pub fn get_placements<'py>(
        &self,
        py: Python<'py>,
        layer_id: u32,
        expert_id: u32,
    ) -> PyResult<Bound<'py, PyList>> {
        let placements = self.inner.placement_table().get_placements(layer_id, expert_id);
        let list = PyList::empty(py);
        for p in &placements {
            let dict = PyDict::new(py);
            dict.set_item("host_id", p.host_id)?;
            dict.set_item("memory_tier", p.memory_tier)?;
            dict.set_item("is_replica", p.is_replica)?;
            dict.set_item(
                "accessible_hosts",
                p.accessible_hosts.clone(),
            )?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Get all experts assigned to a specific host
    pub fn get_host_experts(&self, host_id: u32) -> Vec<(u32, u32)> {
        self.inner.placement_table().get_host_experts(host_id)
    }

    /// Get total number of expert placements
    pub fn total_placements(&self) -> usize {
        self.inner.placement_table().total_placements()
    }

    /// Get placement table version
    pub fn placement_version(&self) -> u64 {
        self.inner.placement_table().version()
    }

    /// Get multi-host expert manager statistics
    pub fn get_statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.get_statistics();
        let dict = PyDict::new(py);
        for (key, value) in &stats {
            dict.set_item(key, *value)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.get_statistics();
        format!(
            "MultiHostExpertManager(total_placements={}, total_accesses={}, local_hits={}, cross_host={})",
            stats.get("total_placements").unwrap_or(&0),
            stats.get("total_accesses").unwrap_or(&0),
            stats.get("local_hits").unwrap_or(&0),
            stats.get("cross_host_hits").unwrap_or(&0),
        )
    }
}

/// Helper: convert ExpertPlacementEvent to a Python dict
fn placement_event_to_dict<'py>(
    py: Python<'py>,
    event: &ExpertPlacementEvent,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("version", event.version)?;
    dict.set_item("source_host", event.source_host)?;
    dict.set_item("timestamp_us", event.timestamp_us)?;

    match &event.event_type {
        ExpertPlacementEventType::ExpertMoved {
            layer_id,
            expert_id,
            from_host,
            to_host,
        } => {
            dict.set_item("event_type", "expert_moved")?;
            dict.set_item("layer_id", *layer_id)?;
            dict.set_item("expert_id", *expert_id)?;
            dict.set_item("from_host", *from_host)?;
            dict.set_item("to_host", *to_host)?;
        }
        ExpertPlacementEventType::ExpertReplicated {
            layer_id,
            expert_id,
            source_host,
            replica_host,
        } => {
            dict.set_item("event_type", "expert_replicated")?;
            dict.set_item("layer_id", *layer_id)?;
            dict.set_item("expert_id", *expert_id)?;
            dict.set_item("source_host_repl", *source_host)?;
            dict.set_item("replica_host", *replica_host)?;
        }
        ExpertPlacementEventType::HostLeft { host_id, graceful } => {
            dict.set_item("event_type", "host_left")?;
            dict.set_item("failed_host_id", *host_id)?;
            dict.set_item("graceful", *graceful)?;
        }
        ExpertPlacementEventType::HostJoined { host_info } => {
            dict.set_item("event_type", "host_joined")?;
            dict.set_item("host_id", host_info.host_id)?;
            dict.set_item("hostname", host_info.hostname.clone())?;
        }
        ExpertPlacementEventType::PlacementRefresh { .. } => {
            dict.set_item("event_type", "placement_refresh")?;
        }
    }

    Ok(dict)
}

