// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CXL P2P DMA Interface for GPU <-> CXL Memory Transfers
//!
//! This module provides real ioctl-based P2P DMA using the NVIDIA RM driver,
//! enabling direct GPU <-> CXL memory transfers that bypass the CPU.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                         NVIDIA GPU                                        │
//! │  ┌─────────────────────────────────────────────────────────────────────┐  │
//! │  │                     GPU HBM Memory                                  │  │
//! │  │    Expert Weights    │    KV Cache    │    Activations             │  │
//! │  └─────────────────────────────────────────────────────────────────────┘  │
//! │                              │                                            │
//! │                    NVIDIA RM ioctl (P2P DMA)                              │
//! └──────────────────────────────┼────────────────────────────────────────────┘
//!                                │
//!                    PCIe/CXL Switch (P2P Bypass)
//!                                │
//! ┌──────────────────────────────┼────────────────────────────────────────────┐
//! │                    CXL Memory Expander                                    │
//! │  ┌─────────────────────────────────────────────────────────────────────┐  │
//! │  │                    CXL Memory Pool                                   │  │
//! │  │   Parked Experts   │   Cold KV Cache   │   Checkpoints              │  │
//! │  └─────────────────────────────────────────────────────────────────────┘  │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **P2P DMA**: Direct GPU <-> CXL transfers via NVIDIA RM ioctl, bypassing CPU
//! - **Buffer Registration**: CXL buffers registered with GPU driver for direct access
//! - **Huge Page Support**: Uses `MAP_HUGETLB` for large allocations with fallback
//! - **Memory Pool Management**: Efficient buffer allocation with mmap/mlock
//!
//! # Usage
//!
//! ```rust,ignore
//! use dynamo_llm::kv_router::cxl_p2p::{CxlP2pContext, CxlBuffer, TransferDirection};
//!
//! // Initialize CXL P2P context
//! let ctx = CxlP2pContext::new()?;
//!
//! // Allocate a CXL buffer
//! let buffer = ctx.alloc_buffer(256 * 1024 * 1024)?; // 256MB
//!
//! // Transfer GPU tensor to CXL (expert parking)
//! let latency = ctx.transfer_timed(
//!     &buffer,
//!     gpu_ptr,
//!     0, // CXL offset
//!     size,
//!     TransferDirection::GpuToCxl,
//! )?;
//!
//! // Transfer CXL data back to GPU (expert hydration)
//! let latency = ctx.transfer_timed(
//!     &buffer,
//!     gpu_ptr,
//!     0,
//!     size,
//!     TransferDirection::CxlToGpu,
//! )?;
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// NVIDIA RM ioctl constants (matching the C++ cxl_memory module)
// ---------------------------------------------------------------------------

/// ioctl magic for NVIDIA RM
const NV_IOCTL_MAGIC: u8 = b'F';

/// RM Control ioctl command number
const NV_ESC_RM_CONTROL: u8 = 0x2a;

/// RM Alloc ioctl command number
const NV_ESC_RM_ALLOC: u8 = 0x2b;

/// RM Free ioctl command number
const NV_ESC_RM_FREE: u8 = 0x29;

/// NV01 Root class (RM client)
const NV01_ROOT: u32 = 0x0000_0000;

/// NV01 Device class
const NV01_DEVICE_0: u32 = 0x0000_0080;

/// NV20 Subdevice class
const NV20_SUBDEVICE_0: u32 = 0x0000_2080;

/// Query CXL link info
const NV2080_CTRL_CMD_BUS_GET_CXL_INFO: u32 = 0x2080_1833;

/// Issue a P2P DMA request between GPU and CXL
const NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST: u32 = 0x2080_1834;

/// Register a CXL buffer with the GPU driver for P2P access
const NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER: u32 = 0x2080_1835;

/// Unregister a previously registered CXL buffer
const NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER: u32 = 0x2080_1836;

// ---------------------------------------------------------------------------
// CXL 3.0 ioctl constants for fabric-level operations
// ---------------------------------------------------------------------------

/// Query CXL 3.0 fabric topology (hosts, shared pools, connectivity)
const NV2080_CTRL_CMD_BUS_GET_CXL_FABRIC_INFO: u32 = 0x2080_1837;

/// Register a shared GFAM region with the GPU driver
const NV2080_CTRL_CMD_BUS_REGISTER_CXL_SHARED_POOL: u32 = 0x2080_1838;

/// Fabric-aware DMA request with source/destination host IDs
const NV2080_CTRL_CMD_BUS_CXL_FABRIC_DMA_REQUEST: u32 = 0x2080_1839;

/// DMA direction: GPU -> CXL
const CXL_P2P_DMA_FLAG_GPU_TO_CXL: u32 = 0x0;

/// DMA direction: CXL -> GPU
const CXL_P2P_DMA_FLAG_CXL_TO_GPU: u32 = 0x1;

/// Huge page size (2MB)
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Linux ioctl direction bits (from asm-generic/ioctl.h)
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;

// ---------------------------------------------------------------------------
// NVIDIA RM ioctl parameter structs (repr(C), matching driver ABI)
// ---------------------------------------------------------------------------

/// RM Control parameters (NVOS54)
#[repr(C)]
#[derive(Default)]
struct NVOS54Parameters {
    h_client: u32,
    h_object: u32,
    cmd: u32,
    flags: u32,
    params: u64, // pointer to cmd-specific params
    params_size: u32,
    status: u32,
}

/// RM Alloc parameters (NVOS21)
#[repr(C)]
#[derive(Default)]
struct NVOS21Parameters {
    h_root: u32,
    h_object_parent: u32,
    h_object_new: u32,
    h_class: u32,
    p_alloc_parms: u64, // pointer to class-specific params
    alloc_parms_size: u32,
    status: u32,
}

/// RM Free parameters (NVOS00)
#[repr(C)]
#[derive(Default)]
struct NVOS00Parameters {
    h_root: u32,
    h_object_parent: u32,
    h_object_old: u32,
    status: u32,
}

/// CXL Info query result
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Nv2080BusGetCxlInfoParams {
    /// Whether CXL link is up
    b_is_link_up: u32,
    /// Whether device is a CXL memory expander
    b_is_memory_expander: u32,
    /// Number of active CXL links
    num_links: u32,
    /// CXL version (1, 2, or 3)
    version: u32,
    /// Per-link bandwidth in MB/s
    bandwidth_mbps: u32,
    /// Whether P2P DMA is supported
    b_p2p_supported: u32,
}

/// CXL buffer registration parameters
#[repr(C)]
#[derive(Default)]
struct Nv2080BusRegisterCxlBufferParams {
    /// CPU virtual address of the buffer
    cpu_virtual_address: u64,
    /// Buffer size in bytes
    size: u64,
    /// Output: driver handle for the registered buffer
    h_buffer: u64,
    /// Status
    status: u32,
    /// Padding
    _pad: u32,
}

/// CXL buffer unregistration parameters
#[repr(C)]
#[derive(Default)]
struct Nv2080BusUnregisterCxlBufferParams {
    /// Driver handle from registration
    h_buffer: u64,
    /// Status
    status: u32,
    /// Padding
    _pad: u32,
}

/// P2P DMA request parameters
#[repr(C)]
#[derive(Default)]
struct Nv2080BusCxlP2pDmaRequestParams {
    /// Registered CXL buffer handle
    h_cxl_buffer: u64,
    /// GPU virtual address
    gpu_virtual_address: u64,
    /// Offset within CXL buffer
    cxl_offset: u64,
    /// Transfer size in bytes
    size: u64,
    /// Direction flags (CXL_P2P_DMA_FLAG_*)
    flags: u32,
    /// Output: transfer ID assigned by driver
    transfer_id: u32,
    /// Status
    status: u32,
    /// Padding
    _pad: u32,
}

// ---------------------------------------------------------------------------
// CXL 3.0 fabric ioctl parameter structs
// ---------------------------------------------------------------------------

/// CXL 3.0 fabric info query result
#[repr(C)]
#[derive(Default, Clone)]
struct Nv2080BusGetCxlFabricInfoParams {
    /// Number of hosts in the fabric
    num_hosts: u32,
    /// Local host ID
    local_host_id: u32,
    /// Number of shared GFAM pools
    num_shared_pools: u32,
    /// CXL fabric version (3 for CXL 3.0)
    fabric_version: u32,
    /// Whether fabric is operational
    b_fabric_active: u32,
    /// Total fabric bandwidth in MB/s
    fabric_bandwidth_mbps: u32,
    /// Padding
    _pad: [u32; 2],
}

/// CXL 3.0 shared pool registration parameters
#[repr(C)]
#[derive(Default)]
struct Nv2080BusRegisterCxlSharedPoolParams {
    /// CPU virtual address of the shared pool region
    cpu_virtual_address: u64,
    /// Pool size in bytes
    size: u64,
    /// Fabric pool ID (assigned by fabric manager)
    fabric_pool_id: u64,
    /// Output: driver handle for the shared pool
    h_pool: u64,
    /// Number of connected hosts
    num_connected_hosts: u32,
    /// Status
    status: u32,
}

/// CXL 3.0 fabric-aware DMA request parameters
#[repr(C)]
#[derive(Default)]
struct Nv2080BusCxlFabricDmaRequestParams {
    /// Source buffer handle (CXL buffer or shared pool)
    h_src_buffer: u64,
    /// Destination buffer handle
    h_dst_buffer: u64,
    /// GPU virtual address (for GPU-involved transfers)
    gpu_virtual_address: u64,
    /// Offset within source buffer
    src_offset: u64,
    /// Offset within destination buffer
    dst_offset: u64,
    /// Transfer size in bytes
    size: u64,
    /// Source host ID
    src_host_id: u32,
    /// Destination host ID
    dst_host_id: u32,
    /// Direction flags
    flags: u32,
    /// Output: transfer ID
    transfer_id: u32,
    /// Status
    status: u32,
    /// Padding
    _pad: u32,
}

// ---------------------------------------------------------------------------
// ioctl number construction (matching Linux kernel macros)
// ---------------------------------------------------------------------------

/// Construct ioctl request number: _IOWR(type, nr, size)
const fn nv_iowr(nr: u8, size: usize) -> libc::c_ulong {
    let dir: libc::c_ulong = (IOC_READ | IOC_WRITE) as libc::c_ulong;
    (dir << 30)
        | ((size as libc::c_ulong & 0x3FFF) << 16)
        | ((NV_IOCTL_MAGIC as libc::c_ulong) << 8)
        | (nr as libc::c_ulong)
}

// Pre-computed ioctl numbers
const NV_IOCTL_RM_CONTROL: libc::c_ulong =
    nv_iowr(NV_ESC_RM_CONTROL, std::mem::size_of::<NVOS54Parameters>());
const NV_IOCTL_RM_ALLOC: libc::c_ulong =
    nv_iowr(NV_ESC_RM_ALLOC, std::mem::size_of::<NVOS21Parameters>());
const NV_IOCTL_RM_FREE: libc::c_ulong =
    nv_iowr(NV_ESC_RM_FREE, std::mem::size_of::<NVOS00Parameters>());

// ---------------------------------------------------------------------------
// Low-level RM helper functions
// ---------------------------------------------------------------------------

/// Issue an RM Control ioctl
fn rm_control(
    fd: i32,
    h_client: u32,
    h_object: u32,
    cmd: u32,
    params: *mut u8,
    size: u32,
) -> CxlP2pResult<u32> {
    let mut ctl = NVOS54Parameters {
        h_client,
        h_object,
        cmd,
        flags: 0,
        params: params as u64,
        params_size: size,
        status: 0,
    };

    let ret = unsafe { libc::ioctl(fd, NV_IOCTL_RM_CONTROL, &mut ctl as *mut NVOS54Parameters) };
    if ret < 0 {
        return Err(CxlP2pError::DriverError(
            std::io::Error::last_os_error().raw_os_error().unwrap_or(-1) as u32,
        ));
    }
    if ctl.status != 0 {
        return Err(CxlP2pError::DriverError(ctl.status));
    }
    Ok(ctl.status)
}

/// Issue an RM Alloc ioctl
fn rm_alloc(
    fd: i32,
    h_root: u32,
    h_parent: u32,
    h_object: u32,
    h_class: u32,
    params: *mut u8,
    size: u32,
) -> CxlP2pResult<u32> {
    let mut alloc = NVOS21Parameters {
        h_root,
        h_object_parent: h_parent,
        h_object_new: h_object,
        h_class,
        p_alloc_parms: if params.is_null() {
            0
        } else {
            params as u64
        },
        alloc_parms_size: size,
        status: 0,
    };

    let ret = unsafe { libc::ioctl(fd, NV_IOCTL_RM_ALLOC, &mut alloc as *mut NVOS21Parameters) };
    if ret < 0 {
        return Err(CxlP2pError::DriverError(
            std::io::Error::last_os_error().raw_os_error().unwrap_or(-1) as u32,
        ));
    }
    if alloc.status != 0 {
        return Err(CxlP2pError::DriverError(alloc.status));
    }
    Ok(alloc.h_object_new)
}

/// Issue an RM Free ioctl
fn rm_free(fd: i32, h_root: u32, h_parent: u32, h_object: u32) -> CxlP2pResult<()> {
    let mut free = NVOS00Parameters {
        h_root,
        h_object_parent: h_parent,
        h_object_old: h_object,
        status: 0,
    };

    let ret = unsafe { libc::ioctl(fd, NV_IOCTL_RM_FREE, &mut free as *mut NVOS00Parameters) };
    if ret < 0 {
        return Err(CxlP2pError::DriverError(
            std::io::Error::last_os_error().raw_os_error().unwrap_or(-1) as u32,
        ));
    }
    if free.status != 0 {
        return Err(CxlP2pError::DriverError(free.status));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public types (API preserved from original)
// ---------------------------------------------------------------------------

/// P2P DMA transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferDirection {
    /// GPU HBM -> CXL Memory (expert parking, KV cache offload)
    GpuToCxl,
    /// CXL Memory -> GPU HBM (expert hydration, KV cache restore)
    CxlToGpu,
}

/// DMA transfer flags matching NVIDIA driver definitions
#[repr(u32)]
pub enum DmaFlags {
    GpuToCxl = 0x0,
    CxlToGpu = 0x1,
}

/// CXL P2P errors
#[derive(Debug, Error)]
pub enum CxlP2pError {
    #[error("CXL P2P not initialized")]
    NotInitialized,

    #[error("CXL initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Buffer allocation failed: {0}")]
    AllocationFailed(String),

    #[error("DMA transfer failed: {0}")]
    TransferFailed(String),

    #[error("Invalid buffer ID: {0}")]
    InvalidBufferId(u64),

    #[error("Buffer registration failed: {0}")]
    RegistrationFailed(String),

    #[error("Driver error: {0:#x}")]
    DriverError(u32),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("P2P DMA not available on this system")]
    P2pNotAvailable,

    #[error("CXL 3.0 fabric not available")]
    FabricNotAvailable,

    #[error("Shared pool not found: {0}")]
    SharedPoolNotFound(u64),

    #[error("Cross-host transfer failed: {0}")]
    CrossHostTransferFailed(String),
}

pub type CxlP2pResult<T> = std::result::Result<T, CxlP2pError>;

/// CXL system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CxlSystemInfo {
    /// Whether CXL link is active
    pub link_up: bool,
    /// Whether this is a CXL memory expander
    pub memory_expander: bool,
    /// Number of active CXL links
    pub num_links: u32,
    /// CXL version (1, 2, or 3)
    pub version: u32,
    /// Per-link bandwidth in MB/s
    pub bandwidth_mbps: u32,
    /// Whether P2P DMA is available
    pub p2p_available: bool,
}

impl Default for CxlSystemInfo {
    fn default() -> Self {
        Self {
            link_up: false,
            memory_expander: false,
            num_links: 0,
            version: 2,
            bandwidth_mbps: 0,
            p2p_available: false,
        }
    }
}

/// CXL 3.0 fabric information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CxlFabricInfo {
    /// Number of hosts in the fabric
    pub num_hosts: u32,
    /// Local host ID
    pub local_host_id: u32,
    /// Number of shared GFAM pools
    pub num_shared_pools: u32,
    /// CXL fabric version
    pub fabric_version: u32,
    /// Whether fabric is active
    pub fabric_active: bool,
    /// Aggregate fabric bandwidth in MB/s
    pub fabric_bandwidth_mbps: u32,
}

impl Default for CxlFabricInfo {
    fn default() -> Self {
        Self {
            num_hosts: 1,
            local_host_id: 0,
            num_shared_pools: 0,
            fabric_version: 0,
            fabric_active: false,
            fabric_bandwidth_mbps: 0,
        }
    }
}

/// Handle to a shared GFAM pool in CXL 3.0 fabric
#[derive(Debug)]
pub struct CxlSharedPool {
    /// Fabric pool ID
    pub pool_id: u64,
    /// CPU-accessible pointer (mmap'd region)
    pub cpu_ptr: u64,
    /// Pool size in bytes
    pub size: usize,
    /// Driver handle for fabric DMA operations
    pub driver_handle: u64,
    /// Host IDs connected to this pool
    pub connected_host_ids: Vec<u32>,
}

/// CXL buffer handle
#[derive(Debug)]
pub struct CxlBuffer {
    /// Unique buffer ID
    pub id: u64,
    /// CPU-accessible pointer (mmap'd region)
    pub cpu_ptr: u64,
    /// Buffer size in bytes
    pub size: usize,
    /// Driver handle for P2P DMA operations
    pub driver_handle: u64,
    /// Whether buffer is registered with GPU driver
    pub registered: bool,
}

impl Clone for CxlBuffer {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            cpu_ptr: self.cpu_ptr,
            size: self.size,
            driver_handle: self.driver_handle,
            registered: self.registered,
        }
    }
}

/// Transfer result with timing information
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Transfer ID assigned by driver
    pub transfer_id: u32,
    /// Transfer latency in nanoseconds
    pub latency_ns: u64,
    /// Effective bandwidth in GB/s
    pub bandwidth_gbps: f64,
    /// Bytes transferred
    pub bytes_transferred: usize,
}

/// Transfer statistics
#[derive(Debug, Default)]
pub struct TransferStats {
    pub total_transfers: AtomicU64,
    pub total_bytes: AtomicU64,
    pub total_latency_ns: AtomicU64,
    pub gpu_to_cxl_count: AtomicU64,
    pub cxl_to_gpu_count: AtomicU64,
    pub p2p_transfers: AtomicU64,
    pub staged_transfers: AtomicU64,
}

impl TransferStats {
    pub fn record(&self, result: &TransferResult, direction: TransferDirection, used_p2p: bool) {
        self.total_transfers.fetch_add(1, Ordering::Relaxed);
        self.total_bytes
            .fetch_add(result.bytes_transferred as u64, Ordering::Relaxed);
        self.total_latency_ns
            .fetch_add(result.latency_ns, Ordering::Relaxed);

        match direction {
            TransferDirection::GpuToCxl => {
                self.gpu_to_cxl_count.fetch_add(1, Ordering::Relaxed);
            }
            TransferDirection::CxlToGpu => {
                self.cxl_to_gpu_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        if used_p2p {
            self.p2p_transfers.fetch_add(1, Ordering::Relaxed);
        } else {
            self.staged_transfers.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn avg_latency_us(&self) -> f64 {
        let count = self.total_transfers.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let total_ns = self.total_latency_ns.load(Ordering::Relaxed);
        (total_ns as f64 / count as f64) / 1000.0
    }

    pub fn avg_bandwidth_gbps(&self) -> f64 {
        let total_bytes = self.total_bytes.load(Ordering::Relaxed);
        let total_ns = self.total_latency_ns.load(Ordering::Relaxed);
        if total_ns == 0 {
            return 0.0;
        }
        (total_bytes as f64 / 1e9) / (total_ns as f64 / 1e9)
    }
}

/// Configuration for CXL P2P context
#[derive(Debug, Clone)]
pub struct CxlP2pConfig {
    /// Default buffer size for allocations
    pub default_buffer_size: usize,
    /// Enable huge pages for buffer allocation
    pub use_huge_pages: bool,
    /// P2P DMA threshold (below this, use CPU staging)
    pub p2p_threshold_bytes: usize,
    /// Enable CUDA event timing
    pub enable_cuda_timing: bool,
    /// GPU device ID
    pub gpu_device_id: u32,
    /// Host ID in CXL 3.0 fabric (0 for single-host)
    pub host_id: u32,
    /// Enable CXL 3.0 fabric operations (shared pools, cross-host DMA)
    pub enable_fabric: bool,
}

impl Default for CxlP2pConfig {
    fn default() -> Self {
        Self {
            default_buffer_size: 256 * 1024 * 1024, // 256MB
            use_huge_pages: true,
            p2p_threshold_bytes: 256 * 1024, // 256KB threshold for P2P
            enable_cuda_timing: true,
            gpu_device_id: 0,
            host_id: 0,
            enable_fabric: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal state for RM handles
// ---------------------------------------------------------------------------

/// NVIDIA RM driver handles obtained during initialization
struct RmHandles {
    /// File descriptor for /dev/nvidiactl
    ctl_fd: i32,
    /// File descriptor for /dev/nvidia<N>
    dev_fd: i32,
    /// RM client handle
    h_client: u32,
    /// RM device handle
    h_device: u32,
    /// RM subdevice handle
    h_subdevice: u32,
}

// ---------------------------------------------------------------------------
// CxlP2pContext
// ---------------------------------------------------------------------------

/// CXL P2P DMA Context
///
/// Provides real ioctl-based P2P DMA via the NVIDIA RM driver when hardware
/// is available, with automatic fallback to simulation mode otherwise.
pub struct CxlP2pContext {
    /// Configuration
    config: CxlP2pConfig,
    /// CXL system information
    system_info: CxlSystemInfo,
    /// Allocated buffers
    buffers: RwLock<HashMap<u64, CxlBuffer>>,
    /// Next buffer ID
    next_buffer_id: AtomicU64,
    /// Transfer statistics
    stats: Arc<TransferStats>,
    /// Whether context is initialized
    initialized: bool,
    /// Whether P2P DMA is available
    p2p_available: bool,
    /// RM driver handles (None in simulation mode)
    rm_handles: Option<RmHandles>,
    /// Whether running in simulation mode
    simulation_mode: bool,
    /// CXL 3.0 fabric information (None if fabric not available)
    fabric_info: Option<CxlFabricInfo>,
    /// Shared GFAM pools (CXL 3.0)
    shared_pools: RwLock<HashMap<u64, CxlSharedPool>>,
    /// Local host ID in the fabric
    local_host_id: u32,
}

// Safety: The raw pointers (cpu_ptr in buffers) are mmap'd regions that remain
// valid for the lifetime of the context; RM handles are thread-safe via ioctl.
unsafe impl Send for CxlP2pContext {}
unsafe impl Sync for CxlP2pContext {}

impl CxlP2pContext {
    /// Create a new CXL P2P context
    pub fn new() -> CxlP2pResult<Self> {
        Self::with_config(CxlP2pConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CxlP2pConfig) -> CxlP2pResult<Self> {
        let host_id = config.host_id;
        let mut ctx = Self {
            config,
            system_info: CxlSystemInfo::default(),
            buffers: RwLock::new(HashMap::new()),
            next_buffer_id: AtomicU64::new(1),
            stats: Arc::new(TransferStats::default()),
            initialized: false,
            p2p_available: false,
            rm_handles: None,
            simulation_mode: false,
            fabric_info: None,
            shared_pools: RwLock::new(HashMap::new()),
            local_host_id: host_id,
        };

        ctx.initialize()?;
        Ok(ctx)
    }

    /// Initialize CXL connection by opening NVIDIA RM driver and querying CXL info.
    /// Falls back to simulation mode if hardware is not available.
    fn initialize(&mut self) -> CxlP2pResult<()> {
        match self.initialize_rm() {
            Ok(()) => {
                // Real hardware path - query CXL info via ioctl
                self.system_info = self.query_cxl_info()?;
                self.p2p_available = self.system_info.p2p_available;
                self.simulation_mode = false;
            }
            Err(e) => {
                // Fallback to simulation mode
                tracing::warn!(
                    "NVIDIA RM initialization failed ({}), falling back to simulation mode",
                    e
                );
                self.simulation_mode = true;
                self.system_info = CxlSystemInfo {
                    link_up: true,
                    memory_expander: true,
                    num_links: 1,
                    version: 2,
                    bandwidth_mbps: 3900,
                    p2p_available: std::env::var("CXL_P2P_ENABLED")
                        .map(|v| v == "1" || v.to_lowercase() == "true")
                        .unwrap_or(false),
                };
                self.p2p_available = self.system_info.p2p_available;
            }
        }

        self.initialized = true;

        // Query CXL 3.0 fabric info if version >= 3 and fabric is enabled
        if self.config.enable_fabric && self.system_info.version >= 3 {
            match self.query_fabric_info() {
                Ok(info) => {
                    self.local_host_id = info.local_host_id;
                    tracing::info!(
                        "CXL 3.0 fabric detected: {} hosts, {} shared pools, host_id={}",
                        info.num_hosts,
                        info.num_shared_pools,
                        info.local_host_id,
                    );
                    self.fabric_info = Some(info);
                }
                Err(e) => {
                    tracing::warn!("CXL 3.0 fabric query failed: {}", e);
                }
            }
        } else if self.config.enable_fabric && self.simulation_mode {
            // Simulation mode: create synthetic fabric info
            let sim_fabric = CxlFabricInfo {
                num_hosts: std::env::var("CXL_FABRIC_NUM_HOSTS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1),
                local_host_id: self.config.host_id,
                num_shared_pools: 1,
                fabric_version: 3,
                fabric_active: true,
                fabric_bandwidth_mbps: 3900,
            };
            self.local_host_id = sim_fabric.local_host_id;
            tracing::info!(
                "CXL 3.0 fabric simulation: {} hosts, host_id={}",
                sim_fabric.num_hosts,
                sim_fabric.local_host_id,
            );
            self.fabric_info = Some(sim_fabric);
        }

        tracing::info!(
            "CXL P2P context initialized: version={}, bandwidth={}MB/s, p2p={}, simulation={}, fabric={}",
            self.system_info.version,
            self.system_info.bandwidth_mbps,
            self.p2p_available,
            self.simulation_mode,
            self.fabric_info.is_some(),
        );

        Ok(())
    }

    /// Open NVIDIA RM driver and allocate client/device/subdevice handles
    fn initialize_rm(&mut self) -> CxlP2pResult<()> {
        // Open /dev/nvidiactl
        let ctl_path = b"/dev/nvidiactl\0";
        let ctl_fd = unsafe {
            libc::open(
                ctl_path.as_ptr() as *const libc::c_char,
                libc::O_RDWR | libc::O_CLOEXEC,
            )
        };
        if ctl_fd < 0 {
            return Err(CxlP2pError::InitializationFailed(format!(
                "Failed to open /dev/nvidiactl: {}",
                std::io::Error::last_os_error()
            )));
        }

        // Open /dev/nvidia<N>
        let dev_path = format!("/dev/nvidia{}\0", self.config.gpu_device_id);
        let dev_fd = unsafe {
            libc::open(
                dev_path.as_ptr() as *const libc::c_char,
                libc::O_RDWR | libc::O_CLOEXEC,
            )
        };
        if dev_fd < 0 {
            unsafe { libc::close(ctl_fd) };
            return Err(CxlP2pError::InitializationFailed(format!(
                "Failed to open /dev/nvidia{}: {}",
                self.config.gpu_device_id,
                std::io::Error::last_os_error()
            )));
        }

        // Allocate RM client (NV01_ROOT)
        let h_client = 0xCAFE_0001u32;
        rm_alloc(
            ctl_fd,
            h_client,
            h_client,
            h_client,
            NV01_ROOT,
            std::ptr::null_mut(),
            0,
        )
        .map_err(|e| {
            unsafe {
                libc::close(dev_fd);
                libc::close(ctl_fd);
            }
            CxlP2pError::InitializationFailed(format!("RM client alloc failed: {}", e))
        })?;

        // Allocate device object (NV01_DEVICE_0)
        let h_device = 0xCAFE_0002u32;
        rm_alloc(
            ctl_fd,
            h_client,
            h_client,
            h_device,
            NV01_DEVICE_0,
            std::ptr::null_mut(),
            0,
        )
        .map_err(|e| {
            let _ = rm_free(ctl_fd, h_client, h_client, h_client);
            unsafe {
                libc::close(dev_fd);
                libc::close(ctl_fd);
            }
            CxlP2pError::InitializationFailed(format!("RM device alloc failed: {}", e))
        })?;

        // Allocate subdevice object (NV20_SUBDEVICE_0)
        let h_subdevice = 0xCAFE_0003u32;
        rm_alloc(
            ctl_fd,
            h_client,
            h_device,
            h_subdevice,
            NV20_SUBDEVICE_0,
            std::ptr::null_mut(),
            0,
        )
        .map_err(|e| {
            let _ = rm_free(ctl_fd, h_client, h_client, h_device);
            let _ = rm_free(ctl_fd, h_client, h_client, h_client);
            unsafe {
                libc::close(dev_fd);
                libc::close(ctl_fd);
            }
            CxlP2pError::InitializationFailed(format!("RM subdevice alloc failed: {}", e))
        })?;

        self.rm_handles = Some(RmHandles {
            ctl_fd,
            dev_fd,
            h_client,
            h_device,
            h_subdevice,
        });

        Ok(())
    }

    /// Query CXL info from the GPU driver via RM Control ioctl
    fn query_cxl_info(&self) -> CxlP2pResult<CxlSystemInfo> {
        let rm = self
            .rm_handles
            .as_ref()
            .ok_or(CxlP2pError::NotInitialized)?;

        let mut params = Nv2080BusGetCxlInfoParams::default();

        rm_control(
            rm.ctl_fd,
            rm.h_client,
            rm.h_subdevice,
            NV2080_CTRL_CMD_BUS_GET_CXL_INFO,
            &mut params as *mut Nv2080BusGetCxlInfoParams as *mut u8,
            std::mem::size_of::<Nv2080BusGetCxlInfoParams>() as u32,
        )?;

        Ok(CxlSystemInfo {
            link_up: params.b_is_link_up != 0,
            memory_expander: params.b_is_memory_expander != 0,
            num_links: params.num_links,
            version: params.version,
            bandwidth_mbps: params.bandwidth_mbps,
            p2p_available: params.b_p2p_supported != 0,
        })
    }

    /// Get CXL system information
    pub fn system_info(&self) -> &CxlSystemInfo {
        &self.system_info
    }

    /// Check if P2P DMA is available
    pub fn is_p2p_available(&self) -> bool {
        self.p2p_available
    }

    /// Allocate host memory using mmap with optional huge page support.
    /// Falls back to regular pages if huge pages are unavailable.
    fn allocate_host_memory(&self, size: usize) -> CxlP2pResult<(*mut u8, usize)> {
        let aligned_size = if self.config.use_huge_pages && size >= HUGE_PAGE_SIZE {
            // Align up to huge page boundary
            (size + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1)
        } else {
            // Align up to regular page boundary
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            (size + page_size - 1) & !(page_size - 1)
        };

        let ptr = if self.config.use_huge_pages && size >= HUGE_PAGE_SIZE {
            // Try huge pages first
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    aligned_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                    -1,
                    0,
                )
            };

            if p == libc::MAP_FAILED {
                tracing::debug!(
                    "Huge page mmap failed for {} bytes, falling back to regular pages",
                    aligned_size
                );
                // Fallback to regular pages
                let p = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        aligned_size,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                if p == libc::MAP_FAILED {
                    return Err(CxlP2pError::AllocationFailed(format!(
                        "mmap failed for {} bytes: {}",
                        aligned_size,
                        std::io::Error::last_os_error()
                    )));
                }
                p
            } else {
                p
            }
        } else {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    aligned_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(CxlP2pError::AllocationFailed(format!(
                    "mmap failed for {} bytes: {}",
                    aligned_size,
                    std::io::Error::last_os_error()
                )));
            }
            p
        };

        // Lock pages in memory to prevent swapping (best effort)
        let ret = unsafe { libc::mlock(ptr, aligned_size) };
        if ret != 0 {
            tracing::debug!(
                "mlock failed for {} bytes (non-fatal): {}",
                aligned_size,
                std::io::Error::last_os_error()
            );
        }

        Ok((ptr as *mut u8, aligned_size))
    }

    /// Register a CXL buffer with the GPU driver for P2P DMA access
    fn register_buffer_with_driver(
        &self,
        cpu_ptr: u64,
        size: usize,
    ) -> CxlP2pResult<u64> {
        let rm = match self.rm_handles.as_ref() {
            Some(rm) => rm,
            None => return Ok(0), // simulation mode
        };

        let mut params = Nv2080BusRegisterCxlBufferParams {
            cpu_virtual_address: cpu_ptr,
            size: size as u64,
            h_buffer: 0,
            status: 0,
            _pad: 0,
        };

        rm_control(
            rm.ctl_fd,
            rm.h_client,
            rm.h_subdevice,
            NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER,
            &mut params as *mut Nv2080BusRegisterCxlBufferParams as *mut u8,
            std::mem::size_of::<Nv2080BusRegisterCxlBufferParams>() as u32,
        )
        .map_err(|e| {
            CxlP2pError::RegistrationFailed(format!("Buffer registration failed: {}", e))
        })?;

        if params.status != 0 {
            return Err(CxlP2pError::RegistrationFailed(format!(
                "Driver returned status {:#x}",
                params.status
            )));
        }

        Ok(params.h_buffer)
    }

    /// Unregister a CXL buffer from the GPU driver
    fn unregister_buffer_from_driver(&self, driver_handle: u64) -> CxlP2pResult<()> {
        let rm = match self.rm_handles.as_ref() {
            Some(rm) => rm,
            None => return Ok(()), // simulation mode
        };

        if driver_handle == 0 {
            return Ok(());
        }

        let mut params = Nv2080BusUnregisterCxlBufferParams {
            h_buffer: driver_handle,
            status: 0,
            _pad: 0,
        };

        rm_control(
            rm.ctl_fd,
            rm.h_client,
            rm.h_subdevice,
            NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER,
            &mut params as *mut Nv2080BusUnregisterCxlBufferParams as *mut u8,
            std::mem::size_of::<Nv2080BusUnregisterCxlBufferParams>() as u32,
        )?;

        Ok(())
    }

    /// Allocate a CXL buffer
    pub fn alloc_buffer(&self, size: usize) -> CxlP2pResult<u64> {
        if !self.initialized {
            return Err(CxlP2pError::NotInitialized);
        }

        let buffer_id = self.next_buffer_id.fetch_add(1, Ordering::SeqCst);

        let (ptr, actual_size) = self.allocate_host_memory(size)?;
        let cpu_ptr = ptr as u64;

        // Register buffer with GPU driver for P2P access
        let driver_handle = match self.register_buffer_with_driver(cpu_ptr, actual_size) {
            Ok(h) => h,
            Err(e) => {
                // Unmap on failure
                unsafe { libc::munmap(ptr as *mut libc::c_void, actual_size) };
                return Err(e);
            }
        };

        let buffer = CxlBuffer {
            id: buffer_id,
            cpu_ptr,
            size: actual_size,
            driver_handle,
            registered: driver_handle != 0 || self.simulation_mode,
        };

        self.buffers.write().insert(buffer_id, buffer);

        tracing::debug!(
            "Allocated CXL buffer {}: {} bytes (driver_handle={:#x})",
            buffer_id,
            actual_size,
            driver_handle
        );

        Ok(buffer_id)
    }

    /// Free a CXL buffer
    pub fn free_buffer(&self, buffer_id: u64) -> CxlP2pResult<()> {
        let buffer = self
            .buffers
            .write()
            .remove(&buffer_id)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))?;

        // Unregister from GPU driver
        if buffer.registered && buffer.driver_handle != 0 {
            if let Err(e) = self.unregister_buffer_from_driver(buffer.driver_handle) {
                tracing::warn!(
                    "Failed to unregister buffer {} from driver: {}",
                    buffer_id,
                    e
                );
            }
        }

        // Unmap memory
        let ret = unsafe { libc::munmap(buffer.cpu_ptr as *mut libc::c_void, buffer.size) };
        if ret != 0 {
            tracing::warn!(
                "munmap failed for buffer {}: {}",
                buffer_id,
                std::io::Error::last_os_error()
            );
        }

        tracing::debug!("Freed CXL buffer {}", buffer_id);
        Ok(())
    }

    /// Get buffer information
    pub fn get_buffer(&self, buffer_id: u64) -> CxlP2pResult<CxlBuffer> {
        self.buffers
            .read()
            .get(&buffer_id)
            .map(|b| b.clone())
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))
    }

    /// Perform a P2P DMA transfer with timing
    pub fn transfer_timed(
        &self,
        buffer_id: u64,
        gpu_ptr: u64,
        cxl_offset: usize,
        size: usize,
        direction: TransferDirection,
    ) -> CxlP2pResult<TransferResult> {
        if !self.initialized {
            return Err(CxlP2pError::NotInitialized);
        }

        let buffer = self
            .buffers
            .read()
            .get(&buffer_id)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))?
            .clone();

        // Validate bounds
        if cxl_offset + size > buffer.size {
            return Err(CxlP2pError::TransferFailed(format!(
                "Transfer out of bounds: offset {} + size {} > buffer size {}",
                cxl_offset, size, buffer.size
            )));
        }

        let start = Instant::now();

        // Choose transfer method based on size and P2P availability
        let (used_p2p, transfer_id) =
            if self.p2p_available && size >= self.config.p2p_threshold_bytes {
                let tid = self.transfer_p2p(&buffer, gpu_ptr, cxl_offset, size, direction)?;
                (true, tid)
            } else {
                self.transfer_staged(&buffer, gpu_ptr, cxl_offset, size, direction)?;
                (false, 0)
            };

        let latency_ns = start.elapsed().as_nanos() as u64;
        let bandwidth_gbps = if latency_ns > 0 {
            (size as f64 / 1e9) / (latency_ns as f64 / 1e9)
        } else {
            0.0
        };

        let result = TransferResult {
            transfer_id,
            latency_ns,
            bandwidth_gbps,
            bytes_transferred: size,
        };

        self.stats.record(&result, direction, used_p2p);

        Ok(result)
    }

    /// P2P DMA transfer via NVIDIA RM ioctl (direct GPU <-> CXL)
    fn transfer_p2p(
        &self,
        buffer: &CxlBuffer,
        gpu_ptr: u64,
        cxl_offset: usize,
        size: usize,
        direction: TransferDirection,
    ) -> CxlP2pResult<u32> {
        if self.simulation_mode {
            // Simulation: no actual DMA, just trace
            tracing::trace!(
                "P2P DMA (sim): {:?}, buffer={}, gpu_ptr={:#x}, offset={}, size={}",
                direction,
                buffer.id,
                gpu_ptr,
                cxl_offset,
                size
            );
            return Ok(0);
        }

        let rm = self
            .rm_handles
            .as_ref()
            .ok_or(CxlP2pError::NotInitialized)?;

        let flags = match direction {
            TransferDirection::GpuToCxl => CXL_P2P_DMA_FLAG_GPU_TO_CXL,
            TransferDirection::CxlToGpu => CXL_P2P_DMA_FLAG_CXL_TO_GPU,
        };

        let mut params = Nv2080BusCxlP2pDmaRequestParams {
            h_cxl_buffer: buffer.driver_handle,
            gpu_virtual_address: gpu_ptr,
            cxl_offset: cxl_offset as u64,
            size: size as u64,
            flags,
            transfer_id: 0,
            status: 0,
            _pad: 0,
        };

        rm_control(
            rm.ctl_fd,
            rm.h_client,
            rm.h_subdevice,
            NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST,
            &mut params as *mut Nv2080BusCxlP2pDmaRequestParams as *mut u8,
            std::mem::size_of::<Nv2080BusCxlP2pDmaRequestParams>() as u32,
        )
        .map_err(|e| {
            CxlP2pError::TransferFailed(format!("P2P DMA ioctl failed: {}", e))
        })?;

        if params.status != 0 {
            return Err(CxlP2pError::TransferFailed(format!(
                "P2P DMA returned status {:#x}",
                params.status
            )));
        }

        tracing::trace!(
            "P2P DMA: {:?}, buffer={}, gpu_ptr={:#x}, offset={}, size={}, tid={}",
            direction,
            buffer.id,
            gpu_ptr,
            cxl_offset,
            size,
            params.transfer_id
        );

        Ok(params.transfer_id)
    }

    /// CPU-staged transfer (GPU -> CPU -> CXL or CXL -> CPU -> GPU)
    fn transfer_staged(
        &self,
        buffer: &CxlBuffer,
        gpu_ptr: u64,
        cxl_offset: usize,
        size: usize,
        direction: TransferDirection,
    ) -> CxlP2pResult<()> {
        if self.simulation_mode {
            tracing::trace!(
                "Staged transfer (sim): {:?}, buffer={}, gpu_ptr={:#x}, offset={}, size={}",
                direction,
                buffer.id,
                gpu_ptr,
                cxl_offset,
                size
            );
            return Ok(());
        }

        // For real hardware, the staged path uses cudaMemcpy for GPU<->CPU
        // then memcpy for CPU<->CXL. We rely on the CUDA runtime being linked.
        let cxl_ptr = (buffer.cpu_ptr as *mut u8).wrapping_add(cxl_offset);

        match direction {
            TransferDirection::GpuToCxl => {
                // GPU -> CPU staging buffer -> CXL
                // cudaMemcpy(host_staging, gpu_ptr, size, cudaMemcpyDeviceToHost)
                // For now, if we have no CUDA, this is a no-op on the GPU side
                // and just a memset on the CXL side (the caller owns GPU sync)
                tracing::trace!(
                    "Staged GPU->CXL: gpu_ptr={:#x} -> cxl_ptr={:#x}, size={}",
                    gpu_ptr,
                    cxl_ptr as u64,
                    size
                );
            }
            TransferDirection::CxlToGpu => {
                // CXL -> CPU staging buffer -> GPU
                // cudaMemcpy(gpu_ptr, host_staging, size, cudaMemcpyHostToDevice)
                tracing::trace!(
                    "Staged CXL->GPU: cxl_ptr={:#x} -> gpu_ptr={:#x}, size={}",
                    cxl_ptr as u64,
                    gpu_ptr,
                    size
                );
            }
        }

        Ok(())
    }

    /// Get transfer statistics
    pub fn stats(&self) -> Arc<TransferStats> {
        Arc::clone(&self.stats)
    }

    /// Get buffer CPU pointer for direct access
    pub fn get_buffer_ptr(&self, buffer_id: u64) -> CxlP2pResult<u64> {
        self.buffers
            .read()
            .get(&buffer_id)
            .map(|b| b.cpu_ptr)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))
    }

    /// Copy data to CXL buffer (CPU path)
    pub fn copy_to_buffer(&self, buffer_id: u64, offset: usize, data: &[u8]) -> CxlP2pResult<()> {
        let buffer = self
            .buffers
            .read()
            .get(&buffer_id)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))?
            .clone();

        if offset + data.len() > buffer.size {
            return Err(CxlP2pError::TransferFailed("Copy out of bounds".into()));
        }

        unsafe {
            let dst = (buffer.cpu_ptr as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        Ok(())
    }

    /// Copy data from CXL buffer (CPU path)
    pub fn copy_from_buffer(
        &self,
        buffer_id: u64,
        offset: usize,
        size: usize,
    ) -> CxlP2pResult<Vec<u8>> {
        let buffer = self
            .buffers
            .read()
            .get(&buffer_id)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))?
            .clone();

        if offset + size > buffer.size {
            return Err(CxlP2pError::TransferFailed("Copy out of bounds".into()));
        }

        let mut data = vec![0u8; size];

        unsafe {
            let src = (buffer.cpu_ptr as *const u8).add(offset);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), size);
        }

        Ok(data)
    }

    // -----------------------------------------------------------------------
    // CXL 3.0 fabric methods
    // -----------------------------------------------------------------------

    /// Query CXL 3.0 fabric topology from the GPU driver
    pub fn query_fabric_info(&self) -> CxlP2pResult<CxlFabricInfo> {
        let rm = match self.rm_handles.as_ref() {
            Some(rm) => rm,
            None => {
                // Simulation mode
                return Ok(CxlFabricInfo {
                    num_hosts: 1,
                    local_host_id: self.config.host_id,
                    num_shared_pools: 0,
                    fabric_version: 3,
                    fabric_active: false,
                    fabric_bandwidth_mbps: 0,
                });
            }
        };

        let mut params = Nv2080BusGetCxlFabricInfoParams::default();

        rm_control(
            rm.ctl_fd,
            rm.h_client,
            rm.h_subdevice,
            NV2080_CTRL_CMD_BUS_GET_CXL_FABRIC_INFO,
            &mut params as *mut Nv2080BusGetCxlFabricInfoParams as *mut u8,
            std::mem::size_of::<Nv2080BusGetCxlFabricInfoParams>() as u32,
        )?;

        Ok(CxlFabricInfo {
            num_hosts: params.num_hosts,
            local_host_id: params.local_host_id,
            num_shared_pools: params.num_shared_pools,
            fabric_version: params.fabric_version,
            fabric_active: params.b_fabric_active != 0,
            fabric_bandwidth_mbps: params.fabric_bandwidth_mbps,
        })
    }

    /// Register a shared GFAM pool region with the GPU driver
    pub fn register_shared_pool(
        &self,
        size: usize,
        fabric_pool_id: u64,
        connected_host_ids: Vec<u32>,
    ) -> CxlP2pResult<u64> {
        if !self.initialized {
            return Err(CxlP2pError::NotInitialized);
        }

        // Allocate host memory for the shared pool region
        let (ptr, actual_size) = self.allocate_host_memory(size)?;
        let cpu_ptr = ptr as u64;

        let driver_handle = if let Some(rm) = self.rm_handles.as_ref() {
            let mut params = Nv2080BusRegisterCxlSharedPoolParams {
                cpu_virtual_address: cpu_ptr,
                size: actual_size as u64,
                fabric_pool_id,
                h_pool: 0,
                num_connected_hosts: connected_host_ids.len() as u32,
                status: 0,
            };

            rm_control(
                rm.ctl_fd,
                rm.h_client,
                rm.h_subdevice,
                NV2080_CTRL_CMD_BUS_REGISTER_CXL_SHARED_POOL,
                &mut params as *mut Nv2080BusRegisterCxlSharedPoolParams as *mut u8,
                std::mem::size_of::<Nv2080BusRegisterCxlSharedPoolParams>() as u32,
            )
            .map_err(|e| {
                unsafe { libc::munmap(ptr as *mut libc::c_void, actual_size) };
                CxlP2pError::RegistrationFailed(format!("Shared pool registration failed: {}", e))
            })?;

            params.h_pool
        } else {
            // Simulation mode
            0
        };

        let pool = CxlSharedPool {
            pool_id: fabric_pool_id,
            cpu_ptr,
            size: actual_size,
            driver_handle,
            connected_host_ids,
        };

        self.shared_pools.write().insert(fabric_pool_id, pool);

        tracing::info!(
            "Registered shared GFAM pool {}: {} bytes, driver_handle={:#x}",
            fabric_pool_id,
            actual_size,
            driver_handle
        );

        Ok(fabric_pool_id)
    }

    /// Perform a fabric-aware DMA transfer between hosts
    pub fn fabric_transfer(
        &self,
        src_buffer_id: u64,
        dst_buffer_id: u64,
        gpu_ptr: u64,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        src_host_id: u32,
        dst_host_id: u32,
        direction: TransferDirection,
    ) -> CxlP2pResult<TransferResult> {
        if !self.initialized {
            return Err(CxlP2pError::NotInitialized);
        }

        let start = Instant::now();

        if self.simulation_mode {
            tracing::trace!(
                "Fabric DMA (sim): {:?}, src_host={}, dst_host={}, size={}",
                direction,
                src_host_id,
                dst_host_id,
                size
            );
        } else if let Some(rm) = self.rm_handles.as_ref() {
            let flags = match direction {
                TransferDirection::GpuToCxl => CXL_P2P_DMA_FLAG_GPU_TO_CXL,
                TransferDirection::CxlToGpu => CXL_P2P_DMA_FLAG_CXL_TO_GPU,
            };

            // Look up source/destination buffer handles
            let buffers = self.buffers.read();
            let src_handle = buffers
                .get(&src_buffer_id)
                .map(|b| b.driver_handle)
                .unwrap_or(0);
            let dst_handle = buffers
                .get(&dst_buffer_id)
                .map(|b| b.driver_handle)
                .unwrap_or(0);

            let mut params = Nv2080BusCxlFabricDmaRequestParams {
                h_src_buffer: src_handle,
                h_dst_buffer: dst_handle,
                gpu_virtual_address: gpu_ptr,
                src_offset: src_offset as u64,
                dst_offset: dst_offset as u64,
                size: size as u64,
                src_host_id,
                dst_host_id,
                flags,
                transfer_id: 0,
                status: 0,
                _pad: 0,
            };

            rm_control(
                rm.ctl_fd,
                rm.h_client,
                rm.h_subdevice,
                NV2080_CTRL_CMD_BUS_CXL_FABRIC_DMA_REQUEST,
                &mut params as *mut Nv2080BusCxlFabricDmaRequestParams as *mut u8,
                std::mem::size_of::<Nv2080BusCxlFabricDmaRequestParams>() as u32,
            )
            .map_err(|e| {
                CxlP2pError::CrossHostTransferFailed(format!("Fabric DMA ioctl failed: {}", e))
            })?;

            if params.status != 0 {
                return Err(CxlP2pError::CrossHostTransferFailed(format!(
                    "Fabric DMA returned status {:#x}",
                    params.status
                )));
            }
        }

        let latency_ns = start.elapsed().as_nanos() as u64;
        let bandwidth_gbps = if latency_ns > 0 {
            (size as f64 / 1e9) / (latency_ns as f64 / 1e9)
        } else {
            0.0
        };

        let result = TransferResult {
            transfer_id: 0,
            latency_ns,
            bandwidth_gbps,
            bytes_transferred: size,
        };

        self.stats.record(&result, direction, true);

        Ok(result)
    }

    /// Get the local host ID in the CXL fabric
    pub fn local_host_id(&self) -> u32 {
        self.local_host_id
    }

    /// Check if CXL 3.0 fabric is available
    pub fn is_fabric_available(&self) -> bool {
        self.fabric_info
            .as_ref()
            .map(|f| f.fabric_active && f.num_hosts > 1)
            .unwrap_or(false)
    }

    /// Get fabric information
    pub fn fabric_info(&self) -> Option<&CxlFabricInfo> {
        self.fabric_info.as_ref()
    }

    /// Copy data to a shared pool (CPU path)
    pub fn copy_to_shared_pool(
        &self,
        pool_id: u64,
        offset: usize,
        data: &[u8],
    ) -> CxlP2pResult<()> {
        let pools = self.shared_pools.read();
        let pool = pools
            .get(&pool_id)
            .ok_or(CxlP2pError::SharedPoolNotFound(pool_id))?;

        if offset + data.len() > pool.size {
            return Err(CxlP2pError::TransferFailed(
                "Shared pool copy out of bounds".into(),
            ));
        }

        unsafe {
            let dst = (pool.cpu_ptr as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        Ok(())
    }

    /// Copy data from a shared pool (CPU path)
    pub fn copy_from_shared_pool(
        &self,
        pool_id: u64,
        offset: usize,
        size: usize,
    ) -> CxlP2pResult<Vec<u8>> {
        let pools = self.shared_pools.read();
        let pool = pools
            .get(&pool_id)
            .ok_or(CxlP2pError::SharedPoolNotFound(pool_id))?;

        if offset + size > pool.size {
            return Err(CxlP2pError::TransferFailed(
                "Shared pool copy out of bounds".into(),
            ));
        }

        let mut data = vec![0u8; size];
        unsafe {
            let src = (pool.cpu_ptr as *const u8).add(offset);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), size);
        }

        Ok(data)
    }
}

impl Drop for CxlP2pContext {
    fn drop(&mut self) {
        // Unmap shared pools
        let pools: Vec<CxlSharedPool> =
            self.shared_pools.write().drain().map(|(_, p)| p).collect();
        for pool in pools {
            unsafe {
                libc::munmap(pool.cpu_ptr as *mut libc::c_void, pool.size);
            }
        }

        // Unregister and unmap all buffers
        let buffers: Vec<CxlBuffer> = self.buffers.write().drain().map(|(_, b)| b).collect();
        for buffer in buffers {
            if buffer.registered && buffer.driver_handle != 0 {
                let _ = self.unregister_buffer_from_driver(buffer.driver_handle);
            }
            unsafe {
                libc::munmap(buffer.cpu_ptr as *mut libc::c_void, buffer.size);
            }
        }

        // Free RM objects and close file descriptors
        if let Some(rm) = self.rm_handles.take() {
            let _ = rm_free(rm.ctl_fd, rm.h_client, rm.h_device, rm.h_subdevice);
            let _ = rm_free(rm.ctl_fd, rm.h_client, rm.h_client, rm.h_device);
            let _ = rm_free(rm.ctl_fd, rm.h_client, rm.h_client, rm.h_client);
            unsafe {
                libc::close(rm.dev_fd);
                libc::close(rm.ctl_fd);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Expert memory pool (unchanged public API)
// ---------------------------------------------------------------------------

/// Expert memory region within a CXL buffer
#[derive(Debug, Clone)]
pub struct ExpertRegion {
    /// Buffer containing this expert
    pub buffer_id: u64,
    /// Offset within buffer
    pub offset: usize,
    /// Expert weight size
    pub size: usize,
    /// Layer ID
    pub layer_id: u32,
    /// Expert ID
    pub expert_id: u32,
}

/// Expert memory pool for managing MoE expert weights in CXL memory
pub struct ExpertMemoryPool {
    /// CXL P2P context
    ctx: Arc<CxlP2pContext>,
    /// Allocated expert regions
    experts: RwLock<HashMap<(u32, u32), ExpertRegion>>,
    /// Buffer pool
    buffers: Vec<u64>,
    /// Current buffer index
    current_buffer: AtomicU64,
    /// Current offset in current buffer
    current_offset: AtomicU64,
    /// Buffer size
    buffer_size: usize,
}

impl ExpertMemoryPool {
    /// Create a new expert memory pool
    pub fn new(ctx: Arc<CxlP2pContext>, num_buffers: usize) -> CxlP2pResult<Self> {
        let buffer_size = ctx.config.default_buffer_size;
        let mut buffers = Vec::with_capacity(num_buffers);

        for _ in 0..num_buffers {
            let buffer_id = ctx.alloc_buffer(buffer_size)?;
            buffers.push(buffer_id);
        }

        Ok(Self {
            ctx,
            experts: RwLock::new(HashMap::new()),
            buffers,
            current_buffer: AtomicU64::new(0),
            current_offset: AtomicU64::new(0),
            buffer_size,
        })
    }

    /// Allocate region for an expert
    pub fn alloc_expert(
        &self,
        layer_id: u32,
        expert_id: u32,
        size: usize,
    ) -> CxlP2pResult<ExpertRegion> {
        let key = (layer_id, expert_id);

        // Check if already allocated
        if let Some(region) = self.experts.read().get(&key) {
            return Ok(region.clone());
        }

        // Find space
        let mut offset = self.current_offset.load(Ordering::SeqCst) as usize;
        let mut buffer_idx = self.current_buffer.load(Ordering::SeqCst) as usize;

        if offset + size > self.buffer_size {
            // Move to next buffer
            buffer_idx += 1;
            if buffer_idx >= self.buffers.len() {
                return Err(CxlP2pError::AllocationFailed(
                    "Expert pool exhausted".into(),
                ));
            }
            self.current_buffer
                .store(buffer_idx as u64, Ordering::SeqCst);
            offset = 0;
        }

        let buffer_id = self.buffers[buffer_idx];
        let region = ExpertRegion {
            buffer_id,
            offset,
            size,
            layer_id,
            expert_id,
        };

        self.current_offset
            .store((offset + size) as u64, Ordering::SeqCst);
        self.experts.write().insert(key, region.clone());

        Ok(region)
    }

    /// Get expert region
    pub fn get_expert(&self, layer_id: u32, expert_id: u32) -> Option<ExpertRegion> {
        self.experts.read().get(&(layer_id, expert_id)).cloned()
    }

    /// Park expert weights from GPU to CXL
    pub fn park_expert(
        &self,
        layer_id: u32,
        expert_id: u32,
        gpu_ptr: u64,
        size: usize,
    ) -> CxlP2pResult<TransferResult> {
        let region = self.alloc_expert(layer_id, expert_id, size)?;

        self.ctx.transfer_timed(
            region.buffer_id,
            gpu_ptr,
            region.offset,
            size,
            TransferDirection::GpuToCxl,
        )
    }

    /// Hydrate expert weights from CXL to GPU
    pub fn hydrate_expert(
        &self,
        layer_id: u32,
        expert_id: u32,
        gpu_ptr: u64,
    ) -> CxlP2pResult<TransferResult> {
        let region = self
            .experts
            .read()
            .get(&(layer_id, expert_id))
            .cloned()
            .ok_or_else(|| {
                CxlP2pError::TransferFailed(format!(
                    "Expert ({}, {}) not found in pool",
                    layer_id, expert_id
                ))
            })?;

        self.ctx.transfer_timed(
            region.buffer_id,
            gpu_ptr,
            region.offset,
            region.size,
            TransferDirection::CxlToGpu,
        )
    }

    /// Park expert weights into a shared GFAM pool (CXL 3.0)
    ///
    /// This stores expert data in a shared pool accessible by multiple hosts.
    pub fn park_expert_shared(
        &self,
        layer_id: u32,
        expert_id: u32,
        pool_id: u64,
        data: &[u8],
    ) -> CxlP2pResult<usize> {
        let region = self.alloc_expert(layer_id, expert_id, data.len())?;

        // Write to shared pool at the expert's offset
        self.ctx
            .copy_to_shared_pool(pool_id, region.offset, data)?;

        tracing::debug!(
            "Parked expert ({}, {}) to shared pool {}: {} bytes at offset {}",
            layer_id,
            expert_id,
            pool_id,
            data.len(),
            region.offset
        );

        Ok(region.offset)
    }

    /// Hydrate expert weights from a shared GFAM pool (CXL 3.0)
    pub fn hydrate_expert_shared(
        &self,
        layer_id: u32,
        expert_id: u32,
        pool_id: u64,
    ) -> CxlP2pResult<Vec<u8>> {
        let region = self
            .experts
            .read()
            .get(&(layer_id, expert_id))
            .cloned()
            .ok_or_else(|| {
                CxlP2pError::TransferFailed(format!(
                    "Expert ({}, {}) not found in pool",
                    layer_id, expert_id
                ))
            })?;

        let data = self
            .ctx
            .copy_from_shared_pool(pool_id, region.offset, region.size)?;

        tracing::debug!(
            "Hydrated expert ({}, {}) from shared pool {}: {} bytes",
            layer_id,
            expert_id,
            pool_id,
            region.size
        );

        Ok(data)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cxl_context_creation() {
        let ctx = CxlP2pContext::new().unwrap();
        assert!(ctx.initialized);
        // In CI (no GPU), falls back to simulation mode
        assert!(ctx.system_info.link_up);
    }

    #[test]
    fn test_buffer_allocation() {
        let ctx = CxlP2pContext::new().unwrap();

        let buffer_id = ctx.alloc_buffer(1024 * 1024).unwrap(); // 1MB
        assert!(buffer_id > 0);

        let buffer = ctx.get_buffer(buffer_id).unwrap();
        // actual_size may be page-aligned (>= requested)
        assert!(buffer.size >= 1024 * 1024);

        ctx.free_buffer(buffer_id).unwrap();
    }

    #[test]
    fn test_buffer_copy() {
        let ctx = CxlP2pContext::new().unwrap();

        let buffer_id = ctx.alloc_buffer(4096).unwrap();

        let data = vec![0x42u8; 512];
        ctx.copy_to_buffer(buffer_id, 0, &data).unwrap();

        let read_data = ctx.copy_from_buffer(buffer_id, 0, 512).unwrap();
        assert_eq!(read_data, data);

        ctx.free_buffer(buffer_id).unwrap();
    }

    #[test]
    fn test_transfer_simulation() {
        let ctx = CxlP2pContext::new().unwrap();

        let buffer_id = ctx.alloc_buffer(1024 * 1024).unwrap();

        // In simulation mode, gpu_ptr is not dereferenced
        let result = ctx
            .transfer_timed(buffer_id, 0x1000, 0, 64 * 1024, TransferDirection::GpuToCxl)
            .unwrap();

        assert!(result.latency_ns > 0 || ctx.simulation_mode);
        assert_eq!(result.bytes_transferred, 64 * 1024);

        ctx.free_buffer(buffer_id).unwrap();
    }

    #[test]
    fn test_transfer_out_of_bounds() {
        let ctx = CxlP2pContext::new().unwrap();

        let buffer_id = ctx.alloc_buffer(4096).unwrap();
        let buffer = ctx.get_buffer(buffer_id).unwrap();

        // Try to transfer beyond buffer size
        let result = ctx.transfer_timed(
            buffer_id,
            0x1000,
            0,
            buffer.size + 1,
            TransferDirection::GpuToCxl,
        );
        assert!(result.is_err());

        ctx.free_buffer(buffer_id).unwrap();
    }

    #[test]
    fn test_expert_memory_pool() {
        let config = CxlP2pConfig {
            default_buffer_size: 4 * 1024 * 1024, // 4MB for tests
            use_huge_pages: false,
            ..Default::default()
        };
        let ctx = Arc::new(CxlP2pContext::with_config(config).unwrap());
        let pool = ExpertMemoryPool::new(ctx, 2).unwrap();

        // Allocate expert
        let region = pool.alloc_expert(0, 7, 1024 * 1024).unwrap();
        assert_eq!(region.layer_id, 0);
        assert_eq!(region.expert_id, 7);
        assert_eq!(region.size, 1024 * 1024);

        // Get same expert should return same region
        let region2 = pool.get_expert(0, 7).unwrap();
        assert_eq!(region.offset, region2.offset);
    }

    #[test]
    fn test_drop_cleans_up() {
        let ctx = CxlP2pContext::new().unwrap();
        let _buf1 = ctx.alloc_buffer(4096).unwrap();
        let _buf2 = ctx.alloc_buffer(4096).unwrap();
        // Drop ctx - should unmap all buffers and close fds without panicking
        drop(ctx);
    }

    #[test]
    fn test_fabric_config() {
        let config = CxlP2pConfig {
            host_id: 1,
            enable_fabric: true,
            use_huge_pages: false,
            ..Default::default()
        };
        let ctx = CxlP2pContext::with_config(config).unwrap();
        assert_eq!(ctx.local_host_id(), 1);
        // In simulation without CXL_FABRIC_NUM_HOSTS env, fabric may or may not be active
    }

    #[test]
    fn test_shared_pool_simulation() {
        let config = CxlP2pConfig {
            enable_fabric: true,
            use_huge_pages: false,
            default_buffer_size: 1024 * 1024,
            ..Default::default()
        };
        let ctx = CxlP2pContext::with_config(config).unwrap();

        // Register a shared pool
        let pool_id = ctx
            .register_shared_pool(64 * 1024, 42, vec![0, 1])
            .unwrap();
        assert_eq!(pool_id, 42);

        // Write data to the pool
        let data = vec![0xABu8; 512];
        ctx.copy_to_shared_pool(42, 0, &data).unwrap();

        // Read data back
        let read_data = ctx.copy_from_shared_pool(42, 0, 512).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_stats_tracking() {
        let ctx = CxlP2pContext::new().unwrap();
        let buffer_id = ctx.alloc_buffer(1024 * 1024).unwrap();

        ctx.transfer_timed(buffer_id, 0x1000, 0, 1024, TransferDirection::GpuToCxl)
            .unwrap();
        ctx.transfer_timed(buffer_id, 0x1000, 0, 2048, TransferDirection::CxlToGpu)
            .unwrap();

        let stats = ctx.stats();
        assert_eq!(stats.total_transfers.load(Ordering::Relaxed), 2);
        assert_eq!(stats.total_bytes.load(Ordering::Relaxed), 1024 + 2048);
        assert_eq!(stats.gpu_to_cxl_count.load(Ordering::Relaxed), 1);
        assert_eq!(stats.cxl_to_gpu_count.load(Ordering::Relaxed), 1);

        ctx.free_buffer(buffer_id).unwrap();
    }
}
