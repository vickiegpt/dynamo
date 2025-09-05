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

use super::*;

use cudarc::driver::{CudaEvent, CudaStream, sys::CUevent_flags, result as cuda_result};
use nixl_sys::Agent as NixlAgent;

use std::sync::Arc;
use std::thread::JoinHandle;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use std::sync::atomic::{AtomicU64, Ordering};
use dynamo_runtime::utils::pool::{Returnable, SyncPoolItem, SyncPool};

// Add debug tracking for event-receiver mapping
static EVENT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub enum CleanupInfo {
    DirectFree(Vec<u64>),  // Device pointers to free directly
    PinnedFree { pointers: Vec<u64> },  // Pinned host memory pointers to free
}

#[derive(Debug, Clone)]
pub struct DebugEventInfo {
    pub event_id: u64,
    pub transfer_direction: String,
    pub _worker_id: Option<u64>,  // WorkerID is u64 (unused but kept for API compatibility)
    pub timestamp: std::time::Instant,
    pub cleanup_ptrs: Option<Vec<u64>>,  // Device pointers to free when event completes (legacy)
    pub cleanup_info: Option<CleanupInfo>,  // Enhanced cleanup with pool support
}

// Pinned Buffer Resource for Pooling
#[derive(Debug)]
pub struct PinnedBuffer {
    pub ptr: u64,
    pub size: usize,
    pub id: u64,
}

impl Returnable for PinnedBuffer {
    fn on_return(&mut self) {
        tracing::debug!("Returning pinned buffer {} ({}KB) to pool", self.id, self.size / 1024);
    }
}

pub type SyncPinnedBufferPool = SyncPool<PinnedBuffer>;

pub struct TransferResources {
    src_buffer: SyncPoolItem<PinnedBuffer>,
    dst_buffer: SyncPoolItem<PinnedBuffer>,
}

impl TransferResources {
    /// Create TransferResources by acquiring 2 buffers from the context
    pub fn acquire_for_kernel_launch(
        ctx: &TransferContext,
        address_count: usize,
    ) -> Result<Self, TransferError> {
        tracing::debug!("🔧 Acquiring TransferResources for {} addresses (need 2 buffers)", address_count);

        // Acquire 2 buffers: one for src addresses, one for dst addresses
        let src_buffer = ctx.acquire_resources_for_transfer_sync(address_count)?;
        let dst_buffer = ctx.acquire_resources_for_transfer_sync(address_count)?;

        tracing::debug!("🎉 TransferResources ready: src=0x{:x}, dst=0x{:x}",
                       src_buffer.ptr, dst_buffer.ptr);

        Ok(Self {
            src_buffer,
            dst_buffer,
        })
    }

    /// Copy address arrays into the pinned buffers
    pub fn copy_addresses_to_buffers(
        &self,
        src_addresses: &[u64],
        dst_addresses: &[u64],
    ) -> Result<(), TransferError> {  // Returns (), not pointers
        if src_addresses.len() != dst_addresses.len() {
            return Err(TransferError::ExecutionError(
                format!("Address array length mismatch: src={}, dst={}",
                       src_addresses.len(), dst_addresses.len()).into()
            ));
        }

        let required_size = src_addresses.len() * std::mem::size_of::<u64>();

        // Check buffer sizes
        if self.src_buffer.size < required_size || self.dst_buffer.size < required_size {
            return Err(TransferError::ExecutionError(
                format!("Buffer too small: {}B needed", required_size).into()
            ));
        }

        // Copy addresses to pinned buffers
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_addresses.as_ptr(),
                self.src_buffer.ptr as *mut u64,
                src_addresses.len()
            );
            std::ptr::copy_nonoverlapping(
                dst_addresses.as_ptr(),
                self.dst_buffer.ptr as *mut u64,
                dst_addresses.len()
            );
        }

        tracing::debug!("📋 Copied {} address pairs to pinned buffers", src_addresses.len());

        Ok(())
    }

    /// Get the source buffer pointer (for kernel launch)
    pub fn src_ptr(&self) -> u64 {
        self.src_buffer.ptr
    }

    /// Get the destination buffer pointer (for kernel launch)
    pub fn dst_ptr(&self) -> u64 {
        self.dst_buffer.ptr
    }

    /// Get the CUDA stream (for kernel launch)
    pub fn cuda_stream<'a>(&self, ctx: &'a TransferContext) -> &'a Arc<CudaStream> {
        ctx.stream()
    }
}

impl Drop for TransferResources {
    fn drop(&mut self) {
        tracing::debug!("🔄 Releasing TransferResources: buffers {} & {} returning to pool",
                       self.src_buffer.id, self.dst_buffer.id);
        // SyncPoolItem Drop handles returning buffers to pool automatically
    }
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub enable_pool: bool,
    pub max_concurrent_transfers: usize,
    pub max_transfer_batch_size: usize,
    pub num_outer_components: usize,
    pub num_layers: usize,
}

pub struct TransferContext {
    nixl_agent: Arc<Option<NixlAgent>>,
    stream: Arc<CudaStream>,
    async_rt_handle: Handle,

    pinned_buffer_pool: Option<SyncPinnedBufferPool>,

    cuda_event_tx: mpsc::UnboundedSender<(CudaEvent, oneshot::Sender<()>, DebugEventInfo)>,
    cuda_event_worker: Option<JoinHandle<()>>,
    cancel_token: CancellationToken,
}

impl TransferContext {
    pub fn new(
        nixl_agent: Arc<Option<NixlAgent>>,
        stream: Arc<CudaStream>,
        async_rt_handle: Handle,
        config: Option<PoolConfig>,
    ) -> Self {

        let (cuda_event_tx, mut cuda_event_rx) =
            mpsc::unbounded_channel::<(CudaEvent, oneshot::Sender<()>, DebugEventInfo)>();

        let cancel_token = CancellationToken::new();

        let cancel_token_clone = cancel_token.clone();
        let cuda_event_worker = Self::setup_cuda_event_worker(cuda_event_rx, cancel_token_clone);
        let pool = if let Some(config) = config {
            if config.enable_pool {
                let pool_size = config.max_concurrent_transfers * 2 + 2;
                let buffer_size = config.max_transfer_batch_size
                    * config.num_outer_components
                    * config.num_layers
                    * std::mem::size_of::<u64>();

                tracing::info!(
                    "🏊 Creating pinned buffer pool: {} buffers × {}KB each ({}L×{}C×{}B×8)",
                    pool_size,
                    buffer_size / 1024,
                    config.num_layers,
                    config.num_outer_components,
                    config.max_transfer_batch_size
                );

                let total_memory_mb = (pool_size * buffer_size) / (1024 * 1024);
                tracing::info!("🏊 Total pool memory: {}MB", total_memory_mb);

                {
                    // Create initial pinned buffers
                    let mut initial_buffers = Vec::with_capacity(pool_size);
                    let mut successful_allocations = 0;

                    for i in 0..pool_size {
                        let ptr = unsafe {
                            crate::block_manager::block::transfer::cuda::allocate_pinned_memory(buffer_size)
                                .map_err(|e| {
                                    tracing::error!("❌ Failed to allocate pinned buffer {}/{}: {}", i+1, pool_size, e);
                                    e
                                })
                                .unwrap_or(0)
                        };

                        if ptr != 0 {
                            let buffer = PinnedBuffer {
                                ptr,
                                size: buffer_size,
                                id: i as u64,
                            };
                            initial_buffers.push(buffer);
                            successful_allocations += 1;
                            tracing::debug!("✅ Allocated pinned buffer {}/{}: 0x{:x} ({}KB)",
                                          i+1, pool_size, ptr, buffer_size / 1024);
                        }
                    }

                    if successful_allocations == pool_size {
                        tracing::info!("🎉 Successfully created pinned buffer pool: {}/{} buffers allocated",
                                     successful_allocations, pool_size);
                    } else {
                        tracing::warn!("⚠️  Partial pool creation: {}/{} buffers allocated",
                                     successful_allocations, pool_size);
                    }

                    if successful_allocations > 0 {
                        Some(SyncPinnedBufferPool::new_direct(initial_buffers))
                    } else {
                        tracing::error!("💥 Failed to allocate any pinned buffers - pool disabled");
                        None
                    }
                }
            } else {
                tracing::debug!("🚫 Pinned buffer pool disabled by configuration");
                None
            }
        } else {
            tracing::debug!("🚫 No pool configuration provided - using fallback allocation");
            None
        };

        Self {
            nixl_agent,
            stream,
            async_rt_handle,
            pinned_buffer_pool: pool,
            cuda_event_tx,
            cuda_event_worker: Some(cuda_event_worker),
            cancel_token,
        }
    }

    fn setup_cuda_event_worker(mut cuda_event_rx: mpsc::UnboundedReceiver<(CudaEvent, oneshot::Sender<()>, DebugEventInfo)>, cancel_token: CancellationToken) -> JoinHandle<()> {
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime for CUDA event worker.");

            runtime.block_on(async move {
                loop {
                    tokio::select! {
                        Some((event, tx, debug_info)) = cuda_event_rx.recv() => {
                            let sync_start = std::time::Instant::now();
                            tracing::debug!("[CUDA_EVENT_WORKER] Starting sync for Event#{} ({}) - queued for {:.2}ms",
                                    debug_info.event_id, debug_info.transfer_direction,
                                    sync_start.duration_since(debug_info.timestamp).as_micros() as f64 / 1000.0);

                            match event.synchronize() {
                                Ok(()) => {
                                    let sync_duration = sync_start.elapsed();

                                    // Clean up device pointers using appropriate method
                                    let cleanup_msg = match (&debug_info.cleanup_info, &debug_info.cleanup_ptrs) {
                                        (Some(CleanupInfo::PinnedFree { pointers }), _) => {
                                            if let Err(_e) = cuda::cleanup_pinned_pointers(pointers.clone()) {
                                                // Fallback to direct free to prevent memory leaks
                                                for ptr in pointers {
                                                    unsafe {
                                                        let _ = cuda_result::free_host((*ptr) as *mut std::ffi::c_void);
                                                    }
                                                }
                                                format!("+ pinned cleanup failed, freed {} ptrs directly", pointers.len())
                                            } else {
                                                format!("+ freed {} pinned ptrs", pointers.len())
                                            }
                                        },
                                        (Some(CleanupInfo::DirectFree(pointers)), _) => {

                                            // Direct free for non-pool managed pointers
                                            for ptr in pointers {
                                                unsafe {
                                                    let _ = cuda_result::free_sync(*ptr);
                                                }
                                            }
                                            format!("+ freed {} ptrs directly", pointers.len())
                                        },
                                        (None, Some(cleanup_ptrs)) => {
                                            // Legacy cleanup - direct free
                                            for ptr in cleanup_ptrs {
                                                unsafe {
                                                    let _ = cuda_result::free_sync(*ptr);
                                                }
                                            }
                                            format!("+ freed {} ptrs (legacy)", cleanup_ptrs.len())
                                        },
                                        (None, None) => {
                                            "- signaling success".to_string()
                                        }
                                    };

                                    tracing::debug!("[CUDA_EVENT_WORKER] Event#{} ({}) completed in {:.2}ms {}",
                                            debug_info.event_id, debug_info.transfer_direction,
                                            sync_duration.as_micros() as f64 / 1000.0, cleanup_msg);

                                    let _ = tx.send(());  // Signal success only when kernel truly completed
                                }
                                Err(e) => {
                                    tracing::error!("CUDA event synchronization failed: {}", e);
                                    tracing::debug!("[CUDA_EVENT_WORKER] Event#{} ({}) FAILED: {} - NOT signaling completion",
                                            debug_info.event_id, debug_info.transfer_direction, e);

                                    // Clean up device pointers even on error to prevent leaks
                                    match (&debug_info.cleanup_info, &debug_info.cleanup_ptrs) {
                                        (Some(CleanupInfo::PinnedFree { pointers }), _) => {
                                            let _ = cuda::cleanup_pinned_pointers(pointers.clone());
                                            tracing::debug!("[CUDA_EVENT_WORKER] Freed {} pinned ptrs despite event failure", pointers.len());
                                        },
                                        (Some(CleanupInfo::DirectFree(pointers)), _) => {
                                            for ptr in pointers {
                                                unsafe {
                                                    let _ = cuda_result::free_sync(*ptr);
                                                }
                                            }
                                            tracing::debug!("[CUDA_EVENT_WORKER] Freed {} ptrs directly despite event failure", pointers.len());
                                        },
                                        (None, Some(cleanup_ptrs)) => {
                                            for ptr in cleanup_ptrs {
                                                unsafe {
                                                    let _ = cuda_result::free_sync(*ptr);
                                                }
                                            }
                                            tracing::debug!("[CUDA_EVENT_WORKER] Cleaned {} ptrs despite event failure (legacy)", cleanup_ptrs.len());
                                        },
                                        (None, None) => {
                                            // No cleanup needed
                                        }
                                    }

                                    // DO NOT call tx.send() - let the receiver timeout or handle the error
                                    // This prevents vLLM from thinking the operation completed successfully
                                }
                            }
                        }
                        _ = cancel_token.cancelled() => {
                            break;
                        }
                    }
                }
            });
        })
    }

    pub fn nixl_agent(&self) -> Arc<Option<NixlAgent>> {
        self.nixl_agent.clone()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn async_rt_handle(&self) -> &Handle {
        &self.async_rt_handle
    }

    pub fn cuda_event(&self, tx: oneshot::Sender<()>) -> Result<(), TransferError> {
        self.cuda_event_with_debug(tx, "UNKNOWN".to_string(), None)
    }

    pub fn acquire_resources_for_transfer_sync(
        &self,
        size: usize,
    ) -> Result<SyncPoolItem<PinnedBuffer>, TransferError> {
        let ptr_array_size = size * std::mem::size_of::<u64>();

        tracing::debug!("🎯 Acquiring pinned buffer: need {} bytes for {} addresses",
                       ptr_array_size, size);

        if let Some(pool) = &self.pinned_buffer_pool {
            tracing::debug!("🏊 Pool available - acquiring buffer (blocking with condvar)...");

            loop {
                // 🔄 Classic condvar pattern: block until available
                let buffer = pool.acquire_blocking();

                if buffer.size >= ptr_array_size {
                    tracing::debug!("✅ Acquired buffer {}: 0x{:x} ({}KB, needed {}KB)",
                                  buffer.id, buffer.ptr, buffer.size / 1024, ptr_array_size / 1024);
                    return Ok(buffer);
                } else {
                    // Return small buffer, condvar will wake us when next buffer available
                    tracing::debug!("⚠️  Buffer {} too small ({}KB needed, {}KB available), returning and retrying",
                                   buffer.id, ptr_array_size / 1024, buffer.size / 1024);
                    drop(buffer);  // Triggers notify_one() for other waiters
                    // Loop continues, acquire_blocking() will wait again via condvar
                }
            }
        } else {
            tracing::warn!("⚠️  No pinned buffer pool configured - this should not happen in production");
            // No pool configured - this is a configuration error, not a fallback case
            Err(TransferError::ExecutionError(
                "No sync pool configured - TransferContext must be created with a pool".into()
            ))
        }
    }

    pub fn calculate_buffer_size(&self, address_count: usize) -> usize {
        address_count * std::mem::size_of::<u64>()
    }

    pub fn cuda_event_with_debug(
        &self,
        tx: oneshot::Sender<()>,
        transfer_direction: String,
        worker_id: Option<u64>
    ) -> Result<(), TransferError> {
        self.cuda_event_with_cleanup(tx, transfer_direction, worker_id, None)
    }

    pub fn cuda_event_with_cleanup(
        &self,
        tx: oneshot::Sender<()>,
        transfer_direction: String,
        worker_id: Option<u64>,
        cleanup_ptrs: Option<Vec<u64>>
    ) -> Result<(), TransferError> {
        self.cuda_event_with_enhanced_cleanup(
            tx,
            transfer_direction,
            worker_id,
            cleanup_ptrs.map(CleanupInfo::DirectFree)
        )
    }

    pub fn cuda_event_with_pinned_cleanup(
        &self,
        tx: oneshot::Sender<()>,
        transfer_direction: String,
        worker_id: Option<u64>,
        pointers: Vec<u64>,
    ) -> Result<(), TransferError> {
        self.cuda_event_with_enhanced_cleanup(
            tx,
            transfer_direction,
            worker_id,
            Some(CleanupInfo::PinnedFree { pointers })
        )
    }

    fn cuda_event_with_enhanced_cleanup(
        &self,
        tx: oneshot::Sender<()>,
        transfer_direction: String,
        worker_id: Option<u64>,
        cleanup_info: Option<CleanupInfo>
    ) -> Result<(), TransferError> {
        let event_id = EVENT_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        let timestamp = std::time::Instant::now();

        let debug_info = DebugEventInfo {
            event_id,
            transfer_direction: transfer_direction.clone(),
            _worker_id: worker_id,
            timestamp,
            cleanup_ptrs: None, // Legacy field, use cleanup_info instead
            cleanup_info: cleanup_info.clone(),
        };

        let cleanup_msg = match &cleanup_info {
            Some(CleanupInfo::PinnedFree { pointers }) => {
                format!(" + pinned cleanup {} ptrs", pointers.len())
            },
            Some(CleanupInfo::DirectFree(ptrs)) => {
                format!(" + direct cleanup {} ptrs", ptrs.len())
            },
            None => String::new()
        };

        tracing::debug!("[TRANSFER_CONTEXT] Recording Event#{} for {} (worker: {:?}){}",
                event_id, transfer_direction, worker_id, cleanup_msg);

        let event = self
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(|e| TransferError::ExecutionError(e.to_string()))?;

        self.cuda_event_tx
            .send((event, tx, debug_info))
            .map_err(|_| TransferError::ExecutionError("CUDA event worker exited.".into()))?;
        Ok(())
    }
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        self.cancel_token.cancel();
        if let Some(handle) = self.cuda_event_worker.take()
            && let Err(e) = handle.join()
        {
            tracing::error!("Error joining CUDA event worker: {:?}", e);
        }
    }
}
