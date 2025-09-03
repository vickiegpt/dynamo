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
use super::cuda::{self, get_memory_tracker};

use cudarc::driver::{CudaEvent, CudaStream, sys::CUevent_flags, result as cuda_result};
use nixl_sys::Agent as NixlAgent;

use std::sync::Arc;
use std::thread::JoinHandle;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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

pub struct TransferContext {
    nixl_agent: Arc<Option<NixlAgent>>,
    stream: Arc<CudaStream>, // Primary stream for backward compatibility
    // Should have pools, RAII objects.
    h2d_stream_pool: Arc<Vec<Arc<CudaStream>>>, // Pool of H2D streams (immutable)
    h2d_stream_counter: Arc<std::sync::atomic::AtomicUsize>, // H2D round-robin counter
    d2h_stream_pool: Arc<Vec<Arc<CudaStream>>>, // Pool of D2H streams (immutable)
    d2h_stream_counter: Arc<std::sync::atomic::AtomicUsize>, // D2H round-robin counter
    async_rt_handle: Handle,

    cuda_event_tx: mpsc::UnboundedSender<(CudaEvent, oneshot::Sender<()>, DebugEventInfo)>,
    cuda_event_worker: Option<JoinHandle<()>>,
    cancel_token: CancellationToken,
}

impl TransferContext {
    pub fn new(
        nixl_agent: Arc<Option<NixlAgent>>,
        stream: Arc<CudaStream>,
        async_rt_handle: Handle,
    ) -> Self {
        Self::new_with_separate_pools(nixl_agent, stream, async_rt_handle, 10, 20) // 10 H2D + 20 D2H streams
    }

    pub fn new_with_separate_pools(
        nixl_agent: Arc<Option<NixlAgent>>,
        primary_stream: Arc<CudaStream>,
        async_rt_handle: Handle,
        h2d_pool_size: usize,
        d2h_pool_size: usize,
    ) -> Self {
        // Create separate H2D and D2H stream pools from the same CUDA context
        let mut h2d_stream_pool = Vec::new();
        for i in 0..h2d_pool_size {
            match primary_stream.context().new_stream() {
                Ok(stream) => {
                    h2d_stream_pool.push(stream);
                    tracing::debug!("Created H2D CUDA stream #{} for parallel execution", i);
                },
                Err(e) => {
                    tracing::warn!("Failed to create H2D CUDA stream #{}: {}", i, e);
                    break; // Use fewer streams if creation fails
                }
            }
        }

        let mut d2h_stream_pool = Vec::new();
        for i in 0..d2h_pool_size {
            match primary_stream.context().new_stream() {
                Ok(stream) => {
                    d2h_stream_pool.push(stream);
                    tracing::debug!("Created D2H CUDA stream #{} for parallel execution", i);
                },
                Err(e) => {
                    tracing::warn!("Failed to create D2H CUDA stream #{}: {}", i, e);
                    break; // Use fewer streams if creation fails
                }
            }
        }

        tracing::debug!("Initialized TransferContext with {} H2D streams and {} D2H streams",
            h2d_stream_pool.len(), d2h_stream_pool.len());

        let (cuda_event_tx, mut cuda_event_rx) =
            mpsc::unbounded_channel::<(CudaEvent, oneshot::Sender<()>, DebugEventInfo)>();

        let cancel_token = CancellationToken::new();

        let cancel_token_clone = cancel_token.clone();
        let cuda_event_worker = std::thread::spawn(move || {
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
                                            // Clean up pinned memory pointers
                                            tracing::debug!("🔧 PINNED CLEANUP: Freeing {} pinned memory pointers", pointers.len());

                                            if let Err(e) = cuda::cleanup_pinned_pointers(pointers.clone()) {
                                                tracing::error!("🚨 PINNED CLEANUP FAILED: {}", e);
                                                // Fallback to direct free to prevent memory leaks
                                                for &ptr in pointers {
                                                    unsafe {
                                                        let _ = cuda_result::free_host(ptr as *mut std::ffi::c_void);
                                                    }
                                                }
                                                format!("+ pinned cleanup failed, freed {} ptrs directly", pointers.len())
                                            } else {
                                                format!("+ freed {} pinned ptrs", pointers.len())
                                            }
                                        },
                                        (Some(CleanupInfo::DirectFree(pointers)), _) => {
                                            // 🔍 MANUAL MEMCHECK: Validate all pointers before direct free
                                            for &ptr in pointers.iter() {
                                                if ptr == 0 {
                                                    tracing::error!("🚨 NULL POINTER IN DIRECT FREE: Event#{} ptr=0x{:x}",
                                                        debug_info.event_id, ptr);
                                                    panic!("Null pointer in direct free - stopping for inspection");
                                                }

                                                // Validate with memory tracker
                                                if let Ok(tracker) = get_memory_tracker().lock() {
                                                    if let Err(e) = tracker.validate_pointer(ptr) {
                                                        tracing::error!("🚨 INVALID POINTER IN DIRECT FREE: {}", e);
                                                        panic!("Invalid pointer in direct free: {}", e);
                                                    }
                                                }
                                            }

                                            // Direct free for non-pool managed pointers
                                            for &ptr in pointers {
                                                unsafe {
                                                    let _ = cuda_result::free_sync(ptr);
                                                }
                                            }
                                            format!("+ freed {} ptrs directly", pointers.len())
                                        },
                                        (None, Some(cleanup_ptrs)) => {
                                            // Legacy cleanup - direct free
                                            for &ptr in cleanup_ptrs {
                                                unsafe {
                                                    let _ = cuda_result::free_sync(ptr);
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
                                            for &ptr in pointers {
                                                unsafe {
                                                    let _ = cuda_result::free_sync(ptr);
                                                }
                                            }
                                            tracing::debug!("[CUDA_EVENT_WORKER] Freed {} ptrs directly despite event failure", pointers.len());
                                        },
                                        (None, Some(cleanup_ptrs)) => {
                                            for &ptr in cleanup_ptrs {
                                                unsafe {
                                                    let _ = cuda_result::free_sync(ptr);
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
                        _ = cancel_token_clone.cancelled() => {
                            break;
                        }
                    }
                }
            });
        });

        Self {
            nixl_agent,
            stream: primary_stream,
            h2d_stream_pool: Arc::new(h2d_stream_pool), // No more Mutex wrapper
            h2d_stream_counter: Arc::new(AtomicUsize::new(0)),
            d2h_stream_pool: Arc::new(d2h_stream_pool), // No more Mutex wrapper
            d2h_stream_counter: Arc::new(AtomicUsize::new(0)),
            async_rt_handle,
            cuda_event_tx,
            cuda_event_worker: Some(cuda_event_worker),
            cancel_token,
        }
    }

    pub fn nixl_agent(&self) -> Arc<Option<NixlAgent>> {
        self.nixl_agent.clone()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Get next available H2D stream using round-robin selection for load balancing
    pub fn next_h2d_stream(&self) -> Arc<CudaStream> {
        // No locks needed - direct access to immutable stream pool!
        if self.h2d_stream_pool.is_empty() {
            // Fall back to primary stream if no H2D pool streams available
            self.stream.clone()
        } else {
            // Round-robin selection across H2D pool + primary stream
            let total_streams = self.h2d_stream_pool.len() + 1; // +1 for primary stream
            let index = self.h2d_stream_counter.fetch_add(1, Ordering::Relaxed) % total_streams;

            if index == 0 {
                // Use primary stream
                self.stream.clone()
            } else {
                // Use H2D pool stream
                self.h2d_stream_pool[index - 1].clone()
            }
        }
    }

    /// Get next available D2H stream using round-robin selection for load balancing
    /// Uses only D2H pool streams (no primary stream fallback)
    pub fn next_d2h_stream(&self) -> Arc<CudaStream> {
        // No locks needed - direct access to immutable stream pool!
        if self.d2h_stream_pool.is_empty() {
            // Fall back to primary stream if no D2H pool streams available
            self.stream.clone()
        } else {
            // Round-robin selection across D2H pool streams only (no primary)
            let total_streams = self.d2h_stream_pool.len();
            let index = self.d2h_stream_counter.fetch_add(1, Ordering::Relaxed) % total_streams;

            // Use D2H pool stream directly
            self.d2h_stream_pool[index].clone()
        }
    }

    /// Get a specific D2H stream by index (0 = first D2H pool stream, 1+ = subsequent D2H pool streams)
    /// Used by auto-stream logic for D2H/D2D transfers
    pub fn stream_by_index(&self, index: usize) -> Arc<CudaStream> {
        // No locks needed - direct access to immutable stream pool!
        if index < self.d2h_stream_pool.len() {
            self.d2h_stream_pool[index].clone()
        } else {
            // Fall back to primary if index out of bounds
            self.stream.clone()
        }
    }

    /// Get total number of available D2H streams (D2H pool only, no primary)
    /// Used by auto-stream logic for D2H/D2D transfers
    pub fn stream_count(&self) -> usize {
        // No locks needed - direct access to immutable stream pool!
        self.d2h_stream_pool.len() // No +1, only pool streams
    }

    pub fn async_rt_handle(&self) -> &Handle {
        &self.async_rt_handle
    }

    pub fn cuda_event(&self, tx: oneshot::Sender<()>) -> Result<(), TransferError> {
        self.cuda_event_with_debug(tx, "UNKNOWN".to_string(), None)
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
