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
                                            if let Err(_e) = cuda::cleanup_pinned_pointers(pointers.clone()) {
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
            stream,
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
