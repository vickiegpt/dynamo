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

use std::collections::{VecDeque, HashMap};
use std::sync::Arc;
use indexmap::IndexSet;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use crate::kv_router::protocols::{DirectRequest, UniqueSequenceHash, MoveBlock};

/// Mock implementation of workers for testing and simulation
pub struct MockWorkers {
    pub num_workers: usize,
    pub max_capacity: usize,
    pub block_size: usize,
    pub active_blocks: Vec<Arc<Mutex<HashMap<UniqueSequenceHash, usize>>>>,
    pub inactive_blocks: Vec<Arc<Mutex<IndexSet<UniqueSequenceHash>>>>,
    pub waiting: Vec<Arc<Mutex<VecDeque<DirectRequest>>>>,
    pub event_tx: mpsc::Sender<MoveBlock>,
    event_loop_handle: JoinHandle<()>,
}

impl MockWorkers {
    /// Create a new MockWorkers instance
    pub fn new(num_workers: usize, max_capacity: usize, block_size: usize) -> Self {
        let mut active_blocks = Vec::with_capacity(num_workers);
        let mut inactive_blocks = Vec::with_capacity(num_workers);
        let mut waiting = Vec::with_capacity(num_workers);
        
        for _ in 0..num_workers {
            active_blocks.push(Arc::new(Mutex::new(HashMap::new())));
            
            // Initialize each IndexMap with max_capacity entries mapping 0:0, 1:1, 2:2, etc.
            let index_map = (0..max_capacity)
                .map(|_| UniqueSequenceHash::default())
                .collect::<IndexSet<_>>();
            inactive_blocks.push(Arc::new(Mutex::new(index_map)));
            
            waiting.push(Arc::new(Mutex::new(VecDeque::new())));
        }
        
        // Create mpsc channel for MoveBlock events
        let (event_tx, event_rx) = mpsc::channel::<MoveBlock>(100);
        
        // Clone Arc references for the event loop
        let active_blocks_clone = active_blocks.clone();
        let inactive_blocks_clone = inactive_blocks.clone();
        let num_workers_clone = num_workers;
        
        // Spawn the event loop
        let event_loop_handle = tokio::spawn(async move {
            let mut event_rx = event_rx;
            
            while let Some(event) = event_rx.recv().await {
                match event {
                    MoveBlock::ToActive(hash, worker_id) => {
                        if worker_id as usize >= num_workers_clone {
                            tracing::error!("Invalid worker_id {}: must be less than {}", worker_id, num_workers_clone);
                            continue;
                        }
                        let mut inactive = inactive_blocks_clone[worker_id].lock().await;
                        
                        // Remove from inactive if it exists
                        if inactive.swap_remove(&hash) {
                            // Now lock active and add with reference count 1
                            let mut active = active_blocks_clone[worker_id].lock().await;
                            active.insert(hash, 1);
                        }
                    },
                    MoveBlock::ToInactive(hash, worker_id) => {
                        if worker_id as usize >= num_workers_clone {
                            tracing::error!("Invalid worker_id {}: must be less than {}", worker_id, num_workers_clone);
                            continue;
                        }
                        let mut active = active_blocks_clone[worker_id].lock().await;
                        
                        // Remove from active if it exists
                        if let Some(_) = active.remove(&hash) {
                            // Now lock inactive and add
                            let mut inactive = inactive_blocks_clone[worker_id].lock().await;
                            inactive.insert(hash);
                        }
                    },
                    MoveBlock::Destroy(hash, worker_id) => {
                        if worker_id as usize >= num_workers_clone {
                            tracing::error!("Invalid worker_id {}: must be less than {}", worker_id, num_workers_clone);
                            continue;
                        }
                        let mut active = active_blocks_clone[worker_id].lock().await;
                        
                        // Remove from active completely
                        active.remove(&hash);
                    }
                }
            }
        });
        
        MockWorkers {
            num_workers,
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
            waiting,
            event_tx,
            event_loop_handle,
        }
    }

    /// Receive a DirectRequest and store it in the waiting queue for the specified worker
    pub async fn receive_request(&mut self, request: DirectRequest) {
        let worker_idx = (request.worker_id as usize) % self.num_workers;
        let mut waiting = self.waiting[worker_idx].lock().await;
        waiting.push_back(request);
    }
    
    /// Get a clone of the event sender to send MoveBlock events
    pub fn get_event_sender(&self) -> mpsc::Sender<MoveBlock> {
        self.event_tx.clone()
    }
}

impl Drop for MockWorkers {
    fn drop(&mut self) {
        // Abort the event loop task when MockWorkers is dropped
        self.event_loop_handle.abort();
    }
}