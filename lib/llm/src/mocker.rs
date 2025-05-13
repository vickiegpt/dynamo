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
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use crate::mocker::protocols::{DirectRequest, UniqueBlock, MoveBlock, MoveBlockType};
use crate::mocker::evictor::LRUEvictor;

pub mod evictor;
pub mod protocols;
pub mod sequence;
pub mod tokens;

/// Mock implementation of worker for testing and simulation
pub struct MockWorkers {
    pub max_capacity: usize,
    pub block_size: usize,
    pub active_blocks: Arc<Mutex<HashMap<UniqueBlock, usize>>>,
    pub inactive_blocks: Arc<Mutex<LRUEvictor<UniqueBlock>>>,
    pub waiting: Arc<Mutex<VecDeque<DirectRequest>>>,
    pub event_tx: mpsc::Sender<MoveBlock>,
    event_loop_handle: JoinHandle<()>,
}

impl MockWorkers {
    /// Create a new MockWorkers instance
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        let active_blocks = Arc::new(Mutex::new(HashMap::new()));
        
        // Initialize LRUEvictor
        let mut evictor = LRUEvictor::new();
        // Get a single timestamp for all initial blocks
        let initial_timestamp = Instant::now().elapsed().as_secs_f64();
        // Populate with default blocks
        for _ in 0..max_capacity {
            let block = UniqueBlock::default();
            evictor.insert(block, initial_timestamp); // Use the same timestamp for all initial blocks
        }
        let inactive_blocks = Arc::new(Mutex::new(evictor));
        
        let waiting = Arc::new(Mutex::new(VecDeque::new()));
        
        // Create mpsc channel for MoveBlock events
        let (event_tx, event_rx) = mpsc::channel::<MoveBlock>(100);
        
        // Clone Arc references for the event loop
        let active_blocks_clone = active_blocks.clone();
        let inactive_blocks_clone = inactive_blocks.clone();
        
        // Spawn the event loop
        let event_loop_handle = tokio::spawn(async move {
            let mut event_rx = event_rx;
            
            while let Some(event) = event_rx.recv().await {
                // Process event based on block_type
                match event.block_type {
                    MoveBlockType::Reuse(hash) => {
                        let mut inactive = inactive_blocks_clone.lock().await;
                        
                        // Remove from inactive if it exists
                        if inactive.contains(&hash) {
                            if let Ok(_) = inactive.remove(&hash) {
                                // Now lock active and add with reference count 1
                                drop(inactive);
                                let mut active = active_blocks_clone.lock().await;
                                active.insert(hash, 1);
                            }
                        }
                    },
                    MoveBlockType::Free(hash) => {
                        let mut active = active_blocks_clone.lock().await;
                        
                        // Remove from active if it exists
                        if let Some(_) = active.remove(&hash) {
                            // Now lock inactive and add
                            drop(active);
                            let mut inactive = inactive_blocks_clone.lock().await;
                            // Use monotonic time instead of system time
                            inactive.insert(hash, Instant::now().elapsed().as_secs_f64());
                        }
                    },
                    MoveBlockType::Destroy(hash) => {
                        let mut active = active_blocks_clone.lock().await;
                        
                        // Remove from active completely
                        active.remove(&hash);
                    },
                    MoveBlockType::Ref(hash) => {
                        let mut active = active_blocks_clone.lock().await;
                        
                        // Increment reference count if it exists in active
                        if let Some(ref_count) = active.get_mut(&hash) {
                            *ref_count += 1;
                        }
                    },
                    MoveBlockType::Unref(hash) => {
                        let mut active = active_blocks_clone.lock().await;
                        
                        // Decrement reference count and check if we need to move to inactive
                        if let Some(ref_count) = active.get_mut(&hash) {
                            *ref_count -= 1;
                            
                            // If reference count reaches zero, remove from active and move to inactive
                            if *ref_count == 0 {
                                active.remove(&hash);
                                drop(active);
                                
                                // Get the inactive blocks
                                let mut inactive = inactive_blocks_clone.lock().await;
                                // Use monotonic time instead of system time
                                inactive.insert(hash.clone(), Instant::now().elapsed().as_secs_f64());
                            }
                        }
                    },
                    MoveBlockType::Evict(hash) => {
                        let mut inactive = inactive_blocks_clone.lock().await;
                        
                        // Evict the oldest entry using LRUEvictor
                        if let Ok(_) = inactive.evict() {
                            // Now add the specified hash to active blocks with reference count 1
                            drop(inactive);
                            let mut active = active_blocks_clone.lock().await;
                            active.insert(hash, 1);
                        }
                    }
                }
            }
        });
        
        MockWorkers {
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
            waiting,
            event_tx,
            event_loop_handle,
        }
    }

    /// Receive a DirectRequest and store it in the waiting queue
    pub async fn receive_request(&mut self, request: DirectRequest) {
        let mut waiting = self.waiting.lock().await;
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