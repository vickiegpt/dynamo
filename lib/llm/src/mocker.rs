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
use crate::mocker::protocols::{DirectRequest, UniqueBlock, MoveBlock};
use crate::mocker::evictor::LRUEvictor;
use crate::mocker::sequence::ActiveSequence;

pub mod evictor;
pub mod protocols;
pub mod sequence;
pub mod tokens;

/// Mock implementation of worker for testing and simulation
pub struct MockWorker {
    pub max_capacity: usize,
    pub block_size: usize,
    pub active_blocks: Arc<Mutex<HashMap<UniqueBlock, usize>>>,
    pub inactive_blocks: Arc<Mutex<LRUEvictor<UniqueBlock>>>,
    pub waiting: Arc<Mutex<VecDeque<DirectRequest>>>,
    pub event_tx: mpsc::Sender<MoveBlock>,
    event_loop_handle: JoinHandle<()>,
}

impl MockWorker {
    /// Create a new MockWorkers instance
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        let active_blocks = Arc::new(Mutex::new(HashMap::new()));
        
        // Initialize LRUEvictor
        let evictor = LRUEvictor::new();
        // No initial blocks, start with empty inactive_blocks
        let inactive_blocks = Arc::new(Mutex::new(evictor));
        
        let waiting = Arc::new(Mutex::new(VecDeque::new()));
        
        // Create mpsc channel for MoveBlock events
        let (event_tx, event_rx) = mpsc::channel::<MoveBlock>(100);
        
        // Initialize start_time
        let start_time = Instant::now();
        
        // Clone Arc references for the event loop
        let active_blocks_clone = active_blocks.clone();
        let inactive_blocks_clone = inactive_blocks.clone();
        let start_time_clone = start_time.clone();
        
        // Spawn the event loop
        let event_loop_handle = tokio::spawn(async move {
            let mut event_rx = event_rx;
            
            while let Some(event) = event_rx.recv().await {
                // Process event based on block_type
                match event {
                    MoveBlock::Make(hash) => {
                        // Directly create the block in active blocks with reference count 1
                        let mut active = active_blocks_clone.lock().await;
                        active.insert(hash, 1);
                    },
                    MoveBlock::Reuse(hash) => {
                        // Always lock active first, then inactive to maintain consistent order
                        let mut active = active_blocks_clone.lock().await;
                        let mut inactive = inactive_blocks_clone.lock().await;
                        
                        // Remove from inactive if it exists
                        if inactive.remove(&hash) {
                            // Insert into active with reference count 1
                            active.insert(hash, 1);
                        }
                    },
                    MoveBlock::Destroy(hash) => {
                        let mut active = active_blocks_clone.lock().await;
                        active.remove(&hash);
                    },
                    MoveBlock::Ref(hash) => {
                        let mut active = active_blocks_clone.lock().await;
                        
                        // Increment reference count if it exists in active
                        if let Some(ref_count) = active.get_mut(&hash) {
                            *ref_count += 1;
                        }
                    },
                    MoveBlock::Deref(hash) => {
                        // Always lock active first, then inactive if needed
                        let mut active = active_blocks_clone.lock().await;
                        
                        // Decrement reference count and check if we need to move to inactive
                        if let Some(ref_count) = active.get_mut(&hash) {
                            *ref_count -= 1;
                            
                            // If reference count reaches zero, remove from active and move to inactive
                            if *ref_count == 0 {
                                active.remove(&hash);
                                
                                // Now lock inactive
                                let mut inactive = inactive_blocks_clone.lock().await;
                                // Use monotonic time instead of system time
                                inactive.insert(hash, start_time_clone.elapsed().as_secs_f64());
                            }
                        }
                    },
                    MoveBlock::Evict(hash) => {
                        // Always lock active first, then inactive to maintain consistent order
                        let mut active = active_blocks_clone.lock().await;
                        let mut inactive = inactive_blocks_clone.lock().await;
                        
                        // Evict the oldest entry using LRUEvictor
                        if let Some(_) = inactive.evict() {
                            // Add the specified hash to active blocks with reference count 1
                            active.insert(hash, 1);
                        }
                    }
                }
            }
        });
        
        MockWorker {
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
    
    /// Get the current capacity (active blocks + inactive blocks)
    pub async fn current_capacity(&self) -> usize {
        let active = self.active_blocks.lock().await.len();
        let inactive = self.inactive_blocks.lock().await.num_objects();
        active + inactive
    }
    
    /// Get the keys of inactive blocks
    pub async fn get_inactive_blocks(&self) -> Vec<UniqueBlock> {
        let inactive = self.inactive_blocks.lock().await;
        inactive.free_table.keys().cloned().collect()
    }
    
    /// Get the keys of active blocks
    pub async fn get_active_blocks(&self) -> Vec<UniqueBlock> {
        let active = self.active_blocks.lock().await;
        active.keys().cloned().collect()
    }
}

impl Drop for MockWorker {
    fn drop(&mut self) {
        // Abort the event loop task when MockWorkers is dropped
        self.event_loop_handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    #[test]
    fn test_block_lifecycle() {
        // Create a runtime for async testing
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            // Create a MockWorker with 2 blocks capacity
            let worker = MockWorker::new(2, 16);
            let event_tx = worker.get_event_sender();
            
            // Create two blocks with HashIdentifier variant
            let block0 = UniqueBlock::HashIdentifier(0);
            let block1 = UniqueBlock::HashIdentifier(1);
            
            // Step 1: Send Evict(0) - Should move a default block out and put block0 in active
            event_tx.send(MoveBlock::Make(block0.clone())).await.unwrap();
            
            // Step 2: Send Evict(1) - Should move another default block out and put block1 in active
            event_tx.send(MoveBlock::Make(block1.clone())).await.unwrap();
            
            // Small delay to ensure events are processed
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            
            // Step 3: Send Unref(1) - Should move block1 from active to inactive
            event_tx.send(MoveBlock::Deref(block1.clone())).await.unwrap();
            
            // Small delay to ensure events are processed
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            
            // Check state: active_blocks should contain only block0, inactive should contain block1
            let active_blocks = worker.get_active_blocks().await;
            let inactive_blocks = worker.get_inactive_blocks().await;
            
            assert_eq!(active_blocks.len(), 1);
            assert!(active_blocks.contains(&block0));
            
            assert_eq!(inactive_blocks.len(), 1);
            assert!(inactive_blocks.contains(&block1));
            
            // Step 4: Send Destroy(0) - Should remove block0 from active
            event_tx.send(MoveBlock::Destroy(block0.clone())).await.unwrap();
            
            // Small delay to ensure events are processed
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            
            // Check final state: active_blocks should be empty, inactive still contains block1
            let active_blocks = worker.get_active_blocks().await;
            let inactive_blocks = worker.get_inactive_blocks().await;
            
            assert_eq!(active_blocks.len(), 0);
            assert_eq!(inactive_blocks.len(), 1);
            assert!(inactive_blocks.contains(&block1));
        });
    }
}