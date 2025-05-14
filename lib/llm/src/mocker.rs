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

use crate::mocker::evictor::LRUEvictor;
use crate::mocker::protocols::{DirectRequest, MoveBlock, UniqueBlock};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::Instant;

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

        // Spawn the event loop
        let event_loop_handle = tokio::spawn(async move {
            let mut event_rx = event_rx;

            while let Some(event) = event_rx.recv().await {
                // Process event based on block_type
                match event {
                    MoveBlock::Use(hash) => {
                        // Always lock active first, then inactive to maintain consistent order
                        let mut active = active_blocks_clone.lock().await;
                        
                        // First check if it already exists in active blocks
                        if let Some(ref_count) = active.get_mut(&hash) {
                            // Block already active, just increment reference count
                            *ref_count += 1;
                            continue;
                        }
                        
                        // Now get inactive lock
                        let mut inactive = inactive_blocks_clone.lock().await;

                        // Then check if it exists in inactive and move it to active if found
                        if inactive.remove(&hash) {
                            // Insert into active with reference count 1
                            active.insert(hash, 1);
                            continue;
                        }

                        // Get counts for capacity check (now we already have both locks)
                        let active_count = active.len();
                        let inactive_count = inactive.num_objects();

                        // If at max capacity, evict the oldest entry from inactive blocks
                        if active_count + inactive_count >= max_capacity {
                            inactive.evict();
                        }
                        
                        // Now insert the new block in active blocks with reference count 1
                        active.insert(hash, 1);
                    }
                    MoveBlock::Destroy(hash) => {
                        let mut active = active_blocks_clone.lock().await;
                        active.remove(&hash);
                    }
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
                                inactive.insert(hash, start_time.elapsed().as_secs_f64());
                            }
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
    fn test_block_lifecycle_stringent() {
        // Create a runtime for async testing
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            // Create a MockWorker with 7 blocks capacity
            let worker = MockWorker::new(10, 16);
            let event_tx = worker.get_event_sender();

            // Helper function to use a block
            async fn use_block(event_tx: &mpsc::Sender<MoveBlock>, id: u64) {
                let block = UniqueBlock::HashIdentifier(id);
                event_tx.send(MoveBlock::Use(block)).await.unwrap();
            }

            // Helper function to destroy a block
            async fn destroy_block(event_tx: &mpsc::Sender<MoveBlock>, id: u64) {
                let block = UniqueBlock::HashIdentifier(id);
                event_tx.send(MoveBlock::Destroy(block)).await.unwrap();
            }

            // Helper function to deref a block
            async fn deref_block(event_tx: &mpsc::Sender<MoveBlock>, id: u64) {
                let block = UniqueBlock::HashIdentifier(id);
                event_tx.send(MoveBlock::Deref(block)).await.unwrap();
            }

            // Helper function to check if active blocks contain expected blocks with expected ref counts
            async fn assert_active_blocks(worker: &MockWorker, expected_blocks: &[(u64, usize)]) {
                let active_blocks_lock = worker.active_blocks.lock().await;
                
                assert_eq!(active_blocks_lock.len(), expected_blocks.len(), 
                           "Active blocks count doesn't match expected");
                
                for &(id, ref_count) in expected_blocks {
                    let block = UniqueBlock::HashIdentifier(id);
                    assert!(active_blocks_lock.contains_key(&block), 
                           "Block {} not found in active blocks", id);
                    assert_eq!(active_blocks_lock.get(&block), Some(&ref_count), 
                              "Block {} has wrong reference count", id);
                }
            }

            // Helper function to check if inactive blocks contain expected blocks
            async fn assert_inactive_blocks(worker: &MockWorker, expected_size: usize, expected_blocks: &[u64]) {
                let inactive_blocks = worker.get_inactive_blocks().await;
                let inactive_blocks_count = worker.inactive_blocks.lock().await.num_objects();
                
                assert_eq!(inactive_blocks_count, expected_size, 
                           "Inactive blocks count doesn't match expected");
                
                for &id in expected_blocks {
                    let block = UniqueBlock::HashIdentifier(id);
                    assert!(inactive_blocks.contains(&block), 
                           "Block {} not found in inactive blocks", id);
                }
            }

            // Helper function to process events
            async fn process_events() {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }

            // First use blocks 0, 1, 2, 3, 4
            for i in 0..5 {
                use_block(&event_tx, i).await;
            }
            process_events().await;

            // Then use blocks 0, 1, 5, 6
            use_block(&event_tx, 0).await;
            use_block(&event_tx, 1).await;
            use_block(&event_tx, 5).await;
            use_block(&event_tx, 6).await;
            process_events().await;

            // Check that the blocks 0 and 1 are in active blocks, both with reference counts of 2
            assert_active_blocks(&worker, &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]).await;

            // Now destroy block 4
            destroy_block(&event_tx, 4).await;
            process_events().await;

            // And deref blocks 3, 2, 1, 0 in this order
            deref_block(&event_tx, 3).await;
            deref_block(&event_tx, 2).await;
            deref_block(&event_tx, 1).await;
            deref_block(&event_tx, 0).await;
            process_events().await;

            // Check that the inactive_blocks is size 2 (via num_objects) and contains 3 and 2
            assert_inactive_blocks(&worker, 2, &[3, 2]).await;
            assert_active_blocks(&worker, &[(0, 1), (1, 1), (5, 1), (6, 1)]).await;

            // Now destroy block 6
            destroy_block(&event_tx, 6).await;
            process_events().await;

            // And deref blocks 5, 1, 0 in this order
            deref_block(&event_tx, 5).await;
            deref_block(&event_tx, 1).await;
            deref_block(&event_tx, 0).await;
            process_events().await;

            // Check that the inactive_blocks is size 5, and contains 0, 1, 2, 3, 5
            assert_inactive_blocks(&worker, 5, &[0, 1, 2, 3, 5]).await;
            assert_active_blocks(&worker, &[]).await;

            // Now use 0, 1, 2, 7, 8, 9
            for i in [0, 1, 2, 7, 8, 9] {
                use_block(&event_tx, i).await;
            }
            process_events().await;

            // Check that the inactive_blocks is size 2, and contains 3 and 5
            assert_inactive_blocks(&worker, 2, &[3, 5]).await;
            assert_active_blocks(&worker, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]).await;

            // Now use blocks 10, 11, 12
            for i in [10, 11, 12] {
                use_block(&event_tx, i).await;
            }
            process_events().await;

            // Check that the inactive_blocks is size 1 and contains only 5
            assert_inactive_blocks(&worker, 1, &[5]).await;
        });
    }
}
