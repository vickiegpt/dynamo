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
use indexmap::IndexMap;
use crate::kv_router::protocols::{DirectRequest, ExternalSequenceBlockHash};
use crate::kv_router::indexer::RadixTree;

/// Mock implementation of workers for testing and simulation
pub struct MockWorkers {
    pub num_workers: usize,
    pub max_capacity: usize,
    pub block_size: usize,
    pub active_blocks: Vec<HashMap<ExternalSequenceBlockHash, usize>>,
    pub inactive_blocks: Vec<IndexMap<ExternalSequenceBlockHash, usize>>,
    pub waiting: Vec<VecDeque<DirectRequest>>,
    pub radix_tree: RadixTree,
}

impl MockWorkers {
    /// Create a new MockWorkers instance
    pub fn new(num_workers: usize, max_capacity: usize, block_size: usize) -> Self {
        let mut active_blocks = Vec::with_capacity(num_workers);
        let mut inactive_blocks = Vec::with_capacity(num_workers);
        let mut waiting = Vec::with_capacity(num_workers);
        
        for _ in 0..num_workers {
            active_blocks.push(HashMap::new());
            
            // Initialize each IndexMap with max_capacity entries mapping 0:0, 1:1, 2:2, etc.
            let mut index_map = IndexMap::with_capacity(max_capacity);
            for i in 0..max_capacity {
                index_map.insert(ExternalSequenceBlockHash(i as u64), i);
            }
            inactive_blocks.push(index_map);
            
            waiting.push(VecDeque::new());
        }
        
        MockWorkers {
            num_workers,
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
            waiting,
            radix_tree: RadixTree::new(),
        }
    }

    /// Receive a DirectRequest and store it in the waiting queue for the specified worker
    pub fn receive_request(&mut self, request: DirectRequest) {
        let worker_idx = (request.worker_id as usize) % self.num_workers;
        self.waiting[worker_idx].push_back(request);
    }
}