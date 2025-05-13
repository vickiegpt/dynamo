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

use std::collections::{HashSet, VecDeque};
use indexmap::IndexSet;
use crate::kv_router::protocols::{DirectRequest, KvCacheStoredBlockData, ExternalSequenceBlockHash, LocalBlockHash};
use crate::kv_router::indexer::{compute_seq_hash_for_blocks, RadixTree, compute_block_hash_for_seq};

/// A sequence that is actively being built, with the ability to add tokens and commit to hashes
pub struct ActiveSequence {
    pub parent_cached: ExternalSequenceBlockHash,
    pub new_input_blocks: Vec<LocalBlockHash>,
    pub new_input_tokens: Vec<u32>,
    pub new_output_tokens: Vec<u32>,
    pub block_size: u64,
}

impl ActiveSequence {
    /// Create a new ActiveSequence instance with the provided parent, input blocks, input tokens, and block size
    pub fn new(
        parent_cached: ExternalSequenceBlockHash,
        new_input_blocks: Vec<LocalBlockHash>,
        new_input_tokens: Vec<u32>,
        block_size: u64
    ) -> Self {
        Self {
            parent_cached,
            new_input_blocks,
            new_input_tokens,
            new_output_tokens: Vec::new(),
            block_size,
        }
    }

    /// Push a token to the output tokens sequence
    pub fn push(&mut self, token: u32) {
        self.new_output_tokens.push(token);
    }

    /// Commit the sequence to block hashes
    pub fn commit(&self) -> Vec<KvCacheStoredBlockData> {
        // Concatenate input and output tokens
        let mut combined_tokens = Vec::with_capacity(self.new_input_tokens.len() + self.new_output_tokens.len());
        combined_tokens.extend_from_slice(&self.new_input_tokens);
        combined_tokens.extend_from_slice(&self.new_output_tokens);
        
        // Calculate how many complete blocks we can make
        let block_size = self.block_size as usize;
        let complete_blocks_tokens = (combined_tokens.len() / block_size) * block_size;
        
        // Truncate tokens to include only complete blocks
        let tokens_to_process = &combined_tokens[0..complete_blocks_tokens];
        
        // Compute the block hashes for the tokens
        let local_hashes_from_tokens = compute_block_hash_for_seq(tokens_to_process, block_size);
        
        // Combine the new_input_blocks with the computed local hashes
        let mut combined_hashes = Vec::with_capacity(self.new_input_blocks.len() + local_hashes_from_tokens.len());
        combined_hashes.extend_from_slice(&self.new_input_blocks);
        combined_hashes.extend_from_slice(&local_hashes_from_tokens);
        
        // Convert to sequence block hashes with the parent
        compute_seq_hash_for_blocks(&combined_hashes, Some(self.parent_cached))
    }
}

/// Mock implementation of workers for testing and simulation
pub struct MockWorkers {
    pub num_workers: u64,
    pub max_capacity: u64,
    pub block_size: u64,
    pub active_blocks: Vec<HashSet<KvCacheStoredBlockData>>,
    pub inactive_blocks: Vec<IndexSet<KvCacheStoredBlockData>>,
    pub waiting_blocks: Vec<VecDeque<KvCacheStoredBlockData>>,
    pub radix_tree: RadixTree,
}

impl MockWorkers {
    /// Create a new MockWorkers instance
    pub fn new(num_workers: u64, max_capacity: u64, block_size: u64) -> Self {
        let mut active_blocks = Vec::with_capacity(num_workers as usize);
        let mut inactive_blocks = Vec::with_capacity(num_workers as usize);
        let mut waiting_blocks = Vec::with_capacity(num_workers as usize);
        
        for _ in 0..num_workers {
            active_blocks.push(HashSet::new());
            inactive_blocks.push(IndexSet::new());
            waiting_blocks.push(VecDeque::new());
        }
        
        MockWorkers {
            num_workers,
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
            waiting_blocks,
            radix_tree: RadixTree::new(),
        }
    }

    /// Receive a DirectRequest and store it in the waiting queue for the specified worker
    /// Also compute and store the sequence block hashes
    pub fn receive_request(&mut self, request: DirectRequest) {
        // Get worker index from worker_id, ensuring it's within bounds
        let worker_idx = (request.worker_id as usize) % self.num_workers as usize;
        
        // Compute sequence block hashes
        let stored_blocks = compute_seq_hash_for_blocks(&request.hashes, None);
        
        // Add the block hashes to the active set for this worker
        for block in stored_blocks {
            self.waiting_blocks[worker_idx].push_back(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_sequence_commit_with_different_token_counts() {
        // Create an ActiveSequence with specified values
        let parent = ExternalSequenceBlockHash(42);
        let input_blocks = vec![LocalBlockHash(50), LocalBlockHash(51)];
        let input_tokens = vec![1, 2, 3];
        let block_size = 16;
        
        let mut sequence = ActiveSequence::new(
            parent,
            input_blocks,
            input_tokens,
            block_size
        );
        
        for i in 0..12 {
            sequence.push(i);
        }
        
        let result1 = sequence.commit();
        assert_eq!(result1.len(), 2);
        
        sequence.push(12);
        let result2 = sequence.commit();
        assert_eq!(result2.len(), 3);
        
        for i in 13..16 {
            sequence.push(i);
        }
        
        let result3 = sequence.commit();
        assert_eq!(result3.len(), 3);
    }
}