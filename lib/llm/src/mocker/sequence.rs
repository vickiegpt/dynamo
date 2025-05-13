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

use std::collections::HashSet;
use crate::mocker::protocols::{GlobalHash, LocalBlockHash};
use crate::mocker::tokens::{compute_seq_hash_for_blocks, compute_block_hash_for_seq};

/// A sequence that is actively being built, with the ability to add tokens and commit to hashes
#[derive(PartialEq, Debug)]
pub struct ActiveSequence {
    pub parent_cached: Option<GlobalHash>,
    pub new_input_blocks: Vec<LocalBlockHash>,
    pub new_input_tokens: Vec<u32>,
    pub new_output_tokens: Vec<u32>,
    pub block_size: usize,
    pub worker_id: usize,
}

impl ActiveSequence {
    /// Create a new ActiveSequence instance with the provided parent, input blocks, input tokens, and block size
    pub fn new(
        parent_cached: Option<GlobalHash>,
        new_input_blocks: Vec<LocalBlockHash>,
        new_input_tokens: Vec<u32>,
        block_size: usize,
        worker_id: usize
    ) -> Self {
        Self {
            parent_cached,
            new_input_blocks,
            new_input_tokens,
            new_output_tokens: Vec::new(),
            block_size,
            worker_id,
        }
    }

    // Helper function to process tokens and split into block hashes and remaining tokens
    fn process_token_blocks(tokens: &[u32], block_size: usize) -> (Vec<LocalBlockHash>, Vec<u32>) {
        if tokens.is_empty() {
            return (Vec::new(), Vec::new());
        }
        
        // Calculate how many complete blocks we can make
        let complete_blocks_tokens = (tokens.len() / block_size) * block_size;
        
        if complete_blocks_tokens == 0 {
            // No complete blocks, just return all tokens
            return (Vec::new(), tokens.to_vec());
        }
        
        // Get the tokens for complete blocks
        let tokens_to_process = &tokens[0..complete_blocks_tokens];
        
        // Compute block hashes from the complete blocks
        let block_hashes = compute_block_hash_for_seq(tokens_to_process, block_size);
        
        // Store the remaining tokens
        let remaining_tokens = tokens[complete_blocks_tokens..].to_vec();
        
        (block_hashes, remaining_tokens)
    }

    /// Create a new ActiveSequence from tokens, finding the longest prefix that exists in the cache
    /// by performing a rolling hash over the tokens. The parent is set to the last matching hash.
    pub fn new_from_tokens_with_cache(
        cache: &HashSet<GlobalHash>,
        tokens: Vec<u32>,
        block_size: usize,
        worker_id: usize
    ) -> Self {
        // Initialize with no parent hash
        let mut parent_hash: Option<GlobalHash> = None;
        let mut processed_blocks = 0;
        
        // Process tokens in block-sized chunks
        let num_complete_blocks = tokens.len() / block_size;
        
        for block_idx in 0..num_complete_blocks {
            let start = block_idx * block_size;
            let end = start + block_size;
            let block_tokens = &tokens[start..end];
            
            // Compute the local hash for this block
            let block_hash = compute_block_hash_for_seq(block_tokens, block_size);
            assert_eq!(block_hash.len(), 1);
            
            // Compute the sequence hash using the current parent
            let seq_blocks = compute_seq_hash_for_blocks(&block_hash, parent_hash);
            assert_eq!(seq_blocks.len(), 1);
            
            let sequence_hash = seq_blocks[0];
            
            // Check if this hash exists in the cache
            if cache.contains(&sequence_hash) {
                // The hash exists, so update the parent and continue
                parent_hash = Some(sequence_hash);
                processed_blocks += 1;
            } else {
                // Hash not found, stop processing
                break;
            }
        }
        
        // Calculate how many tokens we've processed
        let processed_tokens = processed_blocks * block_size;
        
        // Process the remaining tokens
        let remaining_tokens = &tokens[processed_tokens..];
        let (new_input_blocks, new_input_tokens) = Self::process_token_blocks(remaining_tokens, block_size);
        
        Self {
            parent_cached: parent_hash,
            new_input_blocks,
            new_input_tokens,
            new_output_tokens: Vec::new(),
            block_size,
            worker_id,
        }
    }

    /// Push a token to the output tokens sequence
    /// Returns true if this push would leave new_output_tokens.len() % block_size == 1
    pub fn push(&mut self, token: u32) -> bool {
        self.new_output_tokens.push(token);
        let total_tokens = self.new_input_tokens.len() + self.new_output_tokens.len();
        total_tokens % self.block_size == 1
    }

    /// Commit the sequence to block hashes
    pub fn commit(&self) -> Vec<GlobalHash> {
        // Concatenate input and output tokens
        let mut combined_tokens = Vec::with_capacity(self.new_input_tokens.len() + self.new_output_tokens.len());
        combined_tokens.extend_from_slice(&self.new_input_tokens);
        combined_tokens.extend_from_slice(&self.new_output_tokens);
        
        // Process the combined tokens to get block hashes
        let (local_hashes_from_tokens, _) = Self::process_token_blocks(&combined_tokens, self.block_size);
        
        // Combine the new_input_blocks with the computed local hashes
        let mut combined_hashes = Vec::with_capacity(self.new_input_blocks.len() + local_hashes_from_tokens.len());
        combined_hashes.extend_from_slice(&self.new_input_blocks);
        combined_hashes.extend_from_slice(&local_hashes_from_tokens);
        
        // Convert to sequence block hashes with the parent
        compute_seq_hash_for_blocks(&combined_hashes, self.parent_cached)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_sequence_commit_with_different_token_counts() {
        // Create an ActiveSequence with specified values
        let parent = Some(42);
        let input_blocks = vec![50, 50];
        let input_tokens = vec![1, 2, 3];
        let block_size = 16;
        let worker_id = 0;
        
        let mut sequence = ActiveSequence::new(
            parent,
            input_blocks,
            input_tokens,
            block_size,
            worker_id
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

    #[test]
    fn test_sequence_creation_methods_equivalence() {
        // Let's derive PartialEq for GlobalHash if needed
        // (not needed since it already has PartialEq derived)

        // Create tokens from 0 to 16
        let tokens: Vec<u32> = (0..17).collect();
        let block_size = 16;
        let worker_id = 0;
        
        // Create first sequence using new method
        let mut sequence1 = ActiveSequence::new(
            None, // No parent cached
            Vec::new(), // No input blocks
            Vec::new(), // No input tokens
            block_size,
            worker_id
        );
        
        // Push all tokens to sequence1
        for token in &tokens {
            sequence1.push(*token);
        }
        
        // Commit the first sequence
        let committed1 = sequence1.commit();
        
        // Extract the parent hash from the committed sequence
        let parent_hash = committed1.last().map(|block| block);
        
        // Create a dummy cache with the hash from the first sequence
        let mut cache = HashSet::new();
        if let Some(hash) = parent_hash {
            cache.insert(*hash); // Dereference the hash to store the value, not the reference
        }
        
        // Create a second sequence using new_from_tokens_with_cache
        let sequence2 = ActiveSequence::new_from_tokens_with_cache(&cache, tokens, block_size, worker_id);
        
        // Commit the second sequence
        let committed2 = sequence2.commit();
        
        // Compare the lengths of committed sequences
        assert_eq!(committed1.len(), 1, "committed1 should have length 1");
        assert_eq!(committed2.len(), 0, "committed2 should have length 0");
    }
}