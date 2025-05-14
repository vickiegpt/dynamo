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

use crate::mocker::protocols::{GlobalHash, MoveBlock, UniqueBlock};
use crate::mocker::tokens::{
    compute_block_hash_for_seq, compute_seq_hash_for_blocks,
};
use std::cmp::PartialEq;
use tokio::sync::mpsc::Sender;

/// A sequence that is actively being built, with the ability to add tokens and commit to hashes
#[derive(Debug, Clone)]
pub struct ActiveSequence {
    pub global_hashes: Option<Vec<GlobalHash>>,
    pub input_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub block_size: usize,
    pub chunk_size: usize,
    pub max_output_tokens: u64,
    move_block_tx: Option<Sender<MoveBlock>>,
}

impl PartialEq for ActiveSequence {
    fn eq(&self, other: &Self) -> bool {
        self.global_hashes == other.global_hashes &&
        self.input_tokens == other.input_tokens &&
        self.output_tokens == other.output_tokens &&
        self.block_size == other.block_size &&
        self.chunk_size == other.chunk_size &&
        self.max_output_tokens == other.max_output_tokens
        // event_tx is intentionally not compared
    }
}

impl ActiveSequence {
    /// Create a new ActiveSequence instance with the provided tokens
    pub fn new(
        tokens: Vec<u32>,
        block_size: Option<usize>,
        chunk_size: Option<usize>,
        max_output_tokens: u64,
        move_block_tx: Option<Sender<MoveBlock>>,
    ) -> Self {
        let block_size = block_size.unwrap_or(64);
        let chunk_size = chunk_size.unwrap_or(256);
        
        let (global_hashes, remaining_tokens) = if tokens.len() >= block_size {
            // We have at least one complete block, process it
            let complete_blocks_len = (tokens.len() / block_size) * block_size;
            
            // Process complete blocks to get local block hashes
            let local_block_hashes = compute_block_hash_for_seq(&tokens[0..complete_blocks_len], block_size);
            
            // Compute global hashes using rolling hash
            let global_hashes = compute_seq_hash_for_blocks(&local_block_hashes, None);
            
            // Get remaining tokens that don't form a complete block
            let remaining_tokens = tokens[complete_blocks_len..].to_vec();
            
            (Some(global_hashes), remaining_tokens)
        } else {
            // Not enough tokens for a full block, just store them as input tokens
            (None, tokens)
        };
        
        // Only process blocks and send event if event_tx is provided
        if let Some(tx) = &move_block_tx {
            // Build Vec<UniqueBlock> from global_hashes
            let mut unique_blocks = Vec::new();
            if let Some(hashes) = &global_hashes {
                for &hash in hashes {
                    unique_blocks.push(UniqueBlock::HashIdentifier(hash));
                }
            }
            
            // Add a default block if there are remaining tokens
            if !remaining_tokens.is_empty() {
                unique_blocks.push(UniqueBlock::default());
            }
            
            // Send Use event if we have blocks
            if !unique_blocks.is_empty() {
                let _ = tx.try_send(MoveBlock::Use(unique_blocks, None));
            }
        }
        
        Self {
            global_hashes,
            input_tokens: remaining_tokens,
            output_tokens: Vec::new(),
            block_size,
            chunk_size,
            max_output_tokens,
            move_block_tx,
        }
    }
    
    /// Push a token to the output tokens sequence
    pub fn push(&mut self, token: u32) {
        self.output_tokens.push(token);
        
        // Check if we have a complete block
        let total_tokens = self.input_tokens.len() + self.output_tokens.len();
        if total_tokens >= self.block_size {
            // Concatenate input and output tokens
            let mut combined_tokens = Vec::with_capacity(total_tokens);
            combined_tokens.extend_from_slice(&self.input_tokens);
            combined_tokens.extend_from_slice(&self.output_tokens);
            
            // Compute local block hash for the combined tokens
            let local_hash = compute_block_hash_for_seq(&combined_tokens[0..self.block_size], self.block_size);
            
            // Get the last global hash as parent or None if there are no global hashes yet
            let parent_hash = match &self.global_hashes {
                Some(hashes) if !hashes.is_empty() => Some(hashes[hashes.len() - 1]),
                _ => None,
            };
            
            // Compute new global hash by rolling the local hash with the parent
            let new_global_hash = compute_seq_hash_for_blocks(&local_hash, parent_hash);
            
            // Append the new global hash
            if let Some(global_hashes) = &mut self.global_hashes {
                global_hashes.extend(new_global_hash);
            } else {
                self.global_hashes = Some(new_global_hash);
            }
            
            // Clear both token buffers
            self.input_tokens = Vec::new();
            self.output_tokens = Vec::new();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_sequence_push() {
        // Create a sequence with block size 16 initialized with tokens [0..15]
        let initial_tokens: Vec<u32> = (0..15).collect();
        let mut seq1 = ActiveSequence::new(
            initial_tokens,
            Some(16),
            Some(256),
            100,
            None,
        );

        // Push tokens 15 and 16
        seq1.push(15);
        seq1.push(16);

        // Verify state after pushing tokens
        assert_eq!(seq1.global_hashes.as_ref().unwrap().len(), 1);
        assert_eq!(seq1.input_tokens.len(), 0);
        assert_eq!(seq1.output_tokens.len(), 1);

        // Create another sequence with block size 16 initialized with tokens [0..17]
        let extended_tokens: Vec<u32> = (0..17).collect();
        let seq2 = ActiveSequence::new(
            extended_tokens,
            Some(16),
            Some(256),
            100,
            None,
        );

        // Assert that the global hashes contain the same data
        assert!(seq1.global_hashes.is_some() && seq2.global_hashes.is_some());
        assert_eq!(seq1.global_hashes.as_ref().unwrap(), seq2.global_hashes.as_ref().unwrap());
    }
}
