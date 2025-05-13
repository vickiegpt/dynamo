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

use bytemuck::cast_slice;
use xxhash_rust::xxh3;

use crate::mocker::protocols::{LocalBlockHash, GlobalHash as GlobalBlockHash};

/// A vector of `LocalBlockHash` representing the computed hashes for each chunk of tokens.
pub fn compute_block_hash_for_seq(tokens: &[u32], kv_block_size: usize) -> Vec<LocalBlockHash> {
    tokens
        .chunks_exact(kv_block_size) // Split into chunks of kv_block_size elements
        .map(|chunk| {
            // Use cast_slice directly on the u32 chunk, similar to how compute_seq_hash_for_blocks works
            xxh3::xxh3_64_with_seed(cast_slice(chunk), 0)
        })
        .collect()
}

pub fn compute_seq_hash_for_blocks(
    local_hashes: &[LocalBlockHash],
    parent_hash: Option<GlobalBlockHash>,
) -> Vec<GlobalBlockHash> {
    if local_hashes.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(local_hashes.len());
    let mut current_parent = parent_hash;

    for &tokens_hash in local_hashes {
        // Compute the sequence hash based on the parent and current block hash
        let sequence_hash = match current_parent {
            Some(parent) => {
                // Rolling hash: combine parent sequence hash with current block hash
                let hash_input = [parent, tokens_hash];
                let seq_hash = xxh3::xxh3_64_with_seed(cast_slice(&hash_input), 0);
                seq_hash
            }
            None => {
                // First block with no parent: sequence hash is the block hash itself
                tokens_hash
            }
        };

        // Add the global hash to the result
        result.push(sequence_hash);

        // Update parent for next iteration
        current_parent = Some(sequence_hash);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_seq_hash_for_blocks() {
        // Create two vectors of local hashes: [1, 2, 3, 4] and [1, 2, 3]
        let long_vec = vec![
            1,
            2, 
            3,
            4,
        ];
        
        let short_vec = vec![
            1,
            2,
            3,
        ];
        
        // Compute sequence hashes for both vectors with no parent
        let long_result = compute_seq_hash_for_blocks(&long_vec, None);
        let short_result = compute_seq_hash_for_blocks(&short_vec, None);
        
        // Verify correct lengths
        assert_eq!(long_result.len(), 4);
        assert_eq!(short_result.len(), 3);
        
        // Verify that sequence hashes match up to the third position
        for i in 0..3 {
            assert_eq!(long_result[i], short_result[i]);
        }
        
        // For first blocks with no parent, the sequence hash equals the tokens hash
        assert_eq!(long_result[0], 1);
        
        // Compute sequence hashes with a parent hash
        let parent_hash = Some(1000);
        let long_result_with_parent = compute_seq_hash_for_blocks(&long_vec, parent_hash);
        let short_result_with_parent = compute_seq_hash_for_blocks(&short_vec, parent_hash);
        
        // Verify that sequence hashes still match up to the third position
        for i in 0..3 {
            assert_eq!(long_result_with_parent[i], short_result_with_parent[i]);
        }
        
        // First block with parent should have different hash than without parent
        assert_ne!(long_result_with_parent[0], long_result[0]);
    }
}