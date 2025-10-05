// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use uuid::Uuid;

use crate::kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use crate::tokens::blocks::UniqueBlock;
use crate::tokens::{BlockHash, SequenceHash, Token};

pub type NumBlocks = usize;

/// Represents different block movement operations in the cache
/// For Use and Promote variants, parent hash is the second field
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlock {
    Use(Vec<UniqueBlock>),
    Destroy(Vec<UniqueBlock>),
    Deref(Vec<UniqueBlock>),
    Promote(Uuid, SequenceHash, Option<u64>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlockResponse {
    Store(Vec<SequenceHash>, Option<u64>),
    Remove(Vec<SequenceHash>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectRequest {
    pub tokens: Vec<Token>,
    pub max_output_tokens: usize,
    pub uuid: Option<Uuid>,
    pub dp_rank: Option<u32>,
}

/// Represents the cost of prefilling content in the cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillCost {
    pub new_blocks: usize,
    pub new_tokens: usize,
}

impl PrefillCost {
    pub fn predict_prefill_compute(&self, new_tokens: Option<usize>) -> f64 {
        let tokens = new_tokens.unwrap_or(self.new_tokens);
        1.25e-6 * (tokens as f64).powi(2) + 7.41e-2 * (tokens as f64) + 2.62e1
    }
}

/// Signal for output token generation with completion status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSignal {
    pub uuid: Uuid,
    pub completed: bool,
}

/// Configuration arguments for MockVllmEngine
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "owned", build_fn(public))]
pub struct MockEngineArgs {
    #[builder(default = "16384")]
    pub num_gpu_blocks: usize,

    #[builder(default = "64")]
    pub block_size: usize,

    // This was 1024 in the past but reverted back to 256
    #[builder(default = Some(256))]
    pub max_num_seqs: Option<usize>,

    // default for open api server, for llm class it's 16384
    #[builder(default = Some(8192))]
    pub max_num_batched_tokens: Option<usize>,

    #[builder(default = true)]
    pub enable_prefix_caching: bool,

    #[builder(default = true)]
    pub enable_chunked_prefill: bool,

    #[builder(default = "0.01")]
    pub watermark: f64,

    #[builder(default = "1.0")]
    pub speedup_ratio: f64,

    #[builder(default = "1")]
    pub dp_size: u32,

    /// Optional startup time in seconds to simulate engine initialization delay
    #[builder(default = "None")]
    pub startup_time: Option<f64>,
}

impl Default for MockEngineArgs {
    fn default() -> MockEngineArgs {
        MockEngineArgsBuilder::default()
            .build()
            .expect("Failed to build default MockEngineArgs")
    }
}

impl MockEngineArgs {
    pub fn builder() -> MockEngineArgsBuilder {
        MockEngineArgsBuilder::default()
    }

    /// Create MockEngineArgs from a JSON file containing extra engine arguments
    pub fn from_json_file(path: &Path) -> anyhow::Result<Self> {
        let mut builder = Self::builder();

        // Load and parse the JSON file
        let file_content = std::fs::read_to_string(path)?;
        let extra_args: HashMap<String, serde_json::Value> = serde_json::from_str(&file_content)?;

        // Define valid field names
        let valid_fields: HashSet<&str> = [
            "num_gpu_blocks",
            "block_size",
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "watermark",
            "speedup_ratio",
            "dp_size",
            "startup_time",
        ]
        .iter()
        .cloned()
        .collect();

        // Check for invalid arguments
        let invalid_args: Vec<String> = extra_args
            .keys()
            .filter(|key| !valid_fields.contains(key.as_str()))
            .cloned()
            .collect();

        if !invalid_args.is_empty() {
            return Err(anyhow::anyhow!(
                "Invalid arguments found in JSON file: {}. Valid arguments are: {:?}",
                invalid_args.join(", "),
                valid_fields
            ));
        }

        // Apply each extra argument to the builder
        if let Some(value) = extra_args.get("num_gpu_blocks")
            && let Some(num) = value.as_u64()
        {
            builder = builder.num_gpu_blocks(num as usize);
        }

        if let Some(value) = extra_args.get("block_size")
            && let Some(num) = value.as_u64()
        {
            builder = builder.block_size(num as usize);
        }

        if let Some(value) = extra_args.get("max_num_seqs")
            && let Some(num) = value.as_u64()
        {
            builder = builder.max_num_seqs(Some(num as usize));
        }

        if let Some(value) = extra_args.get("max_num_batched_tokens")
            && let Some(num) = value.as_u64()
        {
            builder = builder.max_num_batched_tokens(Some(num as usize));
        }

        if let Some(value) = extra_args.get("enable_prefix_caching")
            && let Some(enabled) = value.as_bool()
        {
            builder = builder.enable_prefix_caching(enabled);
        }

        if let Some(value) = extra_args.get("enable_chunked_prefill")
            && let Some(enabled) = value.as_bool()
        {
            builder = builder.enable_chunked_prefill(enabled);
        }

        if let Some(value) = extra_args.get("watermark")
            && let Some(num) = value.as_f64()
        {
            builder = builder.watermark(num);
        }

        if let Some(value) = extra_args.get("speedup_ratio")
            && let Some(num) = value.as_f64()
        {
            builder = builder.speedup_ratio(num);
        }

        if let Some(value) = extra_args.get("dp_size")
            && let Some(num) = value.as_u64()
        {
            builder = builder.dp_size(num as u32);
        }

        if let Some(value) = extra_args.get("startup_time")
            && let Some(num) = value.as_f64()
        {
            builder = builder.startup_time(Some(num));
        }

        // Build the MockEngineArgs with either defaults or overridden values
        builder
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build MockEngineArgs: {}", e))
    }
}

/// Converts a MoveBlockResponse from the mocker backend into a KvCacheEventData.
///
/// This function assumes that the stored sequence hashes in the response always
/// correspond to the tail part of the local hashes array. This is the expected
/// behavior of KV block storage, where blocks are stored sequentially and the
/// response contains the most recent blocks that were stored.
///
/// # Panics
/// Panics if the number of blocks in the Store response exceeds the length
/// of local_hashes.
pub fn block_response_to_kv_event(
    response: MoveBlockResponse,
    local_hashes: &[BlockHash],
) -> KvCacheEventData {
    match response {
        MoveBlockResponse::Store(full_blocks, parent_hash) => {
            let num_blocks = full_blocks.len();
            let local_hashes_slice = &local_hashes[local_hashes
                .len()
                .checked_sub(num_blocks)
                .expect("local hashes fewer than block response signal")..];

            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                blocks: full_blocks
                    .into_iter()
                    .zip(local_hashes_slice.iter())
                    .map(|(global_hash, local_hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(global_hash),
                        tokens_hash: LocalBlockHash(*local_hash),
                        moe_metadata: None,
                        cxl_metadata: None,
                    })
                    .collect(),
            })
        }
        MoveBlockResponse::Remove(full_blocks) => KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: full_blocks
                .into_iter()
                .map(ExternalSequenceBlockHash)
                .collect(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_block_default_uniqueness() {
        // Create 10 default UniqueBlock instances
        let blocks: Vec<UniqueBlock> = (0..10).map(|_| UniqueBlock::default()).collect();

        // Extract UUIDs from each block
        let mut uuids = Vec::new();
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(uuid) => uuids.push(uuid),
                _ => panic!("Expected UuidIdentifier variant"),
            }
        }

        // Check that all UUIDs are unique by comparing each with every other
        for i in 0..uuids.len() {
            for j in i + 1..uuids.len() {
                assert_ne!(
                    uuids[i], uuids[j],
                    "UUID at index {} and {} are identical",
                    i, j
                );
            }
        }
    }
}
