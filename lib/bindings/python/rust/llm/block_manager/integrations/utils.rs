// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use std::sync::Arc;

use pyo3::prelude::*;

use dynamo_llm::block_manager::{self as bm, block::BlockIdentifier};
use dynamo_llm::tokens::{compute_hash_v2, TokenBlockSequence, Tokens};

use crate::to_pyerr;

type DeviceStorageType = bm::storage::DeviceStorage;

/// Request Inputs
#[pyclass]
#[derive(Debug, Clone)]
pub struct KvRequest {
    request_id: usize,
    lora_name: Option<String>,
    salt_hash: u64,
    tbs: Arc<TokenBlockSequence>,
}

impl KvRequest {
    pub fn request_id(&self) -> usize {
        self.request_id
    }

    pub fn tbs(&self) -> Arc<TokenBlockSequence> {
        self.tbs.clone()
    }
}

#[pymethods]
impl KvRequest {
    #[new]
    #[pyo3(signature = (request_id, tokens, block_size, lora_name=None, salt_hash=None))]
    fn new(
        request_id: usize,
        tokens: Vec<usize>,
        block_size: usize,
        lora_name: Option<String>,
        salt_hash: Option<String>,
    ) -> Self {
        let tokens: Tokens = tokens
            .into_iter()
            .map(|t| t as u32)
            .collect::<Vec<_>>()
            .into();

        // compute salt
        #[derive(serde::Serialize)]
        struct Salt {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<String>,
        }

        let salt = Salt {
            salt: salt_hash,
            lora_name: lora_name.clone(),
        };

        let salt_bytes = serde_json::to_vec(&salt).unwrap();
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        let sequence = Arc::new(TokenBlockSequence::new(tokens, block_size, Some(salt_hash)));

        Self {
            request_id,
            lora_name,
            salt_hash,
            tbs: sequence,
        }
    }

    pub fn sequence_hashes(&self) -> Vec<u64> {
        self.tbs
            .blocks()
            .iter()
            .map(|b| b.sequence_hash())
            .collect()
    }
}

#[derive(Debug)]
pub enum BlockListType {
    Immutable(Vec<bm::block::ImmutableBlock<DeviceStorageType, bm::BasicMetadata>>),
    Mutable(Vec<bm::block::MutableBlock<DeviceStorageType, bm::BasicMetadata>>),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct DynamoKvBlockList {
    blocks: Arc<std::sync::Mutex<BlockListType>>,
}

impl DynamoKvBlockList {
    pub fn new(blocks: BlockListType) -> Self {
        Self {
            blocks: Arc::new(std::sync::Mutex::new(blocks)),
        }
    }
}

#[pymethods]
impl DynamoKvBlockList {
    fn get_block_id(&self, block_idx: usize) -> PyResult<usize> {
        let blocks = self.blocks.lock().unwrap();
        let block_id = match &*blocks {
            BlockListType::Immutable(blocks) => blocks.get(block_idx).map(|b| b.block_id()),
            BlockListType::Mutable(blocks) => blocks.get(block_idx).map(|b| b.block_id()),
        };

        block_id.ok_or_else(|| to_pyerr("block not found"))
    }

    fn get_block_hash(&self, block_idx: usize) -> PyResult<Option<u64>> {
        let blocks = self.blocks.lock().unwrap();
        let sequence_hash = match &*blocks {
            BlockListType::Immutable(blocks) => blocks
                .get(block_idx)
                .ok_or_else(|| to_pyerr("block not found"))?
                .sequence_hash()
                .ok(),
            BlockListType::Mutable(blocks) => blocks
                .get(block_idx)
                .ok_or_else(|| to_pyerr("block not found"))?
                .sequence_hash()
                .ok(),
        };

        Ok(sequence_hash)
    }

    fn block_ids(&self) -> Vec<usize> {
        let blocks = self.blocks.lock().unwrap();
        match &*blocks {
            BlockListType::Immutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
            BlockListType::Mutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
        }
    }

    fn unhashed_block_ids(&self) -> Vec<usize> {
        let blocks = self.blocks.lock().unwrap();
        match &*blocks {
            BlockListType::Immutable(_blocks) => vec![],
            BlockListType::Mutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
        }
    }

    #[pyo3(name = "__len__")]
    fn len(&self) -> usize {
        let blocks = self.blocks.lock().unwrap();
        match &*blocks {
            BlockListType::Immutable(blocks) => blocks.len(),
            BlockListType::Mutable(blocks) => blocks.len(),
        }
    }
}
