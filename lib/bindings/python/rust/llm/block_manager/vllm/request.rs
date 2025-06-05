// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::sync::Arc;

use dynamo_llm::block_manager::{self as bm};
use dynamo_llm::tokens::{compute_hash_v2, TokenBlockSequence, Tokens};

type DeviceStorageType = bm::storage::DeviceStorage;

/// Request Inputs
#[pyclass]
#[derive(Debug, Clone)]
pub struct KvbmRequest {
    request_id: String,
    lora_name: Option<String>,
    salt_hash: u64,
    tbs: Arc<TokenBlockSequence>,
}

#[pymethods]
impl KvbmRequest {
    #[new]
    #[pyo3(signature = (request_id, tokens, block_size, lora_name=None, salt_hash=None))]
    pub fn new(
        request_id: String,
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

        tracing::debug!("tokens: {:?}", tokens);

        // compute salt
        #[derive(Debug, serde::Serialize)]
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

        tracing::debug!("salt: {:?}", salt);

        let salt_bytes = serde_json::to_vec(&salt).unwrap();
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        tracing::debug!("salt_hash: {:?}", salt_hash);

        let sequence = Arc::new(TokenBlockSequence::new(tokens, block_size, Some(salt_hash)));

        tracing::debug!("sequence: {:?}", sequence);

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
