// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::ChatCompletionMessageToolCall;
use serde::{Deserialize, Serialize};

use super::FinishReason;
use crate::protocols::TokenIdType;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PostprocessedResponse {
    /// Model Deployment Card checksum
    pub mdcsum: String,

    // if the number of slots for a given request is greater than 1
    // this indicates the index of the slot for the response
    pub index: Option<usize>,

    pub finish_reason: Option<FinishReason>,

    // new token_ids
    pub token_ids: Vec<TokenIdType>,

    // tokens
    pub tokens: Option<Vec<Option<String>>>,

    // decoded text
    pub text: Option<String>,

    /// cumulative log probabilities
    pub cum_log_probs: Option<f64>,

    pub parsed_components: Option<ParsedComponents>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ParsedComponents {
    pub text: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
}
