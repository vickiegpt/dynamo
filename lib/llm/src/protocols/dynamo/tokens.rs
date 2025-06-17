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

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::Validate;

mod aggregator;
mod delta;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

use super::super::{
    common::{self, SamplingOptionsProvider, StopConditionsProvider},
    openai::{
        nvext::{NvExt, NvExtProvider},
        OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
    },
    TokenIdType,
};
use dynamo_runtime::protocols::annotated::AnnotationsProvider;

/// Dynamo Token Completion Request - mirrors OpenAI completions but with token_ids
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct DynamoTokenCompletionRequest {
    /// Token IDs instead of text prompt (this is the key difference)
    pub token_ids: Vec<TokenIdType>,

    /// Model to use
    pub model: String,

    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,

    /// Sampling temperature
    pub temperature: Option<f32>,

    /// Top-p sampling
    pub top_p: Option<f32>,

    /// Number of completions to generate
    pub n: Option<i32>,

    /// Whether to stream responses
    pub stream: Option<bool>,

    /// Number of log probabilities to return
    pub logprobs: Option<u32>,

    /// Stop sequences
    pub stop: Option<Vec<String>>,

    /// Presence penalty
    pub presence_penalty: Option<f32>,

    /// Frequency penalty  
    pub frequency_penalty: Option<f32>,

    /// Number of candidates to generate server-side
    pub best_of: Option<i32>,

    /// Logit bias (token_id -> bias)
    pub logit_bias: Option<HashMap<String, i32>>,

    /// User identifier
    pub user: Option<String>,

    /// NVIDIA extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

// Reuse OpenAI completion response types - same structure, different input
pub use crate::protocols::openai::completions::{
    CompletionChoice, CompletionResponse, LogprobResult, ResponseFactory,
};

impl NvExtProvider for DynamoTokenCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        // For token-based input, we don't have a raw prompt
        None
    }
}

impl AnnotationsProvider for DynamoTokenCompletionRequest {
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

impl OpenAISamplingOptionsProvider for DynamoTokenCompletionRequest {
    fn get_temperature(&self) -> Option<f32> {
        self.temperature
    }

    fn get_top_p(&self) -> Option<f32> {
        self.top_p
    }

    fn get_frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }

    fn get_presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

impl OpenAIStopConditionsProvider for DynamoTokenCompletionRequest {
    fn get_max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    fn get_min_tokens(&self) -> Option<u32> {
        None
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        self.stop.clone()
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Convert directly to CompletionRequest - this is the key conversion
/// We create a CompletionRequest with PromptType::TokenIds instead of going through preprocessing
impl TryFrom<DynamoTokenCompletionRequest> for common::CompletionRequest {
    type Error = anyhow::Error;

    fn try_from(request: DynamoTokenCompletionRequest) -> Result<Self, Self::Error> {
        if request.token_ids.is_empty() {
            return Err(anyhow::anyhow!("token_ids cannot be empty"));
        }

        let stop_conditions = request
            .extract_stop_conditions()
            .map_err(|e| anyhow::anyhow!("Failed to extract stop conditions: {}", e))?;

        let sampling_options = request
            .extract_sampling_options()
            .map_err(|e| anyhow::anyhow!("Failed to extract sampling options: {}", e))?;

        let annotations = request.annotations(); // Get this before moving token_ids

        // Use TokenIds directly, skip all preprocessing
        let prompt = common::PromptType::TokenIds(request.token_ids);

        Ok(common::CompletionRequest {
            prompt,
            stop_conditions,
            sampling_options,
            mdc_sum: None,
            annotations,
        })
    }
}

// ResponseFactory for creating consistent responses
#[derive(Builder)]
pub struct TokenCompletionResponseFactory {
    #[builder(setter(into))]
    pub model: String,

    #[builder(default)]
    pub system_fingerprint: Option<String>,

    #[builder(default = "format!(\"dynamo-token-{}\", uuid::Uuid::new_v4())")]
    pub id: String,

    #[builder(default = "\"text_completion\".to_string()")]
    pub object: String,

    #[builder(default = "chrono::Utc::now().timestamp() as u64")]
    pub created: u64,
}

impl TokenCompletionResponseFactory {
    pub fn builder() -> TokenCompletionResponseFactoryBuilder {
        TokenCompletionResponseFactoryBuilder::default()
    }

    pub fn make_response(
        &self,
        choice: CompletionChoice,
        usage: Option<crate::protocols::openai::CompletionUsage>,
    ) -> CompletionResponse {
        CompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![choice],
            system_fingerprint: self.system_fingerprint.clone(),
            usage,
        }
    }
}
