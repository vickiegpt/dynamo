// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse};
use crate::{
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::common::{self},
    types::TokenIdType,
};
use dynamo_parsers::{ParserResult, ReasoningParser, ReasoningParserType, ReasoningParserWrapper};

/// Provides a method for generating a [`DeltaGenerator`] from a chat completion request.
impl NvCreateChatCompletionRequest {
    /// Creates a [`DeltaGenerator`] instance based on the chat completion request.
    ///
    /// # Arguments
    /// * `request_id` - The request ID to use for the chat completion response ID.
    ///
    /// # Returns
    /// * [`DeltaGenerator`] configured with model name and response options.
    pub fn response_generator(&self, request_id: String) -> DeltaGenerator {
        let options = DeltaGeneratorOptions {
            enable_usage: self
                .inner
                .stream_options
                .as_ref()
                .map(|opts| opts.include_usage)
                .unwrap_or(false),
            enable_logprobs: self.inner.logprobs.unwrap_or(false)
                || self.inner.top_logprobs.unwrap_or(0) > 0,
            runtime_config: ModelRuntimeConfig::default(),
        };

        DeltaGenerator::new(self.inner.model.clone(), options, request_id)
    }
}

/// Configuration options for the [`DeltaGenerator`], controlling response behavior.
#[derive(Debug, Clone, Default)]
pub struct DeltaGeneratorOptions {
    /// Determines whether token usage statistics should be included in the response.
    pub enable_usage: bool,
    /// Determines whether log probabilities should be included in the response.
    pub enable_logprobs: bool,

    pub runtime_config: ModelRuntimeConfig,
}

/// Generates incremental chat completion responses in a streaming fashion.
#[derive(Debug)]
pub struct DeltaGenerator {
    /// Unique identifier for the chat completion session.
    id: String,
    /// Object type, representing a streamed chat completion response.
    object: String,
    /// Timestamp (Unix epoch) when the response was created.
    created: u32,
    model: String,
    /// Optional system fingerprint for version tracking.
    system_fingerprint: Option<String>,
    /// Optional service tier information for the response.
    service_tier: Option<dynamo_async_openai::types::ServiceTierResponse>,
    /// Tracks token usage for the completion request.
    usage: dynamo_async_openai::types::CompletionUsage,
    /// Counter tracking the number of messages issued.
    msg_counter: u64,
    /// Configuration options for response generation.
    options: DeltaGeneratorOptions,

    /// Reasoning Parser object
    /// This is used to parse reasoning content in the response.
    /// None means no reasoning parsing will be performed.
    reasoning_parser: Option<ReasoningParserWrapper>,
    
    /// Counter for reasoning tokens separate from completion tokens
    reasoning_tokens: u32,
}

impl DeltaGenerator {
    /// Creates a new [`DeltaGenerator`] instance with the specified model and options.
    ///
    /// # Arguments
    /// * `model` - The model name used for response generation.
    /// * `options` - Configuration options for enabling usage and log probabilities.
    /// * `request_id` - The request ID to use for the chat completion response.
    ///
    /// # Returns
    /// * A new instance of [`DeltaGenerator`].
    pub fn new(model: String, options: DeltaGeneratorOptions, request_id: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // SAFETY: Casting from `u64` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until 2106.
        let now: u32 = now.try_into().expect("timestamp exceeds u32::MAX");

        let usage = dynamo_async_openai::types::CompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };

        // Reasoning parser type
        // If no parser is specified (None), no reasoning parsing will be performed
        let reasoning_parser = options
            .runtime_config
            .reasoning_parser
            .as_deref()
            .map(ReasoningParserType::get_reasoning_parser_from_name);

        let chatcmpl_id = format!("chatcmpl-{request_id}");

        Self {
            id: chatcmpl_id,
            object: "chat.completion.chunk".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            service_tier: None,
            usage,
            msg_counter: 0,
            options,
            reasoning_parser,
            reasoning_tokens: 0,
        }
    }

    /// Update runtime configuration and reconfigure the reasoning parser accordingly.
    pub fn set_reasoning_parser(&mut self, runtime_config: ModelRuntimeConfig) {
        self.options.runtime_config = runtime_config.clone();
        match self.options.runtime_config.reasoning_parser.as_deref() {
            Some(name) => {
                self.reasoning_parser =
                    Some(ReasoningParserType::get_reasoning_parser_from_name(name));
            }
            None => {
                self.reasoning_parser = None;
            }
        }
    }

    /// Updates the prompt token usage count.
    ///
    /// # Arguments
    /// * `isl` - The number of prompt tokens used.
    pub fn update_isl(&mut self, isl: u32) {
        self.usage.prompt_tokens = isl;
    }

    pub fn create_logprobs(
        &self,
        tokens: Vec<common::llm_backend::TokenType>,
        token_ids: &[TokenIdType],
        logprobs: Option<common::llm_backend::LogProbs>,
        top_logprobs: Option<common::llm_backend::TopLogprobs>,
    ) -> Option<dynamo_async_openai::types::ChatChoiceLogprobs> {
        if !self.options.enable_logprobs || logprobs.is_none() {
            return None;
        }

        let toks = tokens
            .into_iter()
            .zip(token_ids)
            .map(|(token, token_id)| (token.unwrap_or_default(), *token_id))
            .collect::<Vec<(String, TokenIdType)>>();
        let tok_lps = toks
            .iter()
            .zip(logprobs.unwrap())
            .map(|(_, lp)| lp as f32)
            .collect::<Vec<f32>>();

        let content = top_logprobs.map(|top_logprobs| {
            toks.iter()
                .zip(tok_lps)
                .zip(top_logprobs)
                .map(|(((t, tid), lp), top_lps)| {
                    let mut found_selected_token = false;
                    let mut converted_top_lps = top_lps
                        .iter()
                        .map(|top_lp| {
                            let top_t = top_lp.token.clone().unwrap_or_default();
                            let top_tid = top_lp.token_id;
                            found_selected_token = found_selected_token || top_tid == *tid;
                            dynamo_async_openai::types::TopLogprobs {
                                token: top_t,
                                logprob: top_lp.logprob as f32,
                                bytes: None,
                            }
                        })
                        .collect::<Vec<dynamo_async_openai::types::TopLogprobs>>();
                    if !found_selected_token {
                        // If the selected token is not in the top logprobs, add it
                        converted_top_lps.push(dynamo_async_openai::types::TopLogprobs {
                            token: t.clone(),
                            logprob: lp,
                            bytes: None,
                        });
                    }
                    dynamo_async_openai::types::ChatCompletionTokenLogprob {
                        token: t.clone(),
                        logprob: lp,
                        bytes: None,
                        top_logprobs: converted_top_lps,
                    }
                })
                .collect()
        });

        Some(dynamo_async_openai::types::ChatChoiceLogprobs {
            content,
            refusal: None,
        })
    }

    fn create_reasoning_content(
        &mut self,
        text: &Option<String>,
        token_ids: &[u32],
    ) -> Option<ParserResult> {
        // If no reasoning parser is configured, return None
        let reasoning_parser = self.reasoning_parser.as_mut()?;

        let text_ref = text.as_deref().unwrap_or("");
        if text_ref.is_empty() && token_ids.is_empty() {
            return None;
        }
        let parser_result =
            reasoning_parser.parse_reasoning_streaming_incremental(text_ref, token_ids);

        Some(parser_result)
    }

    /// Creates a choice within a chat completion response.
    ///
    /// # Arguments
    /// * `index` - The index of the choice in the completion response.
    /// * `text` - The text content for the response.
    /// * `finish_reason` - The reason why the response finished (e.g., stop, length, etc.).
    /// * `logprobs` - Optional log probabilities of the generated tokens.
    ///
    /// # Returns
    /// * An [`dynamo_async_openai::types::CreateChatCompletionStreamResponse`] instance representing the choice.
    #[allow(deprecated)]
    pub fn create_choice(
        &mut self,
        index: u32,
        text: Option<String>,
        reasoning_content: Option<String>,
        finish_reason: Option<dynamo_async_openai::types::FinishReason>,
        logprobs: Option<dynamo_async_openai::types::ChatChoiceLogprobs>,
    ) -> NvCreateChatCompletionStreamResponse {
        let delta = dynamo_async_openai::types::ChatCompletionStreamResponseDelta {
            content: text,
            function_call: None,
            tool_calls: None,
            role: if self.msg_counter == 0 {
                Some(dynamo_async_openai::types::Role::Assistant)
            } else {
                None
            },
            refusal: None,
            reasoning_content,
        };

        let choice = dynamo_async_openai::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            logprobs,
        };

        let choices = vec![choice];

        // According to OpenAI spec: when stream_options.include_usage is true,
        // all intermediate chunks should have usage: null
        // The final usage chunk will be sent separately with empty choices
        dynamo_async_openai::types::CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices,
            usage: None, // Always None for chunks with content/choices
            service_tier: self.service_tier.clone(),
        }
    }

    /// Creates a final usage-only chunk for OpenAI compliance.
    /// This should be sent after the last content chunk when stream_options.include_usage is true.
    ///
    /// # Returns
    /// * A [`CreateChatCompletionStreamResponse`] with empty choices and usage stats.
    pub fn create_usage_chunk(&self) -> NvCreateChatCompletionStreamResponse {
        let mut usage = self.usage.clone();
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);
        
        // Add reasoning tokens to total_tokens and populate completion_tokens_details
        usage.total_tokens = usage.total_tokens.saturating_add(self.reasoning_tokens);
        
        // Only set completion_tokens_details if we have reasoning tokens or if reasoning parser is enabled
        if self.reasoning_tokens > 0 || self.reasoning_parser.is_some() {
            use dynamo_async_openai::types::CompletionTokensDetails;
            usage.completion_tokens_details = Some(CompletionTokensDetails {
                accepted_prediction_tokens: None,
                audio_tokens: None,
                reasoning_tokens: Some(self.reasoning_tokens),
                rejected_prediction_tokens: None,
            });
        }

        dynamo_async_openai::types::CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            choices: vec![], // Empty choices for usage-only chunk
            usage: Some(usage),
            service_tier: self.service_tier.clone(),
        }
    }

    /// Check if usage tracking is enabled
    pub fn is_usage_enabled(&self) -> bool {
        self.options.enable_usage
    }
}

/// Implements the [`crate::protocols::openai::DeltaGeneratorExt`] trait for [`DeltaGenerator`], allowing
/// it to transform backend responses into OpenAI-style streaming responses.
impl crate::protocols::openai::DeltaGeneratorExt<NvCreateChatCompletionStreamResponse>
    for DeltaGenerator
{
    /// Converts a backend response into a structured OpenAI-style streaming response.
    ///
    /// # Arguments
    /// * `delta` - The backend response containing generated text and metadata.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionStreamResponse)` if conversion succeeds.
    /// * `Err(anyhow::Error)` if an error occurs.
    fn choice_from_postprocessor(
        &mut self,
        delta: crate::protocols::common::llm_backend::BackendOutput,
    ) -> anyhow::Result<NvCreateChatCompletionStreamResponse> {
        // Calculate token length once for reuse
        let token_length: u32 = if self.options.enable_usage {
            // SAFETY: Casting from `usize` to `u32` could lead to precision loss after `u32::MAX`,
            // but this will not be an issue until context lengths exceed 4_294_967_295.
            delta
                .token_ids
                .len()
                .try_into()
                .expect("token_ids length exceeds u32::MAX")
        } else {
            0
        };

        let logprobs = self.create_logprobs(
            delta.tokens,
            &delta.token_ids,
            delta.log_probs,
            delta.top_logprobs,
        );

        // Map backend finish reasons to OpenAI's finish reasons.
        let finish_reason = match delta.finish_reason {
            Some(common::FinishReason::EoS) => Some(dynamo_async_openai::types::FinishReason::Stop),
            Some(common::FinishReason::Stop) => {
                Some(dynamo_async_openai::types::FinishReason::Stop)
            }
            Some(common::FinishReason::Length) => {
                Some(dynamo_async_openai::types::FinishReason::Length)
            }
            Some(common::FinishReason::Cancelled) => {
                Some(dynamo_async_openai::types::FinishReason::Stop)
            }
            Some(common::FinishReason::ContentFilter) => {
                Some(dynamo_async_openai::types::FinishReason::ContentFilter)
            }
            Some(common::FinishReason::Error(err_msg)) => {
                return Err(anyhow::anyhow!(err_msg));
            }
            None => None,
        };

        // Handle reasoning parsing if enabled, otherwise treat all text as normal
        let (normal_text, reasoning_content) =
            match self.create_reasoning_content(&delta.text, &delta.token_ids) {
                Some(reasoning_parser_result) => {
                    // If we have reasoning parsing results, estimate token distribution
                    if self.options.enable_usage {
                        let reasoning_text = &reasoning_parser_result.reasoning_text;
                        let normal_text = &reasoning_parser_result.normal_text;
                        
                        // Use character-based proportion to estimate reasoning tokens
                        let total_chars = reasoning_text.chars().count() + normal_text.chars().count();
                        if total_chars > 0 {
                            let reasoning_chars = reasoning_text.chars().count();
                            let reasoning_token_ratio = reasoning_chars as f64 / total_chars as f64;
                            let estimated_reasoning_tokens = (token_length as f64 * reasoning_token_ratio).round() as u32;
                            
                            self.reasoning_tokens += estimated_reasoning_tokens;
                            // Subtract reasoning tokens from completion tokens
                            // Only count non-reasoning tokens in completion_tokens
                            self.usage.completion_tokens += token_length.saturating_sub(estimated_reasoning_tokens);
                        } else {
                            // If no text content, count all tokens as completion tokens
                            self.usage.completion_tokens += token_length;
                        }
                    }
                    
                    (
                        reasoning_parser_result.get_some_normal_text(),
                        reasoning_parser_result.get_some_reasoning(),
                    )
                },
                None => {
                    // No reasoning parsing, all tokens are completion tokens
                    if self.options.enable_usage {
                        self.usage.completion_tokens += token_length;
                    }
                    (delta.text, None)
                },
            };

        // Create the streaming response.
        let index = 0;
        let stream_response = self.create_choice(
            index,
            normal_text,
            reasoning_content,
            finish_reason,
            logprobs,
        );

        Ok(stream_response)
    }

    fn get_isl(&self) -> Option<u32> {
        Some(self.usage.prompt_tokens)
    }

    fn create_usage_chunk(&self) -> NvCreateChatCompletionStreamResponse {
        DeltaGenerator::create_usage_chunk(self)
    }

    fn is_usage_enabled(&self) -> bool {
        DeltaGenerator::is_usage_enabled(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers::{ReasoningParserType, ParserResult};
    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use crate::protocols::common::llm_backend::BackendOutput;

    fn create_test_delta_generator() -> DeltaGenerator {
        let options = DeltaGeneratorOptions {
            enable_usage: true,
            enable_logprobs: false,
            runtime_config: ModelRuntimeConfig::default(),
        };
        
        DeltaGenerator::new(
            "test-model".to_string(),
            options,
            "test-request-id".to_string(),
        )
    }

    fn create_test_delta_generator_with_reasoning(parser_name: &str) -> DeltaGenerator {
        let mut runtime_config = ModelRuntimeConfig::default();
        runtime_config.reasoning_parser = Some(parser_name.to_string());
        
        let options = DeltaGeneratorOptions {
            enable_usage: true,
            enable_logprobs: false,
            runtime_config,
        };
        
        DeltaGenerator::new(
            "test-model".to_string(),
            options,
            "test-request-id".to_string(),
        )
    }

    #[test]
    fn test_reasoning_tokens_initialization() {
        let generator = create_test_delta_generator();
        assert_eq!(generator.reasoning_tokens, 0);
        assert!(generator.reasoning_parser.is_none());
    }

    #[test]
    fn test_reasoning_tokens_with_parser() {
        let generator = create_test_delta_generator_with_reasoning("basic");
        assert_eq!(generator.reasoning_tokens, 0);
        assert!(generator.reasoning_parser.is_some());
    }

    #[test]
    fn test_create_usage_chunk_with_reasoning_tokens() {
        let mut generator = create_test_delta_generator_with_reasoning("basic");
        
        // Simulate some reasoning tokens
        generator.reasoning_tokens = 10;
        generator.usage.completion_tokens = 5;
        generator.usage.prompt_tokens = 3;
        
        let usage_chunk = generator.create_usage_chunk();
        
        // Check that usage is populated
        assert!(usage_chunk.usage.is_some());
        let usage = usage_chunk.usage.unwrap();
        
        // Total tokens should include reasoning tokens
        assert_eq!(usage.total_tokens, 3 + 5 + 10); // prompt + completion + reasoning
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(usage.completion_tokens, 5);
        
        // Check completion_tokens_details
        assert!(usage.completion_tokens_details.is_some());
        let details = usage.completion_tokens_details.unwrap();
        assert_eq!(details.reasoning_tokens, Some(10));
        assert_eq!(details.accepted_prediction_tokens, None);
        assert_eq!(details.audio_tokens, None);
        assert_eq!(details.rejected_prediction_tokens, None);
    }

    #[test]
    fn test_create_usage_chunk_no_reasoning_tokens() {
        let generator = create_test_delta_generator();
        
        let usage_chunk = generator.create_usage_chunk();
        
        assert!(usage_chunk.usage.is_some());
        let usage = usage_chunk.usage.unwrap();
        
        // Without reasoning parser, completion_tokens_details should be None
        assert!(usage.completion_tokens_details.is_none());
    }

    #[test]
    fn test_choice_from_postprocessor_without_reasoning() {
        let mut generator = create_test_delta_generator();
        
        let backend_output = BackendOutput {
            text: Some("Hello world".to_string()),
            tokens: vec![], 
            token_ids: vec![1, 2, 3, 4], // 4 tokens
            finish_reason: None,
            log_probs: None,
            top_logprobs: None,
        };
        
        let result = generator.choice_from_postprocessor(backend_output);
        assert!(result.is_ok());
        
        // All tokens should be counted as completion tokens
        assert_eq!(generator.usage.completion_tokens, 4);
        assert_eq!(generator.reasoning_tokens, 0);
    }

    // Note: More comprehensive tests would require mocking the reasoning parser
    // but for now this validates the basic structure and initialization
}
