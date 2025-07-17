// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Module for counting tokens in a response stream and computing performance metrics.
//!
//! This module provides comprehensive token counting and analysis capabilities for streaming
//! chat completion responses. It supports multiple data sources for token counting with
//! different levels of accuracy and includes validation for completeness and consistency.
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use dynamo_llm::perf::{RecordedStream, analyze_token_counting};
//! use std::sync::Arc;
//!
//! // Record a stream (see RecordedStream documentation for details)
//! let recorded_stream: Arc<RecordedStream<_>> = /* ... */;
//!
//! // Option 1: Single token approximation (simplest but least accurate)
//! let analysis = analyze_token_counting(recorded_stream.clone(), None);
//!
//! // Option 2: With a tokenizer (more accurate)
//! use dynamo_llm::tokenizers::Tokenizer;
//! let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;
//! let analysis = analyze_token_counting(recorded_stream, Some(&*tokenizer));
//!
//! // Print comprehensive analysis
//! analysis.print_summary();
//!
//! // Check for validation errors
//! if analysis.has_errors() {
//!     println!("Validation errors found: {:?}", analysis.validation_errors);
//! }
//!
//! // Get token count for specific choice
//! if let Some(tokens) = analysis.total_tokens_for_choice(0) {
//!     println!("Choice 0 generated {} tokens", tokens);
//! }
//!
//! // Check data source consistency
//! let consistency_errors = analysis.validate_data_source_consistency();
//! if !consistency_errors.is_empty() {
//!     println!("Data source inconsistencies: {:?}", consistency_errors);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::perf::RecordedStream;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use crate::protocols::TokenIdType;
use crate::tokenizers::traits::TokenCounter;

pub type TokenCount = usize;
pub type ForwardPassDuration = Duration;

/// Data source used for counting tokens in a specific response chunk.
/// Different sources have different accuracy levels and availability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenDataSource {
    /// OpenAI's usage field
    /// If this is the data source, then every response chunk had a usage field,
    /// and that usage field was self-consistent between chunks.
    ///
    /// This is the preferred data source for counting tokens.
    Usage(TokenCount),

    /// If [`TokenDataSource::Usage`] is not available, but the tokenizer for the model
    /// is available, then we can directly decode the text per chunk to get the token count
    /// per chunk.
    ///
    /// Errors could result if the supplied tokenizer is not compatible with the model.
    Tokenizer(TokenCount),

    /// If neither [`TokenDataSource::Usage`] nor [`TokenDataSource::Tokenizer`] are available,
    /// then the presence of text in the content of the chunk is the signal for tokens being
    /// emitted; however, the number of tokens is not guaranteed to be accurate.
    ///
    /// In this scenario, we assume that each response chunk is a single token; however, this
    /// easily breaks down if the model performs some speculative decoding or batches multiple
    /// tokens into a single response chunk to reduce messaging overheads.
    SingleTokenApproximation,

    /// If this is provided, then we have a custom server-side mechanism for both timing and
    /// counting tokens.
    ///
    /// If this is provided, then we expect one entry per forward pass. This provides the most
    /// accurate timing and token count information as it directly breaks down the time by
    /// forward pass which disambiguates the time for multiple tokens (Usage) and potential
    /// response batching, e.g. [`TokenDataSource::Usage`] or [`TokenDataSource::Tokenizer`]
    /// might tell us that we received 10 tokens, but that could have been from 10 forward passes
    /// batched together or 1 forward pass that emitted 10 tokens via speculative decoding.
    ///
    /// Dynamo Annotations can provide this level of timing details; however, not all frameworks
    /// are fully integrated with Dynamo Annotations.
    ServerSideTiming(TokenCount, ForwardPassDuration),
}

impl TokenDataSource {
    /// Extract the token count from any data source variant
    pub fn token_count(&self) -> TokenCount {
        match self {
            TokenDataSource::Usage(count) => *count,
            TokenDataSource::Tokenizer(count) => *count,
            TokenDataSource::SingleTokenApproximation => 1,
            TokenDataSource::ServerSideTiming(count, _) => *count,
        }
    }

    /// Returns true if this data source is considered accurate
    pub fn is_accurate(&self) -> bool {
        match self {
            TokenDataSource::Usage(_) | TokenDataSource::ServerSideTiming(_, _) => true,
            TokenDataSource::Tokenizer(_) => true, // Assumed accurate if tokenizer matches model
            TokenDataSource::SingleTokenApproximation => false,
        }
    }

    /// Returns a human-readable description of the data source
    pub fn description(&self) -> &'static str {
        match self {
            TokenDataSource::Usage(_) => "OpenAI Usage Field",
            TokenDataSource::Tokenizer(_) => "Tokenizer-based Counting",
            TokenDataSource::SingleTokenApproximation => "Single Token Approximation",
            TokenDataSource::ServerSideTiming(_, _) => "Server-side Timing",
        }
    }
}

/// Token data for a specific choice in a response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceTokenData {
    /// The choice index
    pub choice_index: u32,
    /// Number of tokens in this chunk for this choice
    pub token_count: TokenCount,
    /// Token IDs generated in this chunk (if available)
    pub token_ids: Option<Vec<TokenIdType>>,
    /// Data source used for counting
    pub data_source: TokenDataSource,
    /// Whether this choice has finished (has a finish_reason)
    pub is_finished: bool,
    /// The finish reason if the choice is finished
    pub finish_reason: Option<String>,
    /// Text content that contributed to the token count (for debugging)
    pub content: Option<String>,
}

/// Trait for extracting token information from various response types
pub trait TokenExtractor {
    /// Extract token data organized by choice index
    /// Returns: HashMap<choice_index, ChoiceTokenData>
    fn extract_tokens_by_choice(
        &self,
        tokenizer: Option<&dyn TokenCounter>,
    ) -> HashMap<u32, ChoiceTokenData>;
}

/// Implementation for NvCreateChatCompletionStreamResponse (our main streaming response type)
impl TokenExtractor for NvCreateChatCompletionStreamResponse {
    fn extract_tokens_by_choice(
        &self,
        tokenizer: Option<&dyn TokenCounter>,
    ) -> HashMap<u32, ChoiceTokenData> {
        let mut result = HashMap::new();

        for choice in &self.inner.choices {
            let choice_index = choice.index;
            let content = choice.delta.content.as_deref().unwrap_or("");
            let finish_reason = choice.finish_reason.as_ref().map(|fr| format!("{:?}", fr));
            let is_finished = choice.finish_reason.is_some();

            // Determine token count, token IDs, and data source
            let (token_count, token_ids, data_source) = if let Some(tokenizer) = tokenizer {
                // Use tokenizer if available and there's content
                if !content.is_empty() {
                    match tokenizer.count_tokens_with_ids(content) {
                        Ok((ids, count)) => (count, Some(ids), TokenDataSource::Tokenizer(count)),
                        Err(_) => {
                            // Fallback to single token approximation if tokenizer fails
                            (1, None, TokenDataSource::SingleTokenApproximation)
                        }
                    }
                } else {
                    (0, Some(Vec::new()), TokenDataSource::Tokenizer(0))
                }
            } else {
                // Use single token approximation
                if !content.is_empty() {
                    (1, None, TokenDataSource::SingleTokenApproximation)
                } else {
                    (0, None, TokenDataSource::SingleTokenApproximation)
                }
            };

            result.insert(
                choice_index,
                ChoiceTokenData {
                    choice_index,
                    token_count,
                    token_ids,
                    data_source,
                    is_finished,
                    finish_reason,
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(content.to_string())
                    },
                },
            );
        }

        result
    }
}

/// Event representing token generation at a specific point in the stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    /// Position in the stream (response index)
    pub stream_position: usize,
    /// Number of tokens in this chunk
    pub tokens_in_chunk: TokenCount,
    /// Cumulative tokens up to this point
    pub cumulative_tokens: TokenCount,
    /// Data source used for this chunk
    pub data_source: TokenDataSource,
    /// Content that generated these tokens (for debugging)
    pub content: Option<String>,
}

/// Analysis for a single choice across the entire stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceTokenAnalysis {
    /// Choice index
    pub choice_index: u32,
    /// Total tokens generated for this choice
    pub total_tokens: TokenCount,
    /// Primary data source used (most common in the stream)
    pub primary_data_source: TokenDataSource,
    /// Whether this choice completed with a finish_reason
    pub is_complete: bool,
    /// The finish reason if completed
    pub finish_reason: Option<String>,
    /// Timeline of token generation events
    pub token_timeline: Vec<TokenEvent>,
    /// Number of responses where this choice appeared
    pub responses_with_choice: usize,
    /// Position where the choice finished (if it finished)
    pub finish_position: Option<usize>,
}

/// Comprehensive token analysis for all choices in a recorded stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAnalysis {
    /// Total number of responses analyzed
    pub total_responses: usize,
    /// Analysis results per choice index
    pub choice_analyses: HashMap<u32, ChoiceTokenAnalysis>,
    /// Validation errors found during analysis
    pub validation_errors: Vec<String>,
    /// Whether the stream appears to be complete (all choices finished)
    pub is_stream_complete: bool,
}

/// Validates and analyzes token counting from a recorded stream
pub fn analyze_token_counting<T: TokenExtractor>(
    recorded_stream: Arc<RecordedStream<T>>,
    tokenizer: Option<&dyn TokenCounter>,
) -> TokenAnalysis {
    let mut choice_analyses: HashMap<u32, ChoiceTokenAnalysis> = HashMap::new();
    let mut validation_errors = Vec::new();

    // Track which choices have finished and when
    let mut finished_choices: HashMap<u32, usize> = HashMap::new();

    for (stream_pos, timestamped_response) in recorded_stream.responses().iter().enumerate() {
        let response = &timestamped_response.response;
        let tokens_by_choice = response.extract_tokens_by_choice(tokenizer);

        for (choice_index, choice_token_data) in tokens_by_choice {
            // Get or create choice analysis
            let choice_analysis =
                choice_analyses
                    .entry(choice_index)
                    .or_insert_with(|| ChoiceTokenAnalysis {
                        choice_index,
                        total_tokens: 0,
                        primary_data_source: TokenDataSource::SingleTokenApproximation,
                        is_complete: false,
                        finish_reason: None,
                        token_timeline: Vec::new(),
                        responses_with_choice: 0,
                        finish_position: None,
                    });

            choice_analysis.responses_with_choice += 1;

            // Check if choice was already finished
            if let Some(finish_pos) = finished_choices.get(&choice_index) {
                if choice_token_data.token_count > 0 {
                    validation_errors.push(format!(
                        "Choice {} generated {} tokens at position {} after finishing at position {}",
                        choice_index, choice_token_data.token_count, stream_pos, finish_pos
                    ));
                }
            }

            // Update total tokens
            choice_analysis.total_tokens += choice_token_data.token_count;

            // Record token event
            choice_analysis.token_timeline.push(TokenEvent {
                stream_position: stream_pos,
                tokens_in_chunk: choice_token_data.token_count,
                cumulative_tokens: choice_analysis.total_tokens,
                data_source: choice_token_data.data_source.clone(),
                content: choice_token_data.content,
            });

            // Update finish status
            if choice_token_data.is_finished {
                choice_analysis.is_complete = true;
                choice_analysis.finish_reason = choice_token_data.finish_reason;
                choice_analysis.finish_position = Some(stream_pos);
                finished_choices.insert(choice_index, stream_pos);
            }

            // Update primary data source (use the most accurate one seen)
            if choice_token_data.data_source.is_accurate()
                && !choice_analysis.primary_data_source.is_accurate()
            {
                choice_analysis.primary_data_source = choice_token_data.data_source;
            }
        }
    }

    // Validate that all choices are complete
    let incomplete_choices: Vec<u32> = choice_analyses
        .values()
        .filter(|analysis| !analysis.is_complete)
        .map(|analysis| analysis.choice_index)
        .collect();

    let is_stream_complete = incomplete_choices.is_empty();

    if !incomplete_choices.is_empty() {
        validation_errors.push(format!(
            "Stream incomplete: choices {:?} did not finish with a finish_reason",
            incomplete_choices
        ));
    }

    TokenAnalysis {
        total_responses: recorded_stream.responses().len(),
        choice_analyses,
        validation_errors,
        is_stream_complete,
    }
}

impl TokenAnalysis {
    /// Get total tokens for a specific choice
    pub fn total_tokens_for_choice(&self, choice_index: u32) -> Option<TokenCount> {
        self.choice_analyses
            .get(&choice_index)
            .map(|analysis| analysis.total_tokens)
    }

    /// Get all choice indices present in the analysis
    pub fn choice_indices(&self) -> Vec<u32> {
        let mut indices: Vec<u32> = self.choice_analyses.keys().copied().collect();
        indices.sort();
        indices
    }

    /// Check if the analysis has any validation errors
    pub fn has_errors(&self) -> bool {
        !self.validation_errors.is_empty()
    }

    /// Get a summary of data sources used across all choices
    pub fn data_source_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();
        for analysis in self.choice_analyses.values() {
            let source_name = analysis.primary_data_source.description();
            *summary.entry(source_name.to_string()).or_insert(0) += 1;
        }
        summary
    }

    /// Print a comprehensive summary of the token analysis
    pub fn print_summary(&self) {
        println!("=== Token Analysis Summary ===");
        println!("Total stream responses: {}", self.total_responses);
        println!("Number of choices: {}", self.choice_analyses.len());
        println!("Stream complete: {}", self.is_stream_complete);

        if self.has_errors() {
            println!("\n⚠️  Validation Errors:");
            for error in &self.validation_errors {
                println!("  - {}", error);
            }
        }

        println!("\nData Source Summary:");
        for (source, count) in self.data_source_summary() {
            println!("  {}: {} choices", source, count);
        }

        println!("\nPer-Choice Analysis:");
        for choice_index in self.choice_indices() {
            let analysis = &self.choice_analyses[&choice_index];
            let status = if analysis.is_complete {
                format!(
                    "✓ Complete ({})",
                    analysis.finish_reason.as_deref().unwrap_or("unknown")
                )
            } else {
                "❌ Incomplete".to_string()
            };

            println!(
                "  Choice {}: {} tokens, {} - {}",
                choice_index,
                analysis.total_tokens,
                analysis.primary_data_source.description(),
                status
            );
        }
    }

    /// Get token generation rate (tokens per response that generated tokens) for a choice
    /// This excludes responses with 0 tokens from the calculation
    pub fn token_rate_for_choice(&self, choice_index: u32) -> Option<f64> {
        self.choice_analyses.get(&choice_index).map(|analysis| {
            // Count only responses that actually generated tokens (> 0)
            let responses_with_tokens = analysis
                .token_timeline
                .iter()
                .filter(|event| event.tokens_in_chunk > 0)
                .count();

            if responses_with_tokens > 0 {
                analysis.total_tokens as f64 / responses_with_tokens as f64
            } else {
                0.0
            }
        })
    }

    /// Check for consistency in data sources across the stream
    pub fn validate_data_source_consistency(&self) -> Vec<String> {
        let mut errors = Vec::new();

        for analysis in self.choice_analyses.values() {
            let mut sources: HashMap<String, usize> = HashMap::new();
            for event in &analysis.token_timeline {
                let source_name = event.data_source.description();
                *sources.entry(source_name.to_string()).or_insert(0) += 1;
            }

            if sources.len() > 1 {
                errors.push(format!(
                    "Choice {} used multiple data sources: {:?}",
                    analysis.choice_index, sources
                ));
            }
        }

        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::{RecordedStream, TimestampedResponse};
    use anyhow::Result;
    use async_openai::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
        FinishReason, Role,
    };
    use std::time::Instant;

    // Mock tokenizer for testing
    struct MockTokenizer;

    impl TokenCounter for MockTokenizer {
        fn count_tokens(&self, text: &str) -> crate::tokenizers::Result<TokenCount> {
            // Simple mock: count words as tokens
            Ok(text.split_whitespace().count())
        }

        fn count_tokens_with_ids(
            &self,
            text: &str,
        ) -> crate::tokenizers::Result<(Vec<TokenIdType>, usize)> {
            // Simple mock: create fake token IDs (just indices) and count words
            let count = text.split_whitespace().count();
            let token_ids: Vec<TokenIdType> = (0..count as u32).collect();
            Ok((token_ids, count))
        }
    }

    #[test]
    fn test_single_token_approximation() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello", false, None),
                create_mock_response_with_content(0, " world", false, None),
                create_mock_response_with_content(0, "!", true, Some(FinishReason::Stop)),
            ],
            None,
        );

        assert_eq!(analysis.choice_analyses.len(), 1);
        let choice_analysis = &analysis.choice_analyses[&0];

        // Should count 3 tokens (one per response with content)
        assert_eq!(choice_analysis.total_tokens, 3);
        assert!(matches!(
            choice_analysis.primary_data_source,
            TokenDataSource::SingleTokenApproximation
        ));
        assert!(choice_analysis.is_complete);
        assert_eq!(choice_analysis.finish_reason.as_deref(), Some("Stop"));
    }

    #[test]
    fn test_tokenizer_based_counting() {
        let tokenizer = MockTokenizer;
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello world", false, None),
                create_mock_response_with_content(0, "How are you", false, None),
                create_mock_response_with_content(0, "today?", true, Some(FinishReason::Stop)),
            ],
            Some(&tokenizer),
        );

        assert_eq!(analysis.choice_analyses.len(), 1);
        let choice_analysis = &analysis.choice_analyses[&0];

        // Should count: 2 + 3 + 1 = 6 tokens (word count)
        assert_eq!(choice_analysis.total_tokens, 6);
        assert!(matches!(
            choice_analysis.primary_data_source,
            TokenDataSource::Tokenizer(_)
        ));
        assert!(choice_analysis.is_complete);
    }

    #[test]
    fn test_multiple_choices() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi", false, None),
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, " world", true, Some(FinishReason::Stop)),
                    (1, " there", true, Some(FinishReason::Stop)),
                ]),
            ],
            None,
        );

        assert_eq!(analysis.choice_analyses.len(), 2);
        assert!(analysis.is_stream_complete);

        for choice_index in [0, 1] {
            let choice_analysis = &analysis.choice_analyses[&choice_index];
            assert_eq!(choice_analysis.total_tokens, 2);
            assert!(choice_analysis.is_complete);
        }
    }

    #[test]
    fn test_incomplete_stream_validation() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello", false, None),
                create_mock_response_with_content(0, " world", false, None),
                // Missing finish reason
            ],
            None,
        );

        assert!(!analysis.is_stream_complete);
        assert!(analysis.has_errors());
        assert!(analysis
            .validation_errors
            .iter()
            .any(|err| err.contains("did not finish")));
    }

    #[test]
    fn test_tokens_after_finish_validation() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello", false, None),
                create_mock_response_with_content(0, " world", true, Some(FinishReason::Stop)),
                create_mock_response_with_content(0, "!", false, None), // Invalid: after finish
            ],
            None,
        );

        assert!(analysis.has_errors());
        assert!(analysis
            .validation_errors
            .iter()
            .any(|err| err.contains("after finishing")));
    }

    #[test]
    fn test_empty_content_handling() {
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "", false, None), // Empty content
                create_mock_response_with_content(0, "Hello", false, None),
                create_mock_response_with_content(0, "", true, Some(FinishReason::Stop)), // Empty finish
            ],
            None,
        );

        let choice_analysis = &analysis.choice_analyses[&0];
        // Should count only 1 token (for "Hello")
        assert_eq!(choice_analysis.total_tokens, 1);
        assert!(choice_analysis.is_complete);
    }

    #[test]
    fn test_multiple_choices_different_finish_times() {
        // Choice 0 finishes at response 2, Choice 1 finishes at response 4
        let analysis = create_analysis_with_responses(
            vec![
                // Response 0: Both choices generate tokens
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi there", false, None),
                ]),
                // Response 1: Both choices continue
                create_mock_response_with_multiple_choices(vec![
                    (0, " world", false, None),
                    (1, " my friend", false, None),
                ]),
                // Response 2: Choice 0 finishes, Choice 1 continues
                create_mock_response_with_multiple_choices(vec![
                    (0, "!", true, Some(FinishReason::Stop)),
                    (1, " how", false, None),
                ]),
                // Response 3: Only Choice 1 continues (Choice 0 is finished)
                create_mock_response_with_multiple_choices(vec![(1, " are", false, None)]),
                // Response 4: Choice 1 finishes
                create_mock_response_with_multiple_choices(vec![(
                    1,
                    " you?",
                    true,
                    Some(FinishReason::Stop),
                )]),
            ],
            None,
        );

        // Verify basic structure
        assert_eq!(analysis.choice_analyses.len(), 2);
        assert!(analysis.is_stream_complete);
        assert!(!analysis.has_errors());

        // Check Choice 0: finished early at position 2
        let choice0 = &analysis.choice_analyses[&0];
        assert_eq!(choice0.total_tokens, 3); // "Hello", " world", "!"
        assert!(choice0.is_complete);
        assert_eq!(choice0.finish_position, Some(2));
        assert_eq!(choice0.finish_reason.as_deref(), Some("Stop"));
        assert_eq!(choice0.responses_with_choice, 3); // Appeared in responses 0, 1, 2

        // Check Choice 1: finished later at position 4
        let choice1 = &analysis.choice_analyses[&1];
        assert_eq!(choice1.total_tokens, 5); // "Hi there", " my friend", " how", " are", " you?"
        assert!(choice1.is_complete);
        assert_eq!(choice1.finish_position, Some(4));
        assert_eq!(choice1.finish_reason.as_deref(), Some("Stop"));
        assert_eq!(choice1.responses_with_choice, 5); // Appeared in all responses

        // Verify token timelines
        assert_eq!(choice0.token_timeline.len(), 3);
        assert_eq!(choice1.token_timeline.len(), 5);

        // Check that cumulative tokens are correct
        assert_eq!(choice0.token_timeline[0].cumulative_tokens, 1);
        assert_eq!(choice0.token_timeline[1].cumulative_tokens, 2);
        assert_eq!(choice0.token_timeline[2].cumulative_tokens, 3);

        assert_eq!(choice1.token_timeline[0].cumulative_tokens, 1);
        assert_eq!(choice1.token_timeline[1].cumulative_tokens, 2);
        assert_eq!(choice1.token_timeline[2].cumulative_tokens, 3);
        assert_eq!(choice1.token_timeline[3].cumulative_tokens, 4);
        assert_eq!(choice1.token_timeline[4].cumulative_tokens, 5);
    }

    #[test]
    fn test_choice_missing_from_middle_responses() {
        // Choice 0 appears in responses 0, 2, 4 (missing from 1, 3)
        // Choice 1 appears in responses 1, 3, 4 (missing from 0, 2)
        let analysis = create_analysis_with_responses(
            vec![
                // Response 0: Only Choice 0
                create_mock_response_with_multiple_choices(vec![(0, "Hello", false, None)]),
                // Response 1: Only Choice 1
                create_mock_response_with_multiple_choices(vec![(1, "Hi", false, None)]),
                // Response 2: Only Choice 0
                create_mock_response_with_multiple_choices(vec![(0, " world", false, None)]),
                // Response 3: Only Choice 1
                create_mock_response_with_multiple_choices(vec![(1, " there", false, None)]),
                // Response 4: Both choices finish
                create_mock_response_with_multiple_choices(vec![
                    (0, "!", true, Some(FinishReason::Stop)),
                    (1, "!", true, Some(FinishReason::Stop)),
                ]),
            ],
            None,
        );

        assert_eq!(analysis.choice_analyses.len(), 2);
        assert!(analysis.is_stream_complete);

        // Choice 0: appeared in responses 0, 2, 4
        let choice0 = &analysis.choice_analyses[&0];
        assert_eq!(choice0.total_tokens, 3);
        assert_eq!(choice0.responses_with_choice, 3);
        assert_eq!(choice0.finish_position, Some(4));

        // Choice 1: appeared in responses 1, 3, 4
        let choice1 = &analysis.choice_analyses[&1];
        assert_eq!(choice1.total_tokens, 3);
        assert_eq!(choice1.responses_with_choice, 3);
        assert_eq!(choice1.finish_position, Some(4));
    }

    #[test]
    fn test_validation_tokens_after_finish_multiple_choices() {
        // Choice 0 finishes early but then generates more tokens (invalid)
        // Choice 1 finishes properly
        let analysis = create_analysis_with_responses(
            vec![
                // Response 0: Both choices generate tokens
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi", false, None),
                ]),
                // Response 1: Choice 0 finishes, Choice 1 continues
                create_mock_response_with_multiple_choices(vec![
                    (0, " world", true, Some(FinishReason::Stop)),
                    (1, " there", false, None),
                ]),
                // Response 2: Choice 0 generates more tokens (INVALID), Choice 1 finishes
                create_mock_response_with_multiple_choices(vec![
                    (0, "!", false, None), // This should trigger a validation error
                    (1, "!", true, Some(FinishReason::Stop)),
                ]),
            ],
            None,
        );

        // Should detect the validation error
        assert!(analysis.has_errors());
        assert!(analysis
            .validation_errors
            .iter()
            .any(|err| err.contains("Choice 0") && err.contains("after finishing at position 1")));

        // Choice 0 should still show as complete (finished at position 1)
        let choice0 = &analysis.choice_analyses[&0];
        assert!(choice0.is_complete);
        assert_eq!(choice0.finish_position, Some(1));
        // But total tokens includes the invalid token
        assert_eq!(choice0.total_tokens, 3);

        // Choice 1 should be fine
        let choice1 = &analysis.choice_analyses[&1];
        assert!(choice1.is_complete);
        assert_eq!(choice1.finish_position, Some(2));
        assert_eq!(choice1.total_tokens, 3);
    }

    #[test]
    fn test_stream_incomplete_when_some_choices_dont_finish() {
        // Choice 0 finishes, Choice 1 doesn't finish
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi", false, None),
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, " world!", true, Some(FinishReason::Stop)),
                    (1, " there", false, None), // No finish reason - incomplete
                ]),
            ],
            None,
        );

        // Stream should be incomplete
        assert!(!analysis.is_stream_complete);
        assert!(analysis.has_errors());
        assert!(analysis
            .validation_errors
            .iter()
            .any(|err| err.contains("choices [1]") && err.contains("did not finish")));

        // Choice 0 should be complete
        let choice0 = &analysis.choice_analyses[&0];
        assert!(choice0.is_complete);

        // Choice 1 should be incomplete
        let choice1 = &analysis.choice_analyses[&1];
        assert!(!choice1.is_complete);
        assert_eq!(choice1.finish_position, None);
    }

    #[test]
    fn test_choice_appears_only_for_finish() {
        // Choice 0 generates tokens normally
        // Choice 1 only appears in the final response with finish_reason (edge case)
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![(0, "Hello", false, None)]),
                create_mock_response_with_multiple_choices(vec![(0, " world", false, None)]),
                create_mock_response_with_multiple_choices(vec![
                    (0, "!", true, Some(FinishReason::Stop)),
                    (1, "", true, Some(FinishReason::Stop)), // Only appears for finish
                ]),
            ],
            None,
        );

        assert_eq!(analysis.choice_analyses.len(), 2);
        assert!(analysis.is_stream_complete);

        // Choice 0: normal case
        let choice0 = &analysis.choice_analyses[&0];
        assert_eq!(choice0.total_tokens, 3);
        assert_eq!(choice0.responses_with_choice, 3);

        // Choice 1: only appeared once for finish
        let choice1 = &analysis.choice_analyses[&1];
        assert_eq!(choice1.total_tokens, 0); // No actual content tokens
        assert_eq!(choice1.responses_with_choice, 1);
        assert!(choice1.is_complete);
        assert_eq!(choice1.finish_position, Some(2));
    }

    #[test]
    fn test_different_finish_reasons() {
        // Test multiple choices with different finish reasons
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello", false, None),
                    (1, "Hi", false, None),
                    (2, "Hey", false, None),
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, " world", true, Some(FinishReason::Stop)),
                    (1, " there", true, Some(FinishReason::Length)),
                    (2, " you", true, Some(FinishReason::ContentFilter)),
                ]),
            ],
            None,
        );

        assert!(analysis.is_stream_complete);
        assert_eq!(analysis.choice_analyses.len(), 3);

        // Check different finish reasons
        assert_eq!(
            analysis.choice_analyses[&0].finish_reason.as_deref(),
            Some("Stop")
        );
        assert_eq!(
            analysis.choice_analyses[&1].finish_reason.as_deref(),
            Some("Length")
        );
        assert_eq!(
            analysis.choice_analyses[&2].finish_reason.as_deref(),
            Some("ContentFilter")
        );

        // All should be complete
        assert!(analysis.choice_analyses[&0].is_complete);
        assert!(analysis.choice_analyses[&1].is_complete);
        assert!(analysis.choice_analyses[&2].is_complete);
    }

    #[test]
    fn test_token_rate_calculation_different_appearances() {
        // Choice 0 appears in 3 responses, generates 5 tokens
        // Choice 1 appears in 2 responses, generates 3 tokens
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_multiple_choices(vec![
                    (0, "Hello world", false, None), // 2 tokens with tokenizer
                    (1, "Hi there", false, None),    // 2 tokens with tokenizer
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, "How are", false, None), // 2 tokens
                                                 // Choice 1 missing from this response
                ]),
                create_mock_response_with_multiple_choices(vec![
                    (0, "you?", true, Some(FinishReason::Stop)), // 1 token (single word), total = 5
                    (1, "today?", true, Some(FinishReason::Stop)), // 1 token (single word), total = 3
                ]),
            ],
            Some(&MockTokenizer),
        );

        // Choice 0: 5 tokens / 3 appearances = 1.6666... tokens per response
        assert!((analysis.token_rate_for_choice(0).unwrap() - (5.0 / 3.0)).abs() < 0.001);

        // Choice 1: 3 tokens / 2 appearances = 1.5 tokens per response
        assert_eq!(analysis.token_rate_for_choice(1), Some(1.5));

        // Non-existent choice
        assert_eq!(analysis.token_rate_for_choice(99), None);
    }

    #[test]
    fn test_real_tokenizer_integration() {
        // This test shows how to use the real tokenizer infrastructure
        // Note: This test uses a mock since we don't have a real tokenizer file in tests

        // Demonstrate that our TokenCounter trait works with any Encoder
        let mock_tokenizer = MockTokenizer;

        // Test the TokenCounter methods directly
        let result = mock_tokenizer.count_tokens("Hello world test").unwrap();
        assert_eq!(result, 3); // "Hello", "world", "test"

        let (token_ids, count) = mock_tokenizer.count_tokens_with_ids("Hello world").unwrap();
        assert_eq!(count, 2);
        assert_eq!(token_ids, vec![0, 1]); // Mock IDs

        // Test integration with our analysis system
        let analysis = create_analysis_with_responses(
            vec![
                create_mock_response_with_content(0, "Hello world", false, None),
                create_mock_response_with_content(0, "How are you", false, None),
                create_mock_response_with_content(0, "today?", true, Some(FinishReason::Stop)),
            ],
            Some(&mock_tokenizer),
        );

        let choice_analysis = &analysis.choice_analyses[&0];
        assert_eq!(choice_analysis.total_tokens, 6); // 2 + 3 + 1

        // Verify that token IDs are captured
        let timeline = &choice_analysis.token_timeline;
        assert_eq!(timeline.len(), 3);
        assert!(timeline[0].content.is_some()); // Has content

        // Check that the data source is correctly identified as Tokenizer
        assert!(matches!(
            choice_analysis.primary_data_source,
            TokenDataSource::Tokenizer(_)
        ));
    }

    #[tokio::test]
    async fn test_deepseek_stream_with_extracted_tokenizer() -> Result<()> {
        use crate::tokenizers::Tokenizer;
        use crate::utils::bzip2::Bzip2Extractor;
        use std::sync::Arc;

        // Step 1: Extract and validate the tokenizer
        let tokenizer_path = "tests/data/replays/deepseek-r1-distill-llama-8b/tokenizer-deepseek-r1-distill-llama-8b.json.bz2";
        let extraction = Bzip2Extractor::builder()
            .source_path(tokenizer_path)
            .target_filename("tokenizer.json")
            .extract()?;

        // Validate the BLAKE3 hash
        let expected_hash = "c61f943c9f3266a60a7e00e815591061f17564f297dd84433a101fb43eb15608";
        extraction.validate_blake3_hash(expected_hash)?;
        println!("✓ Tokenizer BLAKE3 hash validated successfully");

        // Step 2: Load the tokenizer
        let tokenizer_path_str = extraction.file_path().to_string_lossy();
        let tokenizer = Tokenizer::from_file(&tokenizer_path_str)?;
        println!("✓ Tokenizer loaded successfully");

        // Step 3: Read and record the deepseek stream
        let stream_path =
            "tests/data/replays/deepseek-r1-distill-llama-8b/chat-completions.stream.1";
        let data_stream = crate::perf::read_annotated_stream_from_file::<
            NvCreateChatCompletionStreamResponse,
        >(stream_path)?;

        // Step 4: Record the stream using the simplified API
        let recorded_stream = crate::perf::record_data_stream(data_stream).await;

        println!(
            "✓ Stream recorded with {} responses",
            recorded_stream.response_count()
        );

        // Step 5: Extract the raw response data from Annotated wrappers
        let raw_responses: Vec<TimestampedResponse<NvCreateChatCompletionStreamResponse>> =
            recorded_stream
                .responses()
                .iter()
                .filter_map(|timestamped| {
                    // Only include responses where data is Some(T)
                    timestamped
                        .response
                        .data
                        .as_ref()
                        .map(|data| TimestampedResponse {
                            response: data.clone(),
                            timestamp: timestamped.timestamp,
                            sequence_number: timestamped.sequence_number,
                        })
                })
                .collect();

        let raw_recorded_stream = RecordedStream::new(
            raw_responses,
            *recorded_stream.start_time(),
            *recorded_stream.end_time(),
        );
        let arc_stream = Arc::new(raw_recorded_stream);

        // Step 6: Analyze token counting with the real tokenizer
        // Use the Tokenizer wrapper directly (not dereferenced)
        let analysis = analyze_token_counting(arc_stream, Some(&tokenizer));

        // Step 7: Validate the analysis results
        println!("=== DeepSeek Stream Analysis Results ===");
        analysis.print_summary();

        // Basic validation
        assert!(
            !analysis.choice_analyses.is_empty(),
            "Should have at least one choice"
        );

        assert!(analysis.choice_analyses.len() == 1);

        // Check that we're using the tokenizer data source
        let choice_0 = analysis
            .choice_analyses
            .get(&0)
            .expect("Should have choice 0");

        assert!(
            matches!(choice_0.primary_data_source, TokenDataSource::Tokenizer(_)),
            "Should use tokenizer data source"
        );

        // Verify we have the expected token count for this specific test case
        assert_eq!(
            choice_0.total_tokens, 32,
            "Expected exactly 32 tokens for this DeepSeek stream"
        );

        // Verify the stream was successfully completed
        assert!(
            analysis.is_stream_complete,
            "Stream should be complete with all choices finished"
        );

        // Verify the specific choice completed properly
        assert!(
            choice_0.is_complete,
            "Choice 0 should be complete with a finish_reason"
        );

        println!(
            "Choice 0: {} tokens, complete: {}",
            choice_0.total_tokens, choice_0.is_complete
        );

        // Validate that we have some token timeline events
        assert!(
            !choice_0.token_timeline.is_empty(),
            "Should have token events"
        );

        // Check data source consistency
        let consistency_errors = analysis.validate_data_source_consistency();
        if !consistency_errors.is_empty() {
            println!("Data source consistency warnings: {:?}", consistency_errors);
        }

        // Print token rate information
        if let Some(rate) = analysis.token_rate_for_choice(0) {
            println!(
                "Token generation rate for choice 0: {:.2} tokens per response (excluding empty responses)",
                rate
            );
        }

        println!("✓ DeepSeek stream analysis completed successfully!");

        Ok(())
    }

    // Helper functions
    fn create_analysis_with_responses(
        responses: Vec<NvCreateChatCompletionStreamResponse>,
        tokenizer: Option<&dyn TokenCounter>,
    ) -> TokenAnalysis {
        let start_time = Instant::now();
        let timestamped_responses = responses
            .into_iter()
            .enumerate()
            .map(|(i, response)| TimestampedResponse::new(response, i))
            .collect();

        let recorded_stream =
            RecordedStream::new(timestamped_responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        analyze_token_counting(arc_stream, tokenizer)
    }

    fn create_mock_response_with_content(
        choice_index: u32,
        content: &str,
        _is_finished: bool,
        finish_reason: Option<FinishReason>,
    ) -> NvCreateChatCompletionStreamResponse {
        #[expect(deprecated)]
        let inner = CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            choices: vec![ChatChoiceStream {
                index: choice_index,
                delta: ChatCompletionStreamResponseDelta {
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(content.to_string())
                    },
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                },
                finish_reason,
                logprobs: None,
            }],
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        };

        NvCreateChatCompletionStreamResponse { inner }
    }

    fn create_mock_response_with_multiple_choices(
        choices_data: Vec<(u32, &str, bool, Option<FinishReason>)>,
    ) -> NvCreateChatCompletionStreamResponse {
        #[expect(deprecated)]
        let choices = choices_data
            .into_iter()
            .map(
                |(choice_index, content, _is_finished, finish_reason)| ChatChoiceStream {
                    index: choice_index,
                    delta: ChatCompletionStreamResponseDelta {
                        content: if content.is_empty() {
                            None
                        } else {
                            Some(content.to_string())
                        },
                        function_call: None,
                        tool_calls: None,
                        role: Some(Role::Assistant),
                        refusal: None,
                    },
                    finish_reason,
                    logprobs: None,
                },
            )
            .collect();

        let inner = CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            choices,
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        };

        NvCreateChatCompletionStreamResponse { inner }
    }
}
