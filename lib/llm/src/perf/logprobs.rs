// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Module for recording logprobs from a streaming response.
//!
//! Logprobs are a bit easier than token counting and timing because they are
//! fully self-contained in the response chunk.
//!
//! In fact, if logprobs are given, they are a good way to count tokens; however,
//! the emission of logprobs is also more costly and generally not available unless
//! explicitly requested.
//!
//! The primary reason to record logprobs is to analyze the possible outputs of
//! a model as a function of sequence position.

use std::sync::Arc;

use crate::perf::RecordedStream;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

/// The type of logprobs observed in the response.
pub enum LogprobType {
    /// If normalized, then all the reported "top_logprobs" sum to 0.
    Normalized,

    /// If unnormalized, then the reported "top_logprobs" are not normalized,
    /// so the sum of the "top_logprobs" will not sum to 0.
    Unnormalized,
}

/// Represents a token with its logprob information
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    /// The token as a string
    pub token: String,
    /// The log probability of this token
    pub logprob: f32,
    /// Optional byte representation of the token
    pub bytes: Option<Vec<u8>>,
}

/// Trait for extracting logprob information from various response types
pub trait LogprobExtractor {
    /// Extract logprobs organized by choice
    /// Returns: Vec<choice> of Vec<position> of Vec<TokenLogprob>
    /// Each position contains all available tokens (main + alternatives) sorted by logprob
    fn extract_logprobs_by_choice(&self) -> Vec<Vec<Vec<TokenLogprob>>>;
}

/// Implementation for NvCreateChatCompletionStreamResponse (our main streaming response type)
impl LogprobExtractor for NvCreateChatCompletionStreamResponse {
    fn extract_logprobs_by_choice(&self) -> Vec<Vec<Vec<TokenLogprob>>> {
        self.inner
            .choices
            .iter()
            .map(|choice| {
                choice
                    .logprobs
                    .as_ref()
                    .and_then(|logprobs| logprobs.content.as_ref())
                    .map(|content| {
                        content
                            .iter()
                            .map(|token_logprob| {
                                // Combine main token with top alternatives
                                let mut all_logprobs = vec![TokenLogprob {
                                    token: token_logprob.token.clone(),
                                    logprob: token_logprob.logprob,
                                    bytes: token_logprob.bytes.clone(),
                                }];

                                // Add top alternatives
                                for top_logprob in &token_logprob.top_logprobs {
                                    all_logprobs.push(TokenLogprob {
                                        token: top_logprob.token.clone(),
                                        logprob: top_logprob.logprob,
                                        bytes: top_logprob.bytes.clone(),
                                    });
                                }

                                all_logprobs
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            })
            .collect()
    }
}

/// Analysis focused on detecting close logprobs indicating model uncertainty
#[derive(Debug, Clone)]
pub struct SensitivityAnalysis {
    /// Total number of positions analyzed
    pub total_positions: usize,
    /// Analysis results per choice
    pub choice_analyses: Vec<ChoiceAnalysis>,
}

/// Analysis for a single choice
#[derive(Debug, Clone)]
pub struct ChoiceAnalysis {
    /// Choice index
    pub choice_index: usize,
    /// All positions with their closeness values, sorted by closeness
    pub position_closeness: Vec<PositionCloseness>,
    /// Number of positions analyzed for this choice
    pub positions_analyzed: usize,
}

/// Closeness information for a position
#[derive(Debug, Clone)]
pub struct PositionCloseness {
    /// Position in the stream (response index)
    pub stream_position: usize,
    /// Position within the token sequence
    pub token_position: usize,
    /// Logprob difference between top 2 candidates
    pub logprob_difference: f32,
    /// All candidates at this position, sorted by logprob (highest first)
    pub candidates: Vec<TokenLogprob>,
}

/// A position where top candidates have close logprobs
#[derive(Debug, Clone)]
pub struct ClosePosition {
    /// Position in the stream (response index)
    pub stream_position: usize,
    /// Position within the token sequence
    pub token_position: usize,
    /// Logprob difference between top 2 candidates
    pub logprob_difference: f32,
    /// Top 2 candidates at this position
    pub top_candidates: Vec<TokenLogprob>,
}

/// Analyzes logprobs from a recorded stream focusing on token similarity/closeness
pub fn analyze_logprob_sensitivity(
    recorded_stream: Arc<RecordedStream<NvCreateChatCompletionStreamResponse>>,
) -> SensitivityAnalysis {
    let mut choice_analyses = Vec::new();

    for (stream_pos, timestamped_response) in recorded_stream.responses().iter().enumerate() {
        let response = &timestamped_response.response;
        let logprobs_by_choice = response.extract_logprobs_by_choice();

        for (choice_idx, choice_logprobs) in logprobs_by_choice.iter().enumerate() {
            // Ensure we have a ChoiceAnalysis for this choice
            if choice_analyses.len() <= choice_idx {
                choice_analyses.resize_with(choice_idx + 1, || ChoiceAnalysis {
                    choice_index: choice_idx,
                    position_closeness: Vec::new(),
                    positions_analyzed: 0,
                });
            }

            let choice_analysis = &mut choice_analyses[choice_idx];

            for (token_pos, position_logprobs) in choice_logprobs.iter().enumerate() {
                if position_logprobs.len() < 2 {
                    continue;
                }

                // Sort by logprob (highest first)
                let mut sorted_candidates = position_logprobs.clone();
                sorted_candidates.sort_by(|a, b| b.logprob.partial_cmp(&a.logprob).unwrap());

                // Calculate difference between top 2
                let logprob_difference =
                    sorted_candidates[0].logprob - sorted_candidates[1].logprob;

                choice_analysis.position_closeness.push(PositionCloseness {
                    stream_position: stream_pos,
                    token_position: token_pos,
                    logprob_difference,
                    candidates: sorted_candidates,
                });

                choice_analysis.positions_analyzed += 1;
            }
        }
    }

    // Sort position closeness by difference (smallest first = most uncertain)
    for choice_analysis in &mut choice_analyses {
        choice_analysis.position_closeness.sort_by(|a, b| {
            a.logprob_difference
                .partial_cmp(&b.logprob_difference)
                .unwrap()
        });
    }

    SensitivityAnalysis {
        total_positions: recorded_stream.responses().len(),
        choice_analyses,
    }
}

impl SensitivityAnalysis {
    /// Get positions below a threshold for a specific choice
    pub fn get_close_positions_for_choice(
        &self,
        choice_index: usize,
        threshold: f32,
    ) -> Vec<&PositionCloseness> {
        self.choice_analyses
            .get(choice_index)
            .map(|analysis| {
                analysis
                    .position_closeness
                    .iter()
                    .filter(|pos| pos.logprob_difference <= threshold)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the closest N positions for a specific choice
    pub fn get_closest_positions_for_choice(
        &self,
        choice_index: usize,
        count: usize,
    ) -> Vec<&PositionCloseness> {
        self.choice_analyses
            .get(choice_index)
            .map(|analysis| analysis.position_closeness.iter().take(count).collect())
            .unwrap_or_default()
    }

    /// Print a summary of the sensitivity analysis
    pub fn print_summary(&self) {
        println!("=== Logprob Sensitivity Analysis Summary ===");
        println!("Total stream positions analyzed: {}", self.total_positions);
        println!("Number of choices: {}", self.choice_analyses.len());
        println!();

        for (i, choice_analysis) in self.choice_analyses.iter().enumerate() {
            println!(
                "Choice {}: {} positions analyzed",
                i, choice_analysis.positions_analyzed
            );

            if !choice_analysis.position_closeness.is_empty() {
                println!("  Closest positions (smallest logprob differences):");
                for (j, pos) in choice_analysis
                    .position_closeness
                    .iter()
                    .take(3)
                    .enumerate()
                {
                    let top_token = &pos.candidates[0].token;
                    let second_token = &pos.candidates[1].token;
                    println!(
                        "    {}: Stream pos {}, token pos {} - '{}' vs '{}' (diff: {:.4})",
                        j + 1,
                        pos.stream_position,
                        pos.token_position,
                        top_token,
                        second_token,
                        pos.logprob_difference
                    );
                }
            }
            println!();
        }
    }

    /// Get percentage of positions with close logprobs for a specific choice
    pub fn close_position_percentage_for_choice(&self, choice_index: usize, threshold: f32) -> f32 {
        if let Some(analysis) = self.choice_analyses.get(choice_index) {
            if analysis.positions_analyzed == 0 {
                return 0.0;
            }
            let close_count = analysis
                .position_closeness
                .iter()
                .filter(|pos| pos.logprob_difference <= threshold)
                .count();
            (close_count as f32 / analysis.positions_analyzed as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Check if multiple tokens are close (within threshold of each other)
    pub fn detect_multiple_close_tokens(
        &self,
        choice_index: usize,
        threshold: f32,
    ) -> Vec<MultipleCloseTokens> {
        let mut results = Vec::new();

        if let Some(analysis) = self.choice_analyses.get(choice_index) {
            for pos in &analysis.position_closeness {
                let close_tokens = self.count_close_tokens_at_position(pos, threshold);
                if close_tokens.close_count > 2 {
                    results.push(close_tokens);
                }
            }
        }

        results
    }

    /// Count how many tokens are close at a specific position
    fn count_close_tokens_at_position(
        &self,
        position: &PositionCloseness,
        threshold: f32,
    ) -> MultipleCloseTokens {
        let top_logprob = position.candidates[0].logprob;
        let mut close_count = 1; // Top token is always included
        let mut close_tokens = vec![position.candidates[0].clone()];

        for candidate in &position.candidates[1..] {
            let diff = top_logprob - candidate.logprob;
            if diff <= threshold {
                close_count += 1;
                close_tokens.push(candidate.clone());
            } else {
                break; // Since candidates are sorted, no need to check further
            }
        }

        let max_difference = if close_count > 1 {
            top_logprob - close_tokens.last().unwrap().logprob
        } else {
            0.0
        };

        MultipleCloseTokens {
            stream_position: position.stream_position,
            token_position: position.token_position,
            close_count,
            close_tokens,
            max_difference,
        }
    }
}

/// Information about multiple close tokens at a position
#[derive(Debug, Clone)]
pub struct MultipleCloseTokens {
    pub stream_position: usize,
    pub token_position: usize,
    pub close_count: usize,
    pub close_tokens: Vec<TokenLogprob>,
    pub max_difference: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::TimestampedResponse;
    use approx::assert_abs_diff_eq;
    use async_openai::types::{
        ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionStreamResponseDelta,
        ChatCompletionTokenLogprob, CreateChatCompletionStreamResponse, FinishReason, Role,
        TopLogprobs,
    };
    use std::time::Instant;

    const FLOAT_EPSILON: f32 = 1e-6;

    #[test]
    fn test_two_tokens_close() {
        let analysis = create_analysis_with_logprobs(vec![
            // Position with two very close tokens
            create_token_logprob(
                "hello",
                -0.1,
                vec![
                    ("world", -0.15), // diff = 0.05
                    ("there", -0.8),  // diff = 0.7
                ],
            ),
        ]);

        let close_positions = analysis.get_close_positions_for_choice(0, 0.1);
        assert_eq!(close_positions.len(), 1);
        assert_abs_diff_eq!(
            close_positions[0].logprob_difference,
            0.05,
            epsilon = FLOAT_EPSILON
        );

        let multiple_close = analysis.detect_multiple_close_tokens(0, 0.1);
        assert_eq!(multiple_close.len(), 0); // Only 2 tokens, so no "multiple" detected
    }

    #[test]
    fn test_three_tokens_close() {
        let analysis = create_analysis_with_logprobs(vec![
            // Position with three close tokens
            create_token_logprob(
                "hello",
                -0.1,
                vec![
                    ("world", -0.12), // diff = 0.02
                    ("there", -0.15), // diff = 0.05
                    ("friend", -0.6), // diff = 0.5
                ],
            ),
        ]);

        let close_positions = analysis.get_close_positions_for_choice(0, 0.1);
        assert_eq!(close_positions.len(), 1);
        assert_abs_diff_eq!(
            close_positions[0].logprob_difference,
            0.02,
            epsilon = FLOAT_EPSILON
        ); // Top 2 difference

        let multiple_close = analysis.detect_multiple_close_tokens(0, 0.1);
        assert_eq!(multiple_close.len(), 1);
        assert_eq!(multiple_close[0].close_count, 3);
        assert_abs_diff_eq!(
            multiple_close[0].max_difference,
            0.05,
            epsilon = FLOAT_EPSILON
        );
    }

    #[test]
    fn test_four_tokens_close() {
        let analysis = create_analysis_with_logprobs(vec![
            // Position with four close tokens
            create_token_logprob(
                "hello",
                -0.1,
                vec![
                    ("world", -0.11),  // diff = 0.01
                    ("there", -0.12),  // diff = 0.02
                    ("friend", -0.14), // diff = 0.04
                    ("buddy", -0.8),   // diff = 0.7
                ],
            ),
        ]);

        let close_positions = analysis.get_close_positions_for_choice(0, 0.1);
        assert_eq!(close_positions.len(), 1);
        assert_abs_diff_eq!(
            close_positions[0].logprob_difference,
            0.01,
            epsilon = FLOAT_EPSILON
        ); // Top 2 difference

        let multiple_close = analysis.detect_multiple_close_tokens(0, 0.05);
        assert_eq!(multiple_close.len(), 1);
        assert_eq!(multiple_close[0].close_count, 4);
        assert_abs_diff_eq!(
            multiple_close[0].max_difference,
            0.04,
            epsilon = FLOAT_EPSILON
        );
    }

    #[test]
    fn test_multiple_choices_analysis() {
        let analysis = create_analysis_with_multiple_choices(vec![
            // Choice 0: Close tokens
            vec![create_token_logprob("hello", -0.1, vec![("world", -0.15)])],
            // Choice 1: Very close tokens
            vec![create_token_logprob("hi", -0.2, vec![("there", -0.201)])],
        ]);

        assert_eq!(analysis.choice_analyses.len(), 2);

        // Check choice 0
        let choice0_close = analysis.get_close_positions_for_choice(0, 0.1);
        assert_eq!(choice0_close.len(), 1);
        assert_abs_diff_eq!(
            choice0_close[0].logprob_difference,
            0.05,
            epsilon = FLOAT_EPSILON
        );

        // Check choice 1
        let choice1_close = analysis.get_close_positions_for_choice(1, 0.1);
        assert_eq!(choice1_close.len(), 1);
        assert_abs_diff_eq!(
            choice1_close[0].logprob_difference,
            0.001,
            epsilon = FLOAT_EPSILON
        );

        // Choice 1 should be closer than choice 0
        let choice1_closest = analysis.get_closest_positions_for_choice(1, 1);
        assert!(choice1_closest[0].logprob_difference < choice0_close[0].logprob_difference);
    }

    #[test]
    fn test_edge_case_single_token() {
        let analysis = create_analysis_with_logprobs(vec![
            // Position with only one token (no alternatives)
            create_token_logprob("hello", -0.1, vec![]),
        ]);

        let close_positions = analysis.get_close_positions_for_choice(0, 1.0);
        assert_eq!(close_positions.len(), 0); // No close positions when only 1 token
    }

    #[test]
    fn test_threshold_filtering() {
        let analysis = create_analysis_with_logprobs(vec![
            create_token_logprob("token1", -0.1, vec![("token2", -0.15)]), // diff = 0.05
            create_token_logprob("token3", -0.2, vec![("token4", -0.4)]),  // diff = 0.2
        ]);

        // With threshold 0.1, only first position should be close
        let close_strict = analysis.get_close_positions_for_choice(0, 0.1);
        assert_eq!(close_strict.len(), 1);
        assert_abs_diff_eq!(
            close_strict[0].logprob_difference,
            0.05,
            epsilon = FLOAT_EPSILON
        );

        // With threshold 0.3, both positions should be close
        let close_permissive = analysis.get_close_positions_for_choice(0, 0.3);
        assert_eq!(close_permissive.len(), 2);

        // Check they're sorted by closeness
        assert!(close_permissive[0].logprob_difference < close_permissive[1].logprob_difference);
    }

    #[test]
    fn test_percentage_calculation() {
        let analysis = create_analysis_with_logprobs(vec![
            create_token_logprob("token1", -0.1, vec![("token2", -0.15)]), // diff = 0.05 - close
            create_token_logprob("token3", -0.2, vec![("token4", -0.4)]),  // diff = 0.2 - not close
            create_token_logprob("token5", -0.3, vec![("token6", -0.32)]), // diff = 0.02 - close
        ]);

        let percentage = analysis.close_position_percentage_for_choice(0, 0.1);
        assert!((percentage - 66.67).abs() < 0.01); // 2 out of 3 positions are close
    }

    // Helper functions for creating test data
    fn create_analysis_with_logprobs(
        token_logprobs: Vec<ChatCompletionTokenLogprob>,
    ) -> SensitivityAnalysis {
        let start_time = Instant::now();
        let response = create_mock_response_with_logprobs(token_logprobs);
        let responses = vec![TimestampedResponse::new(response, 0)];
        let recorded_stream = RecordedStream::new(responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        analyze_logprob_sensitivity(arc_stream, 0.1)
    }

    fn create_analysis_with_multiple_choices(
        choices_logprobs: Vec<Vec<ChatCompletionTokenLogprob>>,
    ) -> SensitivityAnalysis {
        let start_time = Instant::now();
        let response = create_mock_response_with_multiple_choices(choices_logprobs);
        let responses = vec![TimestampedResponse::new(response, 0)];
        let recorded_stream = RecordedStream::new(responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        analyze_logprob_sensitivity(arc_stream, 0.1)
    }

    fn create_token_logprob(
        token: &str,
        logprob: f32,
        top_logprobs: Vec<(&str, f32)>,
    ) -> ChatCompletionTokenLogprob {
        ChatCompletionTokenLogprob {
            token: token.to_string(),
            logprob,
            bytes: None,
            top_logprobs: top_logprobs
                .into_iter()
                .map(|(t, lp)| TopLogprobs {
                    token: t.to_string(),
                    logprob: lp,
                    bytes: None,
                })
                .collect(),
        }
    }

    fn create_mock_response_with_logprobs(
        token_logprobs: Vec<ChatCompletionTokenLogprob>,
    ) -> NvCreateChatCompletionStreamResponse {
        #[expect(deprecated)]
        let inner = CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some("test".to_string()),
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: Some(ChatChoiceLogprobs {
                    content: Some(token_logprobs),
                    refusal: None,
                }),
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
        choices_logprobs: Vec<Vec<ChatCompletionTokenLogprob>>,
    ) -> NvCreateChatCompletionStreamResponse {
        #[expect(deprecated)]
        let choices = choices_logprobs
            .into_iter()
            .enumerate()
            .map(|(i, token_logprobs)| ChatChoiceStream {
                index: i as u32,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some("test".to_string()),
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: Some(ChatChoiceLogprobs {
                    content: Some(token_logprobs),
                    refusal: None,
                }),
            })
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

    #[test]
    fn test_sensitivity_analysis() {
        let start_time = Instant::now();
        let responses = vec![TimestampedResponse::new(create_mock_response(), 0)];

        let recorded_stream = RecordedStream::new(responses, start_time, Instant::now());
        let arc_stream = Arc::new(recorded_stream);

        let analysis = analyze_logprob_sensitivity(arc_stream, 0.5);
        // Basic validation that analysis was created
        assert_eq!(analysis.total_positions, 1);
        assert!(analysis.close_position_percentage_for_choice(0, 0.5) >= 0.0);
    }

    #[test]
    fn test_extract_logprobs_by_choice_empty() {
        let response = create_mock_response();
        let logprobs = response.extract_logprobs_by_choice();
        assert!(logprobs.is_empty() || logprobs[0].is_empty());
    }

    #[test]
    fn test_position_closeness_ordering() {
        let analysis = create_analysis_with_logprobs(vec![
            create_token_logprob("far", -0.1, vec![("alt", -0.8)]), // diff = 0.7
            create_token_logprob("close", -0.1, vec![("alt", -0.12)]), // diff = 0.02
            create_token_logprob("medium", -0.1, vec![("alt", -0.3)]), // diff = 0.2
        ]);

        let positions = &analysis.choice_analyses[0].position_closeness;
        assert_eq!(positions.len(), 3);

        // Should be sorted by closeness (smallest difference first)
        assert!(positions[0].logprob_difference <= positions[1].logprob_difference);
        assert!(positions[1].logprob_difference <= positions[2].logprob_difference);

        // Check actual values
        assert_abs_diff_eq!(
            positions[0].logprob_difference,
            0.02,
            epsilon = FLOAT_EPSILON
        );
        assert_abs_diff_eq!(
            positions[1].logprob_difference,
            0.2,
            epsilon = FLOAT_EPSILON
        );
        assert_abs_diff_eq!(
            positions[2].logprob_difference,
            0.7,
            epsilon = FLOAT_EPSILON
        );
    }

    #[test]
    fn test_multiple_close_tokens_edge_cases() {
        // Test with exactly 3 close tokens
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob(
            "token",
            -0.1,
            vec![
                ("alt1", -0.105), // diff = 0.005
                ("alt2", -0.108), // diff = 0.008
                ("alt3", -0.5),   // diff = 0.4 (not close)
            ],
        )]);

        let multiple_close = analysis.detect_multiple_close_tokens(0, 0.01);
        assert_eq!(multiple_close.len(), 1);
        assert_eq!(multiple_close[0].close_count, 3);
    }

    #[test]
    fn test_choice_analysis_independence() {
        let analysis = create_analysis_with_multiple_choices(vec![
            // Choice 0: 2 positions, 1 close
            vec![
                create_token_logprob("token1", -0.1, vec![("alt1", -0.15)]), // diff = 0.05
                create_token_logprob("token2", -0.1, vec![("alt2", -0.8)]),  // diff = 0.7
            ],
            // Choice 1: 1 position, close
            vec![
                create_token_logprob("token3", -0.1, vec![("alt3", -0.101)]), // diff = 0.001
            ],
        ]);

        assert_eq!(analysis.choice_analyses.len(), 2);
        assert_eq!(analysis.choice_analyses[0].positions_analyzed, 2);
        assert_eq!(analysis.choice_analyses[1].positions_analyzed, 1);

        // Check independence - each choice should have different closeness patterns
        let choice0_close = analysis.get_close_positions_for_choice(0, 0.1);
        let choice1_close = analysis.get_close_positions_for_choice(1, 0.1);

        assert_eq!(choice0_close.len(), 1);
        assert_eq!(choice1_close.len(), 1);

        // Choice 1 should be closer
        assert!(choice1_close[0].logprob_difference < choice0_close[0].logprob_difference);
    }

    #[test]
    fn test_get_closest_positions_boundary() {
        let analysis = create_analysis_with_logprobs(vec![
            create_token_logprob("token1", -0.1, vec![("alt1", -0.15)]),
            create_token_logprob("token2", -0.1, vec![("alt2", -0.25)]),
        ]);

        // Request more positions than available
        let closest = analysis.get_closest_positions_for_choice(0, 10);
        assert_eq!(closest.len(), 2);

        // Request exactly the number available
        let closest = analysis.get_closest_positions_for_choice(0, 2);
        assert_eq!(closest.len(), 2);

        // Request fewer
        let closest = analysis.get_closest_positions_for_choice(0, 1);
        assert_eq!(closest.len(), 1);
    }

    #[test]
    fn test_zero_threshold() {
        let analysis = create_analysis_with_logprobs(vec![
            create_token_logprob("token", -0.1, vec![("alt", -0.1)]), // diff = 0.0
        ]);

        let close_positions = analysis.get_close_positions_for_choice(0, 0.0);
        assert_eq!(close_positions.len(), 1);
        assert_abs_diff_eq!(
            close_positions[0].logprob_difference,
            0.0,
            epsilon = FLOAT_EPSILON
        );
    }

    #[test]
    fn test_nonexistent_choice() {
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob(
            "token",
            -0.1,
            vec![("alt", -0.15)],
        )]);

        // Request analysis for non-existent choice
        let close_positions = analysis.get_close_positions_for_choice(5, 0.1);
        assert!(close_positions.is_empty());

        let closest = analysis.get_closest_positions_for_choice(5, 3);
        assert!(closest.is_empty());

        let percentage = analysis.close_position_percentage_for_choice(5, 0.1);
        assert_eq!(percentage, 0.0);
    }

    #[test]
    fn test_logprob_extractor_with_missing_data() {
        // Test with choice that has no logprobs
        #[expect(deprecated)]
        let inner = CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some("test".to_string()),
                    function_call: None,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: None, // No logprobs
            }],
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        };

        let response = NvCreateChatCompletionStreamResponse { inner };
        let logprobs = response.extract_logprobs_by_choice();
        assert_eq!(logprobs.len(), 1);
        assert!(logprobs[0].is_empty());
    }

    #[test]
    fn test_print_summary_no_panic() {
        let analysis = create_analysis_with_logprobs(vec![create_token_logprob(
            "token",
            -0.1,
            vec![("alt", -0.15)],
        )]);

        // Should not panic when printing summary
        analysis.print_summary();
    }

    fn create_mock_response() -> NvCreateChatCompletionStreamResponse {
        // Create a mock response for testing
        // In practice, this would have real logprobs data
        use async_openai::types::CreateChatCompletionStreamResponse;

        let inner = CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            choices: vec![],
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
