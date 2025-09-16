// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use async_stream::stream;
use dynamo_async_openai::types::{
    ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    FinishReason, FunctionCallStream, Role,
};

use dynamo_parsers::tool_calling::{detect_tool_call_start, try_tool_call_parse_aggregate};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use super::NvCreateChatCompletionStreamResponse;

/// A stream transformer that can "jail" tokens based on configurable start/end sequences
/// When jailed, tokens are accumulated rather than yielded immediately
/// When the jail ends (via end sequence or stream completion), accumulated content is processed and released
pub struct JailedStream {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
}

impl JailedStream {
    /// Create a new builder for configuring a JailedStream
    pub fn builder() -> JailedStreamBuilder {
        JailedStreamBuilder::new()
    }

    /// Apply the jail transformation to a stream of chat completion responses
    /// Consumes self and returns the transformed stream
    pub fn apply<S>(
        self,
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        // Use the stream! macro for cleaner async stream processing
        stream! {
            // State variables
            let mut is_jailed = false;
            let mut accumulated_content: HashMap<u32, String> = HashMap::new();
            let mut last_response_metadata: Option<NvCreateChatCompletionStreamResponse> = None;
            let mut buffered_content = String::new();

            // Pin the stream for iteration (stack pinning is more efficient)
            tokio::pin!(stream);

            // Process each item in the stream
            while let Some(response) = stream.next().await {
                // Handle non-jailed state
                if !is_jailed {
                    if let Some(chat_response) = response.data.as_ref() {
                        // Check if we should jail based on content
                        if let Some(choice) = chat_response.choices.first()
                            && let Some(ref content) = choice.delta.content
                        {
                                // Check for jail start - two paths (evaluate both, not if/else)
                                // Path 1: Check configured start sequences
                                let sequence_match = !self.jail_start_sequences.is_empty()
                                    && self.jail_start_sequences.iter().any(|seq| content.contains(seq));

                                // Path 2: Check for tool call start pattern
                                let tool_call_match = self.tool_call_parser.is_some()
                                    && detect_tool_call_start(content, self.tool_call_parser.as_deref())
                                        .unwrap_or(false);

                                // Jail if either condition is true
                                let should_jail = sequence_match || tool_call_match;

                                if should_jail {
                                    tracing::debug!(
                                        "Jail triggered (sequence: {}, tool_call: {}), starting accumulation",
                                        sequence_match, tool_call_match
                                    );
                                    is_jailed = true;

                                    // Store metadata only when we actually jail
                                    last_response_metadata = response.data.clone();

                                    // Start accumulating for this choice
                                    accumulated_content.insert(choice.index, content.clone());
                                    buffered_content = content.clone();

                                    // Don't yield anything while jailed - just continue accumulating
                                    continue;
                            }
                        }
                    }

                    // Not jailed, yield as-is
                    yield response;
                } else {
                    // We're jailed - accumulate content
                    if let Some(ref chat_response) = response.data {
                        for choice in &chat_response.choices {
                            if let Some(ref content) = choice.delta.content
                                && !content.is_empty()
                            {
                                    // Accumulate content
                                    accumulated_content
                                        .entry(choice.index)
                                        .or_default()
                                        .push_str(content);
                                    buffered_content.push_str(content);

                                    // Check for jail end - two paths
                                    // Path 1: End sequence detected
                                    let sequence_end = !self.jail_end_sequences.is_empty()
                                        && self.jail_end_sequences.iter().any(|seq| buffered_content.contains(seq));

                                    // Path 2: Complete tool call(s) can be parsed (early exit)
                                    let early_exit = self.should_exit_jail_early(&buffered_content);

                                    // Unjail if either condition is true
                                    let should_unjail = sequence_end || early_exit;

                                    if should_unjail {
                                        tracing::debug!(
                                            "Jail exit detected (sequence: {}, early: {}), releasing accumulated content",
                                            sequence_end, early_exit
                                        );
                                        is_jailed = false;

                                        // Process and release accumulated content
                                        if let Some(base_response) = last_response_metadata.take() {
                                            let final_response = self.create_unjailed_response(
                                                base_response,
                                                &accumulated_content,
                                            );
                                            accumulated_content.clear();
                                            buffered_content.clear();
                                            yield final_response;
                                            continue;
                                        }
                                    }

                                    // Still jailed, just continue accumulating without yielding
                            }
                        }
                    }
                }
            }

            // Stream ended - if we're still jailed, release accumulated content
            if is_jailed && !accumulated_content.is_empty() {
                tracing::debug!("Stream ended while jailed, releasing accumulated content");
                if let Some(base_response) = last_response_metadata.take() {
                    let final_response = self.create_unjailed_response(
                        base_response,
                        &accumulated_content,
                    );
                    yield final_response;
                }
            }
        }
    }

    /// Check if accumulated content contains complete tool calls that can be parsed
    /// Returns true if we should exit the jail early
    fn should_exit_jail_early(&self, accumulated: &str) -> bool {
        if let Some(ref parser) = self.tool_call_parser {
            // Try to parse - if successful and we have complete tool calls, exit early
            if let Ok((tool_calls, _)) = try_tool_call_parse_aggregate(accumulated, Some(parser)) {
                return !tool_calls.is_empty();
            }
        }
        false
    }

    /// Create a response with accumulated content, potentially parsing tool calls
    fn create_unjailed_response(
        &self,
        mut base_response: NvCreateChatCompletionStreamResponse,
        accumulated_content: &HashMap<u32, String>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        // Try to parse tool calls from accumulated content
        for (choice_index, accumulated_text) in accumulated_content {
            if let Ok((tool_calls, normal_text)) =
                try_tool_call_parse_aggregate(accumulated_text, self.tool_call_parser.as_deref())
            {
                if !tool_calls.is_empty() {
                    tracing::debug!(
                        "Parsed {} tool calls from accumulated content",
                        tool_calls.len()
                    );

                    // Convert to streaming format
                    let tool_call_chunks: Vec<ChatCompletionMessageToolCallChunk> = tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(idx, tool_call)| ChatCompletionMessageToolCallChunk {
                            index: idx as u32,
                            id: Some(tool_call.id),
                            r#type: Some(tool_call.r#type),
                            function: Some(FunctionCallStream {
                                name: Some(tool_call.function.name),
                                arguments: Some(tool_call.function.arguments),
                            }),
                        })
                        .collect();

                    // Create choice with tool calls
                    #[allow(deprecated)]
                    let final_choice = ChatChoiceStream {
                        index: *choice_index,
                        delta: ChatCompletionStreamResponseDelta {
                            role: Some(Role::Assistant),
                            content: normal_text.filter(|t| !t.is_empty()),
                            tool_calls: Some(tool_call_chunks),
                            function_call: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: Some(FinishReason::ToolCalls),
                        logprobs: None,
                    };

                    base_response.choices = vec![final_choice];
                } else {
                    // No tool calls found, return accumulated text as normal content
                    if let Some(choice) = base_response.choices.get_mut(*choice_index as usize) {
                        choice.delta.content = Some(accumulated_text.clone());
                    }
                }
            } else {
                // Parse failed, return accumulated text as normal content
                if let Some(choice) = base_response.choices.get_mut(*choice_index as usize) {
                    choice.delta.content = Some(accumulated_text.clone());
                }
            }
        }

        Annotated {
            data: Some(base_response),
            id: None,
            event: None,
            comment: None,
        }
    }
}

/// Builder for configuring a JailedStream
pub struct JailedStreamBuilder {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
}

impl JailedStreamBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            jail_start_sequences: Vec::new(),
            jail_end_sequences: Vec::new(),
            tool_call_parser: None,
        }
    }

    /// Add a sequence that triggers jailing when detected
    pub fn jail_start_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.jail_start_sequences.push(sequence.into());
        self
    }

    /// Add multiple sequences that trigger jailing when detected
    pub fn jail_start_sequences(
        mut self,
        sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.jail_start_sequences
            .extend(sequences.into_iter().map(Into::into));
        self
    }

    /// Add a sequence that ends jailing when detected
    pub fn jail_end_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.jail_end_sequences.push(sequence.into());
        self
    }

    /// Add multiple sequences that end jailing when detected
    pub fn jail_end_sequences(
        mut self,
        sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.jail_end_sequences
            .extend(sequences.into_iter().map(Into::into));
        self
    }

    /// Set the tool call parser to use for detection and parsing
    pub fn tool_call_parser(mut self, parser: impl Into<String>) -> Self {
        self.tool_call_parser = Some(parser.into());
        self
    }

    /// Build the configured JailedStream
    pub fn build(self) -> JailedStream {
        JailedStream {
            jail_start_sequences: self.jail_start_sequences,
            jail_end_sequences: self.jail_end_sequences,
            tool_call_parser: self.tool_call_parser,
        }
    }
}

impl Default for JailedStreamBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use futures::stream;

    // Test utilities module - shared test infrastructure
    pub(crate) mod test_utils {
        use super::*;

        /// Helper function to create a mock chat response chunk
        pub fn create_mock_response_chunk(
            content: String,
            index: u32,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let choice = ChatChoiceStream {
                index,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: Some(content),
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            };

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            };

            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
            }
        }

        /// Helper function to create a final response chunk with finish reason
        pub fn create_final_response_chunk(
            index: u32,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let choice = ChatChoiceStream {
                index,
                delta: ChatCompletionStreamResponseDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            };

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            };

            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
            }
        }
    }

    use test_utils::*;

    #[tokio::test]
    async fn test_jailed_stream_with_start_end_sequences() {
        // Create chunks with jail start/end markers
        let chunks = vec![
            create_mock_response_chunk("Hello ".to_string(), 0),
            create_mock_response_chunk("<jail>".to_string(), 0),
            create_mock_response_chunk("This is jailed ".to_string(), 0),
            create_mock_response_chunk("content".to_string(), 0),
            create_mock_response_chunk("</jail>".to_string(), 0),
            create_mock_response_chunk(" World".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with start/end sequences
        let jail = JailedStream::builder()
            .jail_start_sequence("<jail>")
            .jail_end_sequence("</jail>")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // We should only get 3 chunks now:
        // 1. "Hello " (before jail)
        // 2. Accumulated jailed content when jail ends
        // 3. " World" (after jail)
        assert_eq!(results.len(), 3);

        // First chunk should pass through
        assert_eq!(
            results[0].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some("Hello ")
        );

        // When jail ends, accumulated content should be released
        let unjailed_content = &results[1].data.as_ref().unwrap().choices[0].delta.content;
        assert!(unjailed_content.is_some());
        assert!(
            unjailed_content
                .as_ref()
                .unwrap()
                .contains("<jail>This is jailed content</jail>")
        );

        // Last chunk should pass through normally
        assert_eq!(
            results[2].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some(" World")
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_with_tool_calls() {
        // Create chunks representing a tool call
        let chunks = vec![
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"SF\"}}]".to_string(),
                0,
            ),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with tool call parser
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have jailed the content and parsed tool calls at the end
        assert!(!results.is_empty());

        // Check if tool calls were parsed
        if let Some(last_result) = results.last()
            && let Some(ref response_data) = last_result.data
            && let Some(ref tool_calls) = response_data.choices[0].delta.tool_calls
        {
            assert!(!tool_calls.is_empty());
            assert_eq!(
                tool_calls[0].function.as_ref().unwrap().name.as_deref(),
                Some("get_weather")
            );
        }
    }

    #[tokio::test]
    async fn test_jailed_stream_dual_entry_paths() {
        // Test that BOTH sequence AND tool call detection can trigger jail
        let chunks = vec![
            create_mock_response_chunk("Normal text ".to_string(), 0),
            create_mock_response_chunk("<jail><TOOLCALL>".to_string(), 0), // Both triggers
            create_mock_response_chunk("Jailed content".to_string(), 0),
            create_mock_response_chunk("</jail>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Configure with both sequences AND tool call parser
        let jail = JailedStream::builder()
            .jail_start_sequence("<jail>")
            .jail_end_sequence("</jail>")
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // First chunk should pass through
        assert_eq!(
            results[0].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some("Normal text ")
        );

        // Jail should trigger and accumulate
        assert!(results.len() >= 2);
    }

    #[tokio::test]
    async fn test_jailed_stream_early_exit() {
        // Test early exit when complete tool call is detected
        let chunks = vec![
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"test\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {}}]".to_string(), 0),
            create_mock_response_chunk("</TOOLCALL>More text".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should detect complete tool call and exit early
        assert!(!results.is_empty());

        // Check if tool calls were parsed
        let has_tool_calls = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        });
        assert!(
            has_tool_calls,
            "Should have parsed tool calls with early exit"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_no_jailing() {
        // Create normal content chunks
        let chunks = vec![
            create_mock_response_chunk("Hello ".to_string(), 0),
            create_mock_response_chunk("World".to_string(), 0),
            create_final_response_chunk(0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with sequences that won't match
        let jail = JailedStream::builder()
            .jail_start_sequence("<NOTPRESENT>")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // All chunks should pass through unchanged
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some("Hello ")
        );
        assert_eq!(
            results[1].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some("World")
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_hermes_parser() {
        // Test Hermes parser with <tool_call> markers
        let chunks = vec![
            create_mock_response_chunk("I'll help you with that. ".to_string(), 0),
            create_mock_response_chunk("<tool_call>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"search_web\", ".to_string(), 0),
            create_mock_response_chunk(
                "\"arguments\": {\"query\": \"weather today\"}}".to_string(),
                0,
            ),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
            create_mock_response_chunk(" Let me search for that.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Hermes parser
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have initial text, tool call result, and final text
        assert!(!results.is_empty());

        // Check if tool calls were parsed correctly
        let has_tool_calls = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        });
        assert!(has_tool_calls, "Should have parsed Hermes tool calls");

        // Check that we have the search_web function
        let has_search_web = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tcs| {
                    tcs.iter().any(|tc| {
                        tc.function
                            .as_ref()
                            .and_then(|f| f.name.as_deref())
                            .map(|name| name == "search_web")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        });
        assert!(has_search_web, "Should have parsed search_web function");
    }

    #[tokio::test]
    async fn test_jailed_stream_mistral_parser() {
        // Test Mistral parser with [{ pattern
        let chunks = vec![
            create_mock_response_chunk("Sure, I can help. ".to_string(), 0),
            create_mock_response_chunk("[{".to_string(), 0),
            create_mock_response_chunk("\"name\": \"calculate\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"expression\": \"2+2\"}".to_string(), 0),
            create_mock_response_chunk("}]".to_string(), 0),
            create_mock_response_chunk(" The calculation is done.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Mistral parser
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have initial text, tool call result, and final text
        assert!(!results.is_empty());

        // Check if tool calls were parsed correctly
        let has_tool_calls = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        });
        assert!(has_tool_calls, "Should have parsed Mistral tool calls");

        // Check that we have the calculate function
        let has_calculate = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tcs| {
                    tcs.iter().any(|tc| {
                        tc.function
                            .as_ref()
                            .and_then(|f| f.name.as_deref())
                            .map(|name| name == "calculate")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        });
        assert!(has_calculate, "Should have parsed calculate function");
    }

    #[tokio::test]
    async fn test_jailed_stream_mistral_parser_with_tool_calls_marker() {
        // Test Mistral parser with [TOOL_CALLS] marker
        let chunks = vec![
            create_mock_response_chunk("Let me check that for you. ".to_string(), 0),
            create_mock_response_chunk("[TOOL_CALLS]".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"get_time\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"timezone\": \"UTC\"}}]".to_string(), 0),
            create_mock_response_chunk(" Here's the time.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Mistral parser
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have initial text, tool call result, and final text
        assert!(!results.is_empty());

        // Check if tool calls were parsed correctly
        let has_tool_calls = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        });
        assert!(
            has_tool_calls,
            "Should have parsed Mistral [TOOL_CALLS] format"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_phi4_parser() {
        // Test Phi4 parser with functools[ pattern
        let chunks = vec![
            create_mock_response_chunk("I'll analyze this data. ".to_string(), 0),
            create_mock_response_chunk("functools[".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"analyze_data\", ".to_string(), 0),
            create_mock_response_chunk(
                "\"arguments\": {\"dataset\": \"sales_data\"}}".to_string(),
                0,
            ),
            create_mock_response_chunk("]".to_string(), 0),
            create_mock_response_chunk(" Analysis complete.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Phi4 parser
        let jail = JailedStream::builder().tool_call_parser("phi4").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have initial text, tool call result, and final text
        assert!(!results.is_empty());

        // Check if tool calls were parsed correctly
        let has_tool_calls = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        });
        assert!(has_tool_calls, "Should have parsed Phi4 tool calls");

        // Check that we have the analyze_data function
        let has_analyze_data = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tcs| {
                    tcs.iter().any(|tc| {
                        tc.function
                            .as_ref()
                            .and_then(|f| f.name.as_deref())
                            .map(|name| name == "analyze_data")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        });
        assert!(has_analyze_data, "Should have parsed analyze_data function");
    }

    #[tokio::test]
    async fn test_jailed_stream_llama3_json_parser() {
        // Test llama3_json parser with <|python_tag|> pattern
        let chunks = vec![
            create_mock_response_chunk("Let me run some code. ".to_string(), 0),
            create_mock_response_chunk("<|python_tag|>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"execute_code\", ".to_string(), 0),
            create_mock_response_chunk(
                "\"arguments\": {\"code\": \"print('Hello')\"}}".to_string(),
                0,
            ),
            create_mock_response_chunk(" Done executing.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with llama3_json parser
        let jail = JailedStream::builder()
            .tool_call_parser("llama3_json")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have initial text, tool call result, and final text
        assert!(!results.is_empty());

        // Check if tool calls were parsed correctly
        let has_tool_calls = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        });
        assert!(has_tool_calls, "Should have parsed llama3_json tool calls");

        // Check that we have the execute_code function
        let has_execute_code = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tcs| {
                    tcs.iter().any(|tc| {
                        tc.function
                            .as_ref()
                            .and_then(|f| f.name.as_deref())
                            .map(|name| name == "execute_code")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        });
        assert!(has_execute_code, "Should have parsed execute_code function");
    }

    #[tokio::test]
    async fn test_jailed_stream_false_positive_json() {
        // Test with text that looks like it might contain tool calls but doesn't match parser patterns
        let chunks = vec![
            create_mock_response_chunk("I can explain JSON format. ".to_string(), 0),
            create_mock_response_chunk("Here's an example: { \"key\": \"value\" }".to_string(), 0),
            create_mock_response_chunk(" is a simple JSON object. ".to_string(), 0),
            create_mock_response_chunk("Hope that helps!".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with mistral parser (which specifically looks for [{ or [TOOL_CALLS] patterns)
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should pass through all chunks since no mistral-specific patterns are present
        assert!(!results.is_empty());

        // Verify no tool calls were detected
        let has_tool_calls = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        });
        assert!(
            !has_tool_calls,
            "Should not detect tool calls in JSON explanation text"
        );

        // Verify content is preserved correctly
        let has_json_content = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .map(|content| {
                    content.contains("JSON format") || content.contains("simple JSON object")
                })
                .unwrap_or(false)
        });
        assert!(has_json_content, "Should preserve JSON explanation content");
    }

    #[tokio::test]
    async fn test_jailed_stream_malformed_tool_call() {
        // Test with malformed JSON in tool calls
        let chunks = vec![
            create_mock_response_chunk("Let me call a function. ".to_string(), 0),
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"broken_func\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"param\": incomplete".to_string(), 0), // Malformed JSON
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(" Function call attempt finished.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with nemotron_deci parser
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should not panic and should handle malformed JSON gracefully
        assert!(!results.is_empty());

        // Should still process the content even if JSON is malformed
        let has_content = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .map(|content| !content.is_empty())
                .unwrap_or(false)
        });
        assert!(
            has_content,
            "Should still have content even with malformed JSON"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_partial_tool_call() {
        // Test stream that ends mid-tool call
        let chunks = vec![
            create_mock_response_chunk("Starting function call. ".to_string(), 0),
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"incomplete_func\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {".to_string(), 0),
            // Stream ends abruptly without closing the tool call
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with nemotron_deci parser
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should handle partial tool call gracefully
        assert!(!results.is_empty());

        // First chunk should pass through
        assert!(
            results
                .first()
                .and_then(|r| r.data.as_ref())
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .map(|content| content.contains("Starting function call"))
                .unwrap_or(false)
        );

        // Should release accumulated content when stream ends
        let has_accumulated_content = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .map(|content| {
                    content.contains("<TOOLCALL>") || content.contains("incomplete_func")
                })
                .unwrap_or(false)
        });
        assert!(
            has_accumulated_content,
            "Should release accumulated partial tool call content"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_empty_stream() {
        // Test with completely empty input stream
        let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = vec![];
        let input_stream = stream::iter(chunks);

        // Create JailedStream
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .jail_start_sequence("<jail>")
            .jail_end_sequence("</jail>")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should handle empty stream gracefully without panicking
        assert!(results.is_empty(), "Empty stream should produce no results");
    }

    #[tokio::test]
    async fn test_jailed_stream_multiple_tool_calls() {
        // Test multiple sequential tool calls
        let chunks = vec![
            create_mock_response_chunk("I'll help with multiple tasks. ".to_string(), 0),
            // First tool call
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}]".to_string(),
                0,
            ),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(" Now let me get the time. ".to_string(), 0),
            // Second tool call
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"EST\"}}]".to_string(),
                0,
            ),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(" Both tasks completed!".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have processed multiple tool calls
        assert!(!results.is_empty());

        // Count the number of tool calls detected
        let tool_call_count = results
            .iter()
            .filter(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .and_then(|c| c.delta.tool_calls.as_ref())
                    .map(|tc| !tc.is_empty())
                    .unwrap_or(false)
            })
            .count();

        assert!(tool_call_count > 0, "Should detect multiple tool calls");

        // Check that both function names are present in results
        let has_weather = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tcs| {
                    tcs.iter().any(|tc| {
                        tc.function
                            .as_ref()
                            .and_then(|f| f.name.as_deref())
                            .map(|name| name == "get_weather")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        });

        let has_time = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tcs| {
                    tcs.iter().any(|tc| {
                        tc.function
                            .as_ref()
                            .and_then(|f| f.name.as_deref())
                            .map(|name| name == "get_time")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        });

        assert!(has_weather, "Should have get_weather function");
        assert!(has_time, "Should have get_time function");
    }

    #[tokio::test]
    async fn test_jailed_stream_tool_call_across_many_chunks() {
        // Split a tool call across many small chunks
        let chunks = vec![
            create_mock_response_chunk("I'll process your request. ".to_string(), 0),
            create_mock_response_chunk("<".to_string(), 0),
            create_mock_response_chunk("T".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("C".to_string(), 0),
            create_mock_response_chunk("A".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk(">".to_string(), 0),
            create_mock_response_chunk("[".to_string(), 0),
            create_mock_response_chunk("{".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk("n".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("m".to_string(), 0),
            create_mock_response_chunk("e".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk(":".to_string(), 0),
            create_mock_response_chunk(" ".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk("p".to_string(), 0),
            create_mock_response_chunk("r".to_string(), 0),
            create_mock_response_chunk("o".to_string(), 0),
            create_mock_response_chunk("c".to_string(), 0),
            create_mock_response_chunk("e".to_string(), 0),
            create_mock_response_chunk("s".to_string(), 0),
            create_mock_response_chunk("s".to_string(), 0),
            create_mock_response_chunk("_".to_string(), 0),
            create_mock_response_chunk("d".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("t".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk(",".to_string(), 0),
            create_mock_response_chunk(" ".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("r".to_string(), 0),
            create_mock_response_chunk("g".to_string(), 0),
            create_mock_response_chunk("u".to_string(), 0),
            create_mock_response_chunk("m".to_string(), 0),
            create_mock_response_chunk("e".to_string(), 0),
            create_mock_response_chunk("n".to_string(), 0),
            create_mock_response_chunk("t".to_string(), 0),
            create_mock_response_chunk("s".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk(":".to_string(), 0),
            create_mock_response_chunk(" ".to_string(), 0),
            create_mock_response_chunk("{".to_string(), 0),
            create_mock_response_chunk("}".to_string(), 0),
            create_mock_response_chunk("}".to_string(), 0),
            create_mock_response_chunk("]".to_string(), 0),
            create_mock_response_chunk("<".to_string(), 0),
            create_mock_response_chunk("/".to_string(), 0),
            create_mock_response_chunk("T".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("C".to_string(), 0),
            create_mock_response_chunk("A".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk(">".to_string(), 0),
            create_mock_response_chunk(" Processing complete!".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should handle tool call split across many chunks
        assert!(!results.is_empty());

        // Should detect the tool call despite fragmentation
        let has_tool_calls = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        });
        assert!(
            has_tool_calls,
            "Should detect tool call across many fragments"
        );

        // Should have the process_data function
        let has_process_data = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tcs| {
                    tcs.iter().any(|tc| {
                        tc.function
                            .as_ref()
                            .and_then(|f| f.name.as_deref())
                            .map(|name| name == "process_data")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        });
        assert!(has_process_data, "Should have parsed process_data function");

        // Verify initial and final text are preserved
        let has_initial_text = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .map(|content| content.contains("I'll process your request"))
                .unwrap_or(false)
        });
        assert!(has_initial_text, "Should preserve initial text");

        let has_final_text = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .map(|content| content.contains("Processing complete"))
                .unwrap_or(false)
        });
        assert!(has_final_text, "Should preserve final text");
    }
}
