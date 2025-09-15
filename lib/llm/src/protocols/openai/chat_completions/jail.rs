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
                                // Check for jail start sequences
                                let should_jail = if !self.jail_start_sequences.is_empty() {
                                    // Check configured start sequences
                                    self.jail_start_sequences.iter().any(|seq| content.contains(seq))
                                } else {
                                    // Fall back to tool call detection if no sequences configured
                                    detect_tool_call_start(content, self.tool_call_parser.as_deref())
                                        .unwrap_or(false)
                                };

                                if should_jail {
                                    tracing::debug!("Jail triggered, starting accumulation");
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

                                    // Check for jail end sequences
                                    let should_unjail = if !self.jail_end_sequences.is_empty() {
                                        self.jail_end_sequences.iter().any(|seq| buffered_content.contains(seq))
                                    } else {
                                        false
                                    };

                                    if should_unjail {
                                        tracing::debug!("Jail end sequence detected, releasing accumulated content");
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
}
