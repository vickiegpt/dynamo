// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use async_stream::stream;
use dynamo_async_openai::types::{
    ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    FinishReason, FunctionCallStream, Role,
};

use dynamo_parsers::tool_calling::{detect_tool_call_start, try_tool_call_parse_aggregate};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use super::NvCreateChatCompletionStreamResponse;

/// State tracking for an individual choice during jail processing
#[derive(Debug, Clone)]
struct ChoiceJailState {
    /// The choice index (0, 1, 2, ...)
    index: u32,
    /// Whether this choice is currently jailed
    is_jailed: bool,
    /// Accumulated content for this choice while jailed
    accumulated_content: String,
}

impl ChoiceJailState {
    /// Create a new jail state for a choice
    fn new(index: u32) -> Self {
        Self {
            index,
            is_jailed: false,
            accumulated_content: String::new(),
        }
    }

    /// Start jailing this choice with initial content
    fn start_jail(&mut self, initial_content: &str) {
        self.is_jailed = true;
        self.accumulated_content = initial_content.to_string();
    }

    /// Add content to this choice's accumulation
    fn accumulate(&mut self, content: &str) {
        if self.is_jailed {
            self.accumulated_content.push_str(content);
        }
    }

    /// End jailing and return the accumulated content
    fn end_jail(&mut self) -> String {
        self.is_jailed = false;
        std::mem::take(&mut self.accumulated_content)
    }

    /// Clear accumulated content without ending jail
    fn clear(&mut self) {
        self.accumulated_content.clear();
    }
}

/// Collection of choice jail states with deterministic ordering
#[derive(Debug, Clone)]
struct ChoiceJailStateCollection {
    /// Vec of states, always kept sorted by choice index for deterministic iteration
    states: Vec<ChoiceJailState>,
}

impl ChoiceJailStateCollection {
    /// Create a new empty collection
    fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Get or create state for a choice index
    fn get_or_create_state(&mut self, index: u32) -> &mut ChoiceJailState {
        // Find the position where this index should be
        match self.states.binary_search_by_key(&index, |s| s.index) {
            Ok(pos) => {
                // Found existing state
                &mut self.states[pos]
            }
            Err(insert_pos) => {
                // Need to create new state
                let new_state = ChoiceJailState::new(index);
                self.states.insert(insert_pos, new_state);
                &mut self.states[insert_pos]
            }
        }
    }

    /// Get state for a choice index if it exists
    fn get_state(&self, index: u32) -> Option<&ChoiceJailState> {
        self.states.iter().find(|s| s.index == index)
    }

    /// Get mutable state for a choice index if it exists
    fn get_state_mut(&mut self, index: u32) -> Option<&mut ChoiceJailState> {
        self.states.iter_mut().find(|s| s.index == index)
    }

    /// Check if any choice is jailed
    fn has_jailed_choices(&self) -> bool {
        self.states.iter().any(|s| s.is_jailed)
    }

    /// Get all jailed states in deterministic order (sorted by index)
    fn jailed_states(&self) -> impl Iterator<Item = &ChoiceJailState> {
        self.states.iter().filter(|s| s.is_jailed)
    }

    /// Get all jailed states mutably in deterministic order
    fn jailed_states_mut(&mut self) -> impl Iterator<Item = &mut ChoiceJailState> {
        self.states.iter_mut().filter(|s| s.is_jailed)
    }

    /// Clear all states
    fn clear(&mut self) {
        self.states.clear();
    }

    /// Create HashMap compatible with existing create_unjailed_response method
    /// TODO: Remove this once we refactor create_unjailed_response to use the new structure
    fn to_hashmap(&self) -> HashMap<u32, String> {
        self.states
            .iter()
            .filter(|s| s.is_jailed && !s.accumulated_content.is_empty())
            .map(|s| (s.index, s.accumulated_content.clone()))
            .collect()
    }
}

/// Emission mode for handling multiple choices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmissionMode {
    /// Pack multiple choices in the same chunk (default, matches original behavior)
    Packed,
    /// Emit one choice per chunk for OpenAI compatibility
    SingleChoicePerChunk,
}

impl Default for EmissionMode {
    fn default() -> Self {
        Self::Packed
    }
}

/// A stream transformer that can "jail" tokens based on configurable start/end sequences
/// When jailed, tokens are accumulated rather than yielded immediately
/// When the jail ends (via end sequence or stream completion), accumulated content is processed and released
pub struct JailedStream {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
    emission_mode: EmissionMode,
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
            // State variables - using new deterministic choice state management
            let mut choice_states = ChoiceJailStateCollection::new();
            let mut last_response_metadata: Option<NvCreateChatCompletionStreamResponse> = None;
            let mut buffered_content = String::new();
            // Track Annotated metadata for preservation
            let mut last_annotated_id: Option<String> = None;
            let mut last_annotated_event: Option<String> = None;
            let mut last_annotated_comment: Option<Vec<String>> = None;

            // Pin the stream for iteration (stack pinning is more efficient)
            tokio::pin!(stream);

            // Process each item in the stream
            while let Some(response) = stream.next().await {
                if let Some(chat_response) = response.data.as_ref() {
                    let mut any_choices_jailed = false;
                    let mut any_choices_unjailed = false;
                    let mut unjailed_choice_indices = HashSet::new();

                    // Process each choice independently
                    for choice in &chat_response.choices {
                        if let Some(ref content) = choice.delta.content {
                            let choice_state = choice_states.get_or_create_state(choice.index);

                            // Check if this choice should start jailing
                            if !choice_state.is_jailed {
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
                                        "Choice {} jail triggered (sequence: {}, tool_call: {}), starting accumulation",
                                        choice.index, sequence_match, tool_call_match
                                    );

                                    // Store metadata only when we actually jail (first time)
                                    if last_response_metadata.is_none() {
                                        last_response_metadata = response.data.clone();
                                        // Preserve Annotated metadata for correlation
                                        last_annotated_id = response.id.clone();
                                        last_annotated_event = response.event.clone();
                                        last_annotated_comment = response.comment.clone();
                                    }

                                    // Start accumulating for this choice
                                    choice_state.start_jail(content);
                                    if choice.index == 0 {
                                        buffered_content = content.clone();
                                    }
                                    any_choices_jailed = true;
                                }
                            } else {
                                // Choice is already jailed, accumulate content
                                choice_state.accumulate(content);
                                if choice.index == 0 {
                                    buffered_content.push_str(content);
                                }
                                any_choices_jailed = true;

                                // Check for jail end - two paths
                                // Path 1: End sequence detected
                                let end_marker_info = if !self.jail_end_sequences.is_empty() {
                                    self.jail_end_sequences.iter()
                                        .find_map(|seq| {
                                            choice_state.accumulated_content.find(seq).map(|pos| (pos + seq.len(), seq.clone()))
                                        })
                                } else { None };

                                // Path 2: Complete tool call(s) can be parsed (early exit)
                                let early_exit = self.should_exit_jail_early(&choice_state.accumulated_content);

                                // Determine if this choice should unjail
                                if end_marker_info.is_some() || early_exit {
                                    tracing::debug!(
                                        "Choice {} jail exit detected (end_marker: {}, early: {}), releasing accumulated content",
                                        choice.index, end_marker_info.is_some(), early_exit
                                    );

                                    // Determine split position for content
                                    let split_pos = if let Some((end_pos, _)) = end_marker_info {
                                        end_pos
                                    } else if early_exit {
                                        // For early exit, find where the complete tool call ends
                                        if let Some(parser) = &self.tool_call_parser {
                                            if let Ok((_, _)) = try_tool_call_parse_aggregate(&choice_state.accumulated_content, Some(parser)) {
                                                self.find_tool_call_end_position(&choice_state.accumulated_content, parser)
                                            } else {
                                                choice_state.accumulated_content.len()
                                            }
                                        } else {
                                            choice_state.accumulated_content.len()
                                        }
                                    } else {
                                        choice_state.accumulated_content.len()
                                    };

                                    // Split the content for this choice
                                    let (jailed_part, trailing_part) = choice_state.accumulated_content.split_at(split_pos);

                                    // Store the content to be emitted
                                    let jailed_content = jailed_part.to_string();
                                    let trailing_content = if !trailing_part.is_empty() {
                                        Some(trailing_part.to_string())
                                    } else {
                                        None
                                    };

                                    // End jailing for this choice
                                    choice_state.end_jail();

                                    // Emit the unjailed content for this choice
                                    if let Some(base_response) = last_response_metadata.as_ref() {
                                        // Create a HashMap with just this choice for emission
                                        let mut single_choice_content = HashMap::new();
                                        single_choice_content.insert(choice.index, jailed_content);

                                        let unjailed_response = self.create_unjailed_response(
                                            base_response.clone(),
                                            &single_choice_content,
                                            last_annotated_id.clone(),
                                            last_annotated_event.clone(),
                                            last_annotated_comment.clone(),
                                        );
                                        yield unjailed_response;

                                        // Emit trailing content if any exists
                                        if let Some(trailing) = trailing_content {
                                            let mut trailing_response = base_response.clone();
                                            // Find the choice in the response and update its content
                                            for response_choice in &mut trailing_response.choices {
                                                if response_choice.index == choice.index {
                                                    response_choice.delta.content = Some(trailing);
                                                    response_choice.delta.tool_calls = None;
                                                    response_choice.finish_reason = None;
                                                    break;
                                                }
                                            }

                                            let trailing_annotated = Annotated {
                                                data: Some(trailing_response),
                                                id: last_annotated_id.clone(),
                                                event: last_annotated_event.clone(),
                                                comment: last_annotated_comment.clone(),
                                            };
                                            yield trailing_annotated;
                                        }
                                    }

                                    any_choices_unjailed = true;
                                    unjailed_choice_indices.insert(choice.index);
                                }
                            }
                        }
                    }

                    // Determine what to emit based on jail states
                    if !any_choices_jailed {
                        // No choices are jailed, emit according to emission mode
                        let metadata = (response.id.clone(), response.event.clone(), response.comment.clone());
                        let responses = self.emit_response(chat_response.choices.clone(), chat_response, metadata);
                        for emitted_response in responses {
                            yield emitted_response;
                        }
                    } else if any_choices_unjailed {
                        // Some choices have finished jailing and been emitted above
                        // Now handle any remaining non-jailed choices in this chunk

                        // Create a response with only the non-jailed choices from this chunk
                        // Exclude choices that unjailed in this chunk to avoid double emission
                        let mut pass_through_choices = Vec::new();
                        for choice in &chat_response.choices {
                            // Skip choices that just unjailed in this chunk
                            if unjailed_choice_indices.contains(&choice.index) {
                                continue;
                            }

                            if let Some(choice_state) = choice_states.get_state(choice.index) {
                                if !choice_state.is_jailed {
                                    // This choice is not jailed, include it in pass-through
                                    pass_through_choices.push(choice.clone());
                                }
                            } else {
                                // No state means this choice was never jailed, include it
                                pass_through_choices.push(choice.clone());
                            }
                        }

                        // Emit non-jailed choices if any
                        if !pass_through_choices.is_empty() {
                            let metadata = (response.id.clone(), response.event.clone(), response.comment.clone());
                            let responses = self.emit_response(pass_through_choices, chat_response, metadata);
                            for emitted_response in responses {
                                yield emitted_response;
                            }
                        }
                    } else {
                        // All jailed choices are still accumulating, don't yield anything
                        continue;
                    }
                } else {
                    // No response data, pass through as-is
                    yield response;
                }
            }

            // Stream ended - if any choices are still jailed, release accumulated content
            if choice_states.has_jailed_choices() {
                tracing::debug!("Stream ended while jailed, releasing accumulated content");
                if let Some(base_response) = last_response_metadata.take() {
                    // Convert to HashMap for compatibility with existing create_unjailed_response method
                    let accumulated_content = choice_states.to_hashmap();
                    let final_response = self.create_unjailed_response(
                        base_response,
                        &accumulated_content,
                        last_annotated_id.clone(),
                        last_annotated_event.clone(),
                        last_annotated_comment.clone(),
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

    /// Find the exact position where the tool call ends for splitting content
    /// This handles the early exit case where we have trailing content after the tool call
    fn find_tool_call_end_position(&self, content: &str, parser: &str) -> usize {
        match parser {
            "hermes" => {
                // For Hermes, look for </tool_call> marker
                if let Some(pos) = content.find("</tool_call>") {
                    pos + "</tool_call>".len()
                } else {
                    content.len()
                }
            }
            "nemotron_deci" => {
                // For Nemotron, look for </TOOLCALL> marker
                if let Some(pos) = content.find("</TOOLCALL>") {
                    pos + "</TOOLCALL>".len()
                } else {
                    content.len()
                }
            }
            "mistral" => {
                // For Mistral, look for [/TOOL_CALLS] marker or end of JSON array
                if let Some(pos) = content.find("[/TOOL_CALLS]") {
                    pos + "[/TOOL_CALLS]".len()
                } else if let Some(pos) = content.rfind(']') {
                    // Find the last ] which should be the end of the tool calls array
                    pos + 1
                } else {
                    content.len()
                }
            }
            "phi4" => {
                // For Phi4, look for <|tool_call|> end marker
                if let Some(pos) = content.rfind("<|tool_call|>") {
                    // Look for the next occurrence after this position
                    if let Some(end_pos) = content[pos..].find(">") {
                        pos + end_pos + 1
                    } else {
                        content.len()
                    }
                } else {
                    content.len()
                }
            }
            "llama3_json" => {
                // For Llama3 JSON, there's no explicit end marker
                // The end is determined by complete JSON parsing
                // Return full content length to avoid early splitting
                content.len()
            }
            _ => {
                // Unknown parser, default to full content
                content.len()
            }
        }
    }

    /// Emit a response based on the configured emission mode
    fn emit_response(
        &self,
        choices: Vec<ChatChoiceStream>,
        base_response: &NvCreateChatCompletionStreamResponse,
        annotated_metadata: (Option<String>, Option<String>, Option<Vec<String>>),
    ) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
        let (id, event, comment) = annotated_metadata;

        match self.emission_mode {
            EmissionMode::Packed => {
                // Pack all choices into a single response
                let mut response = base_response.clone();
                response.choices = choices;

                vec![Annotated {
                    data: Some(response),
                    id,
                    event,
                    comment,
                }]
            }
            EmissionMode::SingleChoicePerChunk => {
                // Emit each choice in a separate response
                choices
                    .into_iter()
                    .map(|choice| {
                        let mut response = base_response.clone();
                        response.choices = vec![choice];

                        Annotated {
                            data: Some(response),
                            id: id.clone(),
                            event: event.clone(),
                            comment: comment.clone(),
                        }
                    })
                    .collect()
            }
        }
    }

    /// Create a response with accumulated content, potentially parsing tool calls
    fn create_unjailed_response(
        &self,
        mut base_response: NvCreateChatCompletionStreamResponse,
        accumulated_content: &HashMap<u32, String>,
        id: Option<String>,
        event: Option<String>,
        comment: Option<Vec<String>>,
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
            id,
            event,
            comment,
        }
    }
}

/// Builder for configuring a JailedStream
pub struct JailedStreamBuilder {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
    emission_mode: EmissionMode,
}

impl JailedStreamBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            jail_start_sequences: Vec::new(),
            jail_end_sequences: Vec::new(),
            tool_call_parser: None,
            emission_mode: EmissionMode::default(),
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

    /// Set the emission mode for handling multiple choices
    pub fn emission_mode(mut self, mode: EmissionMode) -> Self {
        self.emission_mode = mode;
        self
    }

    /// Enable single choice per chunk emission for OpenAI compatibility
    pub fn single_choice_per_chunk(mut self) -> Self {
        self.emission_mode = EmissionMode::SingleChoicePerChunk;
        self
    }

    /// Enable packed emission mode (multiple choices per chunk)
    pub fn packed_emission(mut self) -> Self {
        self.emission_mode = EmissionMode::Packed;
        self
    }

    /// Build the configured JailedStream
    pub fn build(self) -> JailedStream {
        JailedStream {
            jail_start_sequences: self.jail_start_sequences,
            jail_end_sequences: self.jail_end_sequences,
            tool_call_parser: self.tool_call_parser,
            emission_mode: self.emission_mode,
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

        /// Helper function to create a mock chat response chunk with metadata
        pub fn create_annotated_chunk(
            content: String,
            index: u32,
            id: Option<String>,
            event: Option<String>,
            comment: Option<Vec<String>>,
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
                id,
                event,
                comment,
            }
        }

        /// Helper function to create a multi-choice chunk
        pub fn create_multi_choice_chunk(
            choices_content: Vec<(String, u32)>, // (content, index)
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            let choices: Vec<ChatChoiceStream> = choices_content
                .into_iter()
                .map(|(content, index)| {
                    #[allow(deprecated)]
                    ChatChoiceStream {
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
                    }
                })
                .collect();

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices,
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

    #[tokio::test]
    async fn test_jailed_stream_preserves_metadata() {
        // Test metadata preservation through jail processing
        let test_id = Some("correlation-id-123".to_string());
        let test_event = Some("request-processing".to_string());
        let test_comment = Some(vec![
            "upstream-correlation".to_string(),
            "debug-info".to_string(),
        ]);

        // Create chunks with specific metadata for the jail trigger
        let chunks = vec![
            create_annotated_chunk(
                "I'll help you with that. ".to_string(),
                0,
                None, // No metadata on first chunk
                None,
                None,
            ),
            create_annotated_chunk(
                "<tool_call>".to_string(),
                0,
                test_id.clone(), // Metadata on jail trigger chunk
                test_event.clone(),
                test_comment.clone(),
            ),
            create_mock_response_chunk("{\"name\": \"search_web\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"query\": \"test\"}}".to_string(), 0),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
            create_mock_response_chunk(" Processing complete.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Hermes parser
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should get 3 chunks: before jail, tool call response, after jail
        assert!(
            results.len() >= 3,
            "Should have at least 3 chunks, got {}",
            results.len()
        );

        // Find the synthesized tool call response chunk
        let tool_call_chunk = results
            .iter()
            .find(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .map(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                    .unwrap_or(false)
            })
            .expect("Should have a tool call response chunk");

        // Verify metadata is preserved
        assert_eq!(
            tool_call_chunk.id, test_id,
            "ID should be preserved from jail trigger chunk"
        );
        assert_eq!(
            tool_call_chunk.event, test_event,
            "Event should be preserved from jail trigger chunk"
        );
        assert_eq!(
            tool_call_chunk.comment, test_comment,
            "Comment should be preserved from jail trigger chunk"
        );

        // Verify tool call was parsed correctly
        let tool_calls = &tool_call_chunk.data.as_ref().unwrap().choices[0]
            .delta
            .tool_calls;
        assert!(tool_calls.is_some(), "Should have tool calls");
        let tool_calls = tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1, "Should have exactly one tool call");
        assert_eq!(
            tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .name
                .as_ref()
                .unwrap(),
            "search_web"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_preserves_metadata_on_stream_end() {
        // Test metadata preservation when stream ends while jailed
        let test_id = Some("end-correlation-456".to_string());
        let test_event = Some("stream-termination".to_string());
        let test_comment = Some(vec!["incomplete-processing".to_string()]);

        // Create chunks that end while jailed (no explicit end marker)
        let chunks = vec![
            create_mock_response_chunk("Starting function call: ".to_string(), 0),
            create_annotated_chunk(
                "<tool_call>".to_string(), // This chunk triggers jail and has metadata
                0,
                test_id.clone(),
                test_event.clone(),
                test_comment.clone(),
            ),
            create_mock_response_chunk(
                "{\"name\": \"incomplete_call\"".to_string(), // No closing brace
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Hermes parser
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should get 2 chunks: first chunk passes through, stream end releases accumulated
        assert_eq!(results.len(), 2, "Should have exactly 2 chunks");

        // The second chunk is the accumulated content released when stream ended
        let accumulated_chunk = &results[1];

        // Verify metadata is preserved from the jail trigger
        assert_eq!(
            accumulated_chunk.id, test_id,
            "ID should be preserved when stream ends while jailed"
        );
        assert_eq!(
            accumulated_chunk.event, test_event,
            "Event should be preserved when stream ends while jailed"
        );
        assert_eq!(
            accumulated_chunk.comment, test_comment,
            "Comment should be preserved when stream ends while jailed"
        );

        // Verify accumulated content is returned
        let content = &accumulated_chunk.data.as_ref().unwrap().choices[0]
            .delta
            .content;
        assert!(content.is_some(), "Should have accumulated content");
        let content = content.as_ref().unwrap();
        assert!(
            content.contains("<tool_call>"),
            "Should contain jail start marker in accumulated content"
        );
        assert!(
            content.contains("incomplete_call"),
            "Should contain accumulated incomplete content"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_metadata_edge_cases() {
        // Test edge cases: empty metadata, partial metadata, etc.
        let chunks = vec![
            create_annotated_chunk(
                "Text with ".to_string(),
                0,
                Some("".to_string()), // Empty string ID
                None,                 // No event
                Some(vec![]),         // Empty comment vector
            ),
            create_annotated_chunk(
                "<tool_call>".to_string(),
                0,
                None,                                 // No ID
                Some("partial-metadata".to_string()), // Only event
                None,                                 // No comment
            ),
            create_mock_response_chunk("{\"name\": \"test\", \"arguments\": {}}".to_string(), 0),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Find the tool call response
        let tool_call_chunk = results
            .iter()
            .find(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .map(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                    .unwrap_or(false)
            })
            .expect("Should have a tool call response chunk");

        // Verify partial metadata is preserved correctly
        assert_eq!(tool_call_chunk.id, None, "Should preserve None ID");
        assert_eq!(
            tool_call_chunk.event,
            Some("partial-metadata".to_string()),
            "Should preserve event"
        );
        assert_eq!(
            tool_call_chunk.comment, None,
            "Should preserve None comment"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_trailing_content_same_chunk() {
        // Regression test for GitHub issue: trailing content after end marker in same chunk
        let chunks = vec![
            create_mock_response_chunk("I'll help you. ".to_string(), 0),
            create_mock_response_chunk("<tool_call>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"search\", \"arguments\": {}}".to_string(), 0),
            // This chunk contains both the end marker AND trailing content
            create_mock_response_chunk(
                "</tool_call>trailing text that should not be lost".to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should get: initial text, tool call response, trailing text
        assert!(
            results.len() >= 3,
            "Should have at least 3 chunks, got {}",
            results.len()
        );

        // Find the tool call response
        let tool_call_chunk = results
            .iter()
            .find(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .map(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                    .unwrap_or(false)
            })
            .expect("Should have a tool call response chunk");

        // Verify tool call was parsed correctly
        let tool_calls = &tool_call_chunk.data.as_ref().unwrap().choices[0]
            .delta
            .tool_calls;
        assert!(tool_calls.is_some(), "Should have tool calls");
        assert_eq!(
            tool_calls.as_ref().unwrap().len(),
            1,
            "Should have exactly one tool call"
        );

        // CRITICAL: Verify trailing content is preserved in a separate chunk
        let trailing_chunk = results
            .iter()
            .find(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .and_then(|c| c.delta.content.as_ref())
                    .map(|content| content.contains("trailing text that should not be lost"))
                    .unwrap_or(false)
            })
            .expect("Should have a chunk with trailing content");

        // Verify the trailing content is exactly what we expect
        let trailing_content = &trailing_chunk.data.as_ref().unwrap().choices[0]
            .delta
            .content;
        assert_eq!(
            trailing_content.as_deref(),
            Some("trailing text that should not be lost"),
            "Trailing content should be preserved exactly"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_early_exit_with_trailing() {
        // Test early exit (complete tool call detected) with trailing content
        let chunks = vec![
            create_mock_response_chunk("Starting task: ".to_string(), 0),
            create_mock_response_chunk(
                "<tool_call>{\"name\": \"complete_task\", \"arguments\": {}}".to_string(),
                0,
            ),
            // Early exit should happen here, but we also have trailing content
            create_mock_response_chunk("</tool_call> Task completed successfully.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should get: initial text, tool call response, trailing text
        assert!(
            results.len() >= 3,
            "Should have at least 3 chunks, got {}",
            results.len()
        );

        // Verify we have a tool call response
        let has_tool_call = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .map(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                .unwrap_or(false)
        });
        assert!(has_tool_call, "Should have a tool call response");

        // CRITICAL: Verify trailing content after early exit is preserved
        let trailing_chunk = results
            .iter()
            .find(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .and_then(|c| c.delta.content.as_ref())
                    .map(|content| content.contains("Task completed successfully"))
                    .unwrap_or(false)
            })
            .expect("Should have a chunk with trailing content after early exit");

        let trailing_content = &trailing_chunk.data.as_ref().unwrap().choices[0]
            .delta
            .content;
        assert_eq!(
            trailing_content.as_deref(),
            Some(" Task completed successfully."),
            "Trailing content after early exit should be preserved"
        );
    }

    #[tokio::test]
    async fn test_multiple_choices_independent_jailing() {
        // Test that different choices can jail and unjail independently
        // This test will FAIL with the current HashMap-based implementation
        let chunks = vec![
            // Chunk 1: All choices start normally
            create_multi_choice_chunk(vec![
                ("Starting task A. ".to_string(), 0),
                ("Starting task B. ".to_string(), 1),
                ("Starting task C. ".to_string(), 2),
            ]),
            // Chunk 2: Choice 0 starts tool call (gets jailed), others continue
            create_multi_choice_chunk(vec![
                ("<tool_call>".to_string(), 0),    // Choice 0 jailed
                ("Continuing B. ".to_string(), 1), // Choice 1 continues
                ("Continuing C. ".to_string(), 2), // Choice 2 continues
            ]),
            // Chunk 3: Choice 0 still jailed, Choice 2 starts tool call
            create_multi_choice_chunk(vec![
                ("{\"name\": \"tool_a\"".to_string(), 0), // Choice 0 still jailed
                ("More B content. ".to_string(), 1),      // Choice 1 continues
                ("<tool_call>".to_string(), 2),           // Choice 2 now jailed
            ]),
            // Chunk 4: Choice 0 finishes tool call, Choice 2 continues tool call
            create_multi_choice_chunk(vec![
                (", \"arguments\": {}}</tool_call>".to_string(), 0), // Choice 0 unjails
                ("Final B. ".to_string(), 1),                        // Choice 1 continues
                ("{\"name\": \"tool_c\"}".to_string(), 2),           // Choice 2 still jailed
            ]),
            // Chunk 5: Choice 2 finishes tool call
            create_multi_choice_chunk(vec![
                ("After tool A. ".to_string(), 0), // Choice 0 continues after unjail
                ("Done with B. ".to_string(), 1),  // Choice 1 continues
                ("</tool_call>".to_string(), 2),   // Choice 2 unjails
            ]),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // EXPECTED BEHAVIOR (will fail with current implementation):
        // - Choice 1 should stream continuously (never jailed)
        // - Choice 0 should jail from chunk 2 until chunk 4
        // - Choice 2 should jail from chunk 3 until chunk 5
        // - Each choice should emit independently

        // Verify choice 1 was never interrupted (should have ~5 chunks of content)
        let choice_1_chunks: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.index == 1 && c.delta.content.is_some())
            .collect();

        assert!(
            choice_1_chunks.len() >= 4,
            "Choice 1 should have multiple continuous chunks, got {}",
            choice_1_chunks.len()
        );

        // Verify choice 0 has a tool call response
        let choice_0_tool_calls: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.index == 0 && c.finish_reason == Some(FinishReason::ToolCalls))
            .collect();

        assert!(
            !choice_0_tool_calls.is_empty(),
            "Choice 0 should have tool call response"
        );

        // Verify choice 2 has a tool call response
        let choice_2_tool_calls: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.index == 2 && c.finish_reason == Some(FinishReason::ToolCalls))
            .collect();

        assert!(
            !choice_2_tool_calls.is_empty(),
            "Choice 2 should have tool call response"
        );
    }

    #[tokio::test]
    async fn test_deterministic_choice_ordering() {
        // Test that choices are processed in deterministic order (0, 1, 2...)
        // This test will FAIL with the current HashMap implementation
        let chunks = vec![
            // All choices have tool calls that complete at the same time
            create_multi_choice_chunk(vec![
                (
                    "<tool_call>{\"name\": \"tool_0\", \"arguments\": {}}</tool_call>".to_string(),
                    0,
                ),
                (
                    "<tool_call>{\"name\": \"tool_1\", \"arguments\": {}}</tool_call>".to_string(),
                    1,
                ),
                (
                    "<tool_call>{\"name\": \"tool_2\", \"arguments\": {}}</tool_call>".to_string(),
                    2,
                ),
            ]),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Find all tool call responses
        let mut tool_call_responses: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.finish_reason == Some(FinishReason::ToolCalls))
            .collect();

        // Sort by the order they appear in the results
        // With HashMap, this order will be non-deterministic
        // With Vec, this should always be [0, 1, 2]
        tool_call_responses.sort_by_key(|c| c.index);

        assert_eq!(
            tool_call_responses.len(),
            3,
            "Should have 3 tool call responses"
        );

        // Run this test multiple times to verify determinism
        for run in 0..5 {
            let chunks = vec![create_multi_choice_chunk(vec![
                (
                    "<tool_call>{\"name\": \"tool_0\", \"arguments\": {}}</tool_call>".to_string(),
                    0,
                ),
                (
                    "<tool_call>{\"name\": \"tool_1\", \"arguments\": {}}</tool_call>".to_string(),
                    1,
                ),
                (
                    "<tool_call>{\"name\": \"tool_2\", \"arguments\": {}}</tool_call>".to_string(),
                    2,
                ),
            ])];

            let input_stream = stream::iter(chunks);
            let jail = JailedStream::builder().tool_call_parser("hermes").build();
            let jailed_stream = jail.apply(input_stream);
            let run_results: Vec<_> = jailed_stream.collect().await;

            let run_responses: Vec<_> = run_results
                .iter()
                .filter_map(|r| r.data.as_ref())
                .flat_map(|d| &d.choices)
                .filter(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                .collect();

            // The order should be consistent across runs
            // This will fail with HashMap due to non-deterministic iteration
            let indices: Vec<u32> = run_responses.iter().map(|c| c.index).collect();
            assert_eq!(
                indices,
                vec![0, 1, 2],
                "Choice processing order should be deterministic on run {}",
                run
            );
        }
    }

    #[tokio::test]
    async fn test_multiple_choices_usage_aggregation() {
        // Test that usage is correctly aggregated across multiple choices
        // This test demonstrates how usage should work with n>1

        // For now, this test just documents expected behavior
        // It will need to be expanded once usage aggregation is implemented

        let chunks = vec![create_multi_choice_chunk(vec![
            ("Response A with many tokens".to_string(), 0), // ~5 tokens
            ("Response B".to_string(), 1),                  // ~2 tokens
            ("Response C has even more tokens than A".to_string(), 2), // ~8 tokens
        ])];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // TODO: Once usage aggregation is implemented, verify:
        // - Usage chunk has choices: [] (empty array)
        // - completion_tokens = sum of all choices (~15 total)
        // - prompt_tokens counted once
        // - total_tokens = prompt_tokens + completion_tokens

        // For now, just verify we got some results
        assert!(!results.is_empty(), "Should have some results");
    }
}
