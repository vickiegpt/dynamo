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

use anyhow::Result;
use futures::{Stream, StreamExt};

use super::{CompletionChoice, CompletionResponse, TokenCompletionResponseFactory};
use crate::protocols::{
    common::{FinishReason, StreamingCompletionResponse},
    openai::CompletionUsage,
    Annotated, DataStream,
};

/// Generates streaming completion deltas from internal StreamingCompletionResponse
/// This converts internal protocol to OpenAI-compatible format for token completions
pub struct DeltaGenerator {
    factory: TokenCompletionResponseFactory,
}

impl DeltaGenerator {
    pub fn new(model: String) -> Result<Self> {
        let factory = TokenCompletionResponseFactory::builder()
            .model(model)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build response factory: {}", e))?;

        Ok(Self { factory })
    }

    /// Convert a stream of internal StreamingCompletionResponse to CompletionResponse deltas
    pub fn generate_deltas(
        &self,
        stream: DataStream<Annotated<StreamingCompletionResponse>>,
    ) -> impl Stream<Item = Annotated<CompletionResponse>> + '_ {
        stream.map(move |annotated_response| {
            let response = match annotated_response.data {
                Some(response) => response,
                None => {
                    // Pass through empty responses
                    return Annotated {
                        data: None,
                        id: annotated_response.id,
                        event: annotated_response.event,
                        comment: annotated_response.comment,
                    };
                }
            };

            // Convert internal response to OpenAI format
            let choice = match self.convert_delta_to_choice(&response) {
                Ok(choice) => choice,
                Err(_) => {
                    // Return error response
                    return Annotated {
                        data: None,
                        id: annotated_response.id,
                        event: Some("error".to_string()),
                        comment: Some(vec!["Failed to convert response delta".to_string()]),
                    };
                }
            };

            let usage = response.delta.usage.map(|u| CompletionUsage {
                prompt_tokens: u.input_tokens_count as i32,
                completion_tokens: u.output_tokens_count as i32,
                total_tokens: (u.input_tokens_count + u.output_tokens_count) as i32,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            });

            let completion_response = self.factory.make_response(choice, usage);

            Annotated {
                data: Some(completion_response),
                id: annotated_response.id,
                event: annotated_response.event,
                comment: annotated_response.comment,
            }
        })
    }

    fn convert_delta_to_choice(
        &self,
        response: &StreamingCompletionResponse,
    ) -> Result<CompletionChoice> {
        let text = response.delta.text.clone().unwrap_or_default();
        let index = response.delta.index.unwrap_or(0) as u64;

        let finish_reason = match &response.delta.finish_reason {
            Some(FinishReason::EoS) => Some("stop".to_string()),
            Some(FinishReason::Stop) => Some("stop".to_string()),
            Some(FinishReason::Length) => Some("length".to_string()),
            Some(FinishReason::Error(err_msg)) => {
                return Err(anyhow::anyhow!("finish_reason::error = {}", err_msg));
            }
            Some(FinishReason::Cancelled) => Some("cancelled".to_string()),
            None => None,
        };

        Ok(CompletionChoice {
            text,
            index,
            finish_reason,
            logprobs: None, // TODO: implement logprobs support
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{Delta, Usage};
    use futures::stream;

    fn create_test_streaming_response(
        text: Option<String>,
        finish_reason: Option<FinishReason>,
        index: Option<usize>,
    ) -> Annotated<StreamingCompletionResponse> {
        Annotated {
            data: Some(StreamingCompletionResponse {
                delta: Delta {
                    is_complete: finish_reason.is_some(),
                    finish_reason,
                    token_ids: None,
                    tokens: None,
                    text,
                    sequence_length: None,
                    index,
                    cum_log_probs: None,
                    err_msg: None,
                    usage: Some(Usage {
                        input_tokens_count: 10,
                        output_tokens_count: 5,
                    }),
                },
                logprobs: None,
            }),
            id: Some("test-id".to_string()),
            event: None,
            comment: None,
        }
    }

    #[tokio::test]
    async fn test_delta_generation() {
        let generator = DeltaGenerator::new("test-model".to_string()).unwrap();

        let internal_responses = vec![
            create_test_streaming_response(Some("Hello".to_string()), None, Some(0)),
            create_test_streaming_response(Some(" world".to_string()), None, Some(0)),
            create_test_streaming_response(
                Some("!".to_string()),
                Some(FinishReason::Stop),
                Some(0),
            ),
        ];

        let stream = Box::pin(stream::iter(internal_responses));
        let mut delta_stream = generator.generate_deltas(stream);

        // Test first delta
        let delta1 = delta_stream.next().await.unwrap();
        assert!(delta1.data.is_some());
        let response1 = delta1.data.unwrap();
        assert_eq!(response1.choices[0].text, "Hello");
        assert_eq!(response1.choices[0].index, 0);
        assert!(response1.choices[0].finish_reason.is_none());

        // Test second delta
        let delta2 = delta_stream.next().await.unwrap();
        assert!(delta2.data.is_some());
        let response2 = delta2.data.unwrap();
        assert_eq!(response2.choices[0].text, " world");

        // Test final delta
        let delta3 = delta_stream.next().await.unwrap();
        assert!(delta3.data.is_some());
        let response3 = delta3.data.unwrap();
        assert_eq!(response3.choices[0].text, "!");
        assert_eq!(response3.choices[0].finish_reason, Some("stop".to_string()));
    }

    #[tokio::test]
    async fn test_empty_delta() {
        let generator = DeltaGenerator::new("test-model".to_string()).unwrap();

        let empty_response = Annotated {
            data: None,
            id: Some("test-id".to_string()),
            event: Some("keep-alive".to_string()),
            comment: None,
        };

        let stream = Box::pin(stream::iter(vec![empty_response]));
        let mut delta_stream = generator.generate_deltas(stream);

        let delta = delta_stream.next().await.unwrap();
        assert!(delta.data.is_none());
        assert_eq!(delta.event, Some("keep-alive".to_string()));
    }
}
