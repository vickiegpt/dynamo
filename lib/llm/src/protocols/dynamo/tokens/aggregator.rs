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

use std::{collections::HashMap, str::FromStr};

use anyhow::Result;
use futures::StreamExt;

use super::{CompletionChoice, CompletionResponse};
use crate::protocols::{common::FinishReason, openai::CompletionUsage, Annotated, DataStream};

/// Aggregates a stream of [`CompletionResponse`]s into a single [`CompletionResponse`].
/// This is identical to the OpenAI aggregator but for token completions.
pub struct DeltaAggregator {
    id: String,
    model: String,
    created: u64,
    usage: Option<CompletionUsage>,
    system_fingerprint: Option<String>,
    choices: HashMap<u64, DeltaChoice>,
    error: Option<String>,
}

struct DeltaChoice {
    index: u64,
    text: String,
    finish_reason: Option<FinishReason>,
    logprobs: Option<super::LogprobResult>,
}

impl Default for DeltaAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAggregator {
    pub fn new() -> Self {
        Self {
            id: "".to_string(),
            model: "".to_string(),
            created: 0,
            usage: None,
            system_fingerprint: None,
            choices: HashMap::new(),
            error: None,
        }
    }

    /// Aggregates a stream of [`Annotated<CompletionResponse>`]s into a single [`CompletionResponse`].
    pub async fn apply(
        stream: DataStream<Annotated<CompletionResponse>>,
    ) -> Result<CompletionResponse> {
        let aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                let delta = match delta.ok() {
                    Ok(delta) => delta,
                    Err(error) => {
                        aggregator.error = Some(error);
                        return aggregator;
                    }
                };

                if aggregator.error.is_none() && delta.data.is_some() {
                    // note: we could extract annotations here and add them to the aggregator
                    // to be return as part of the NIM Response Extension
                    // TODO(#14) - Aggregate Annotation

                    // these are cheap to move so we do it every time since we are consuming the delta
                    let delta = delta.data.unwrap();
                    aggregator.id = delta.id;
                    aggregator.model = delta.model;
                    aggregator.created = delta.created;
                    if let Some(usage) = delta.usage {
                        aggregator.usage = Some(usage);
                    }
                    if let Some(system_fingerprint) = delta.system_fingerprint {
                        aggregator.system_fingerprint = Some(system_fingerprint);
                    }

                    // handle the choices
                    for choice in delta.choices {
                        let state_choice =
                            aggregator
                                .choices
                                .entry(choice.index)
                                .or_insert(DeltaChoice {
                                    index: choice.index,
                                    text: "".to_string(),
                                    finish_reason: None,
                                    logprobs: choice.logprobs,
                                });

                        state_choice.text.push_str(&choice.text);

                        // todo - handle logprobs

                        if let Some(finish_reason) = choice.finish_reason {
                            let reason = FinishReason::from_str(&finish_reason).ok();
                            state_choice.finish_reason = reason;
                        }
                    }
                }
                aggregator
            })
            .await;

        // If we have an error, return it
        let aggregator = if let Some(error) = aggregator.error {
            return Err(anyhow::anyhow!(error));
        } else {
            aggregator
        };

        // extra the aggregated deltas and sort by index
        let mut choices: Vec<_> = aggregator
            .choices
            .into_values()
            .map(CompletionChoice::from)
            .collect();

        choices.sort_by(|a, b| a.index.cmp(&b.index));

        Ok(CompletionResponse {
            id: aggregator.id,
            created: aggregator.created,
            usage: aggregator.usage,
            model: aggregator.model,
            object: "text_completion".to_string(),
            system_fingerprint: aggregator.system_fingerprint,
            choices,
        })
    }
}

impl From<DeltaChoice> for CompletionChoice {
    fn from(delta: DeltaChoice) -> Self {
        let finish_reason = delta.finish_reason.map(|reason| reason.to_string());

        CompletionChoice {
            index: delta.index,
            text: delta.text,
            finish_reason,
            logprobs: delta.logprobs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::dynamo::tokens::{CompletionChoice, CompletionResponse};
    use futures::stream;

    fn create_test_delta(
        index: u64,
        text: &str,
        finish_reason: Option<String>,
    ) -> Annotated<CompletionResponse> {
        Annotated {
            data: Some(CompletionResponse {
                id: "dynamo-token-test".to_string(),
                model: "test-model".to_string(),
                created: 1234567890,
                usage: None,
                system_fingerprint: None,
                choices: vec![CompletionChoice {
                    index,
                    text: text.to_string(),
                    finish_reason,
                    logprobs: None,
                }],
                object: "text_completion".to_string(),
            }),
            id: Some("dynamo-token-test".to_string()),
            event: None,
            comment: None,
        }
    }

    #[tokio::test]
    async fn test_empty_stream() {
        let stream: DataStream<Annotated<CompletionResponse>> = Box::pin(stream::empty());
        let result = DeltaAggregator::apply(stream).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.id, "");
        assert_eq!(response.model, "");
        assert_eq!(response.choices.len(), 0);
    }

    #[tokio::test]
    async fn test_single_delta() {
        let annotated_delta = create_test_delta(0, "Hello, world!", Some("stop".to_string()));
        let stream = Box::pin(stream::iter(vec![annotated_delta]));
        let result = DeltaAggregator::apply(stream).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.id, "dynamo-token-test");
        assert_eq!(response.model, "test-model");
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.text, "Hello, world!");
        assert_eq!(choice.finish_reason, Some("stop".to_string()));
    }

    #[tokio::test]
    async fn test_multiple_deltas_same_choice() {
        let annotated_delta1 = create_test_delta(0, "Hello,", None);
        let annotated_delta2 = create_test_delta(0, " world!", Some("stop".to_string()));
        let annotated_deltas = vec![annotated_delta1, annotated_delta2];
        let stream = Box::pin(stream::iter(annotated_deltas));
        let result = DeltaAggregator::apply(stream).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.text, "Hello, world!");
        assert_eq!(choice.finish_reason, Some("stop".to_string()));
    }
}
