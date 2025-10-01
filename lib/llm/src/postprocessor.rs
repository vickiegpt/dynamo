// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Postprocessor consists of the following modules
//!
//! - `translation`: This module converts the BackendOutput to PostprocessedResponse
//! - `apply`: This module applies any post-processing logic like reasoning parsing
//! - `format`: This module formats the response according to the target protocol
//!
//! The Postprocessor will accept BackendOutput and transform it to the final response format.

use anyhow::Result;
use futures::Stream;
use futures::stream::StreamExt;
use std::sync::Arc;

use crate::model_card::{ModelDeploymentCard, ModelInfo};
use crate::protocols::common::postprocessor::PostprocessedResponse;

use dynamo_runtime::engine::{AsyncEngine, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyOut, Operator, async_trait};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::protocols::{
    common::llm_backend::{BackendOutput, EmbeddingsEngineOutput},
    openai::{
        chat_completions::NvCreateChatCompletionStreamResponse,
        completions::NvCreateCompletionResponse, embeddings::NvCreateEmbeddingResponse,
    },
};

pub struct OpenAIPostprocessor {
    mdcsum: String,
    _model_info: Arc<dyn ModelInfo>,
    /// Per-model runtime configuration for post-processing
    _runtime_config: crate::local_model::runtime_config::ModelRuntimeConfig,
}

impl OpenAIPostprocessor {
    pub fn new(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        let mdcsum = mdc.mdcsum();
        let Some(model_info) = mdc.model_info else {
            anyhow::bail!(
                "Blank ModelDeploymentCard cannot be used for post-processing, no model_info"
            );
        };
        let model_info = model_info.get_model_info()?;
        let runtime_config = mdc.runtime_config.clone();

        Ok(Arc::new(Self {
            mdcsum,
            _model_info: model_info,
            _runtime_config: runtime_config,
        }))
    }

    /// Transform BackendOutput to PostprocessedResponse
    pub fn transform_backend_output(
        &self,
        backend_output: BackendOutput,
    ) -> Result<PostprocessedResponse> {
        Ok(PostprocessedResponse {
            mdcsum: self.mdcsum.clone(),
            index: backend_output.index.map(|i| i as usize),
            finish_reason: backend_output.finish_reason,
            token_ids: backend_output.token_ids,
            tokens: Some(backend_output.tokens),
            text: backend_output.text,
            cum_log_probs: backend_output.cum_log_probs,
            parsed_components: None, // TODO: Implement reasoning/tool parsing
        })
    }

    /// Transform a stream of BackendOutput to PostprocessedResponse
    pub fn transform_backend_stream<S>(
        &self,
        stream: S,
    ) -> impl Stream<Item = Annotated<PostprocessedResponse>> + Send
    where
        S: Stream<Item = Annotated<BackendOutput>> + Send + 'static,
    {
        let mdcsum = self.mdcsum.clone();
        stream.map(move |annotated_backend| {
            annotated_backend.map_data(|backend_output| {
                Ok(PostprocessedResponse {
                    mdcsum: mdcsum.clone(),
                    index: backend_output.index.map(|i| i as usize),
                    finish_reason: backend_output.finish_reason,
                    token_ids: backend_output.token_ids,
                    tokens: Some(backend_output.tokens),
                    text: backend_output.text,
                    cum_log_probs: backend_output.cum_log_probs,
                    parsed_components: None, // TODO: Implement reasoning/tool parsing
                })
            })
        })
    }
}

/// Operator implementation for Chat Completions
/// This transforms BackendOutput -> NvCreateChatCompletionStreamResponse
#[async_trait]
impl
    Operator<
        ManyOut<Annotated<BackendOutput>>, // Input from backend
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, // Output to client
        ManyOut<Annotated<PostprocessedResponse>>, // Forward edge
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, // Backward edge
    > for OpenAIPostprocessor
{
    async fn generate(
        &self,
        request: ManyOut<Annotated<BackendOutput>>,
        _next: Arc<
            dyn AsyncEngine<
                    ManyOut<Annotated<PostprocessedResponse>>,
                    ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
                    Error,
                >,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        // Extract context from the input stream
        let context = request.context();

        // Clone the necessary data to avoid lifetime issues
        let mdcsum = self.mdcsum.clone();

        // Transform BackendOutput directly to final ChatCompletion format
        let final_stream = request.map(move |annotated_backend| {
            annotated_backend.map_data(|backend_output| {
                // For now, create a basic PostprocessedResponse and then convert
                let _postprocessed = PostprocessedResponse {
                    mdcsum: mdcsum.clone(),
                    index: backend_output.index.map(|i| i as usize),
                    finish_reason: backend_output.finish_reason,
                    token_ids: backend_output.token_ids,
                    tokens: Some(backend_output.tokens),
                    text: backend_output.text,
                    cum_log_probs: backend_output.cum_log_probs,
                    parsed_components: None, // TODO: Implement reasoning/tool parsing
                };
                // TODO: Convert PostprocessedResponse to NvCreateChatCompletionStreamResponse
                todo!("Implement PostprocessedResponse to NvCreateChatCompletionStreamResponse conversion")
            })
        });

        Ok(ResponseStream::new(Box::pin(final_stream), context))
    }
}

/// Operator implementation for Completions  
/// This transforms BackendOutput -> NvCreateCompletionResponse
#[async_trait]
impl
    Operator<
        ManyOut<Annotated<BackendOutput>>, // Input from backend
        ManyOut<Annotated<NvCreateCompletionResponse>>, // Output to client
        ManyOut<Annotated<PostprocessedResponse>>, // Forward edge
        ManyOut<Annotated<NvCreateCompletionResponse>>, // Backward edge
    > for OpenAIPostprocessor
{
    async fn generate(
        &self,
        request: ManyOut<Annotated<BackendOutput>>,
        _next: Arc<
            dyn AsyncEngine<
                    ManyOut<Annotated<PostprocessedResponse>>,
                    ManyOut<Annotated<NvCreateCompletionResponse>>,
                    Error,
                >,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        // Extract context from the input stream
        let context = request.context();

        // Clone the necessary data to avoid lifetime issues
        let mdcsum = self.mdcsum.clone();

        // Transform BackendOutput directly to final Completion format
        let final_stream = request.map(move |annotated_backend| {
            annotated_backend.map_data(|backend_output| {
                // For now, create a basic PostprocessedResponse and then convert
                let _postprocessed = PostprocessedResponse {
                    mdcsum: mdcsum.clone(),
                    index: backend_output.index.map(|i| i as usize),
                    finish_reason: backend_output.finish_reason,
                    token_ids: backend_output.token_ids,
                    tokens: Some(backend_output.tokens),
                    text: backend_output.text,
                    cum_log_probs: backend_output.cum_log_probs,
                    parsed_components: None, // TODO: Implement reasoning/tool parsing
                };

                // TODO: Convert PostprocessedResponse to NvCreateCompletionResponse
                todo!("Implement PostprocessedResponse to NvCreateCompletionResponse conversion")
            })
        });

        Ok(ResponseStream::new(Box::pin(final_stream), context))
    }
}

/// Operator implementation for Embeddings
/// This transforms EmbeddingsEngineOutput -> NvCreateEmbeddingResponse  
#[async_trait]
impl
    Operator<
        ManyOut<Annotated<EmbeddingsEngineOutput>>, // Input from backend
        ManyOut<Annotated<NvCreateEmbeddingResponse>>, // Output to client
        ManyOut<Annotated<EmbeddingsEngineOutput>>, // Forward edge (pass-through)
        ManyOut<Annotated<NvCreateEmbeddingResponse>>, // Backward edge
    > for OpenAIPostprocessor
{
    async fn generate(
        &self,
        request: ManyOut<Annotated<EmbeddingsEngineOutput>>,
        _next: Arc<
            dyn AsyncEngine<
                    ManyOut<Annotated<EmbeddingsEngineOutput>>,
                    ManyOut<Annotated<NvCreateEmbeddingResponse>>,
                    Error,
                >,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateEmbeddingResponse>>, Error> {
        // Extract context from the input stream
        let context = request.context();

        // Transform EmbeddingsEngineOutput to NvCreateEmbeddingResponse directly
        let final_stream = request.map(|annotated| {
            annotated.map_data(|_engine_output| {
                // TODO: Convert EmbeddingsEngineOutput to NvCreateEmbeddingResponse
                // This should be similar to what's in preprocessor.rs transform_embedding_postprocessor_stream
                todo!("Implement EmbeddingsEngineOutput to NvCreateEmbeddingResponse conversion")
            })
        });

        Ok(ResponseStream::new(Box::pin(final_stream), context))
    }
}
