// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use std::sync::Arc;

use super::{openai::ErrorResponse, service_v2, RouteDoc};
use crate::protocols::{
    openai::completions::CompletionResponse, token_completions::DynamoTokenCompletionRequest,
};
use dynamo_runtime::pipeline::Context;

/// Token completion endpoint - goes directly to backend, bypassing OpenAI preprocessing
#[tracing::instrument(skip_all)]
async fn token_completions(
    State(state): State<Arc<service_v2::State>>,
    Json(request): Json<DynamoTokenCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let model = &request.model;
    let request_id = uuid::Uuid::new_v4().to_string();

    // Get token completion engine (same pattern as OpenAI!)
    let engine = state
        .manager()
        .get_token_completion_engine(model)
        .map_err(|_| ErrorResponse::model_not_found())?;

    // Execute request (same pattern as OpenAI!)
    let request_context = Context::with_id(request, request_id);
    let stream = engine
        .generate(request_context)
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Generation failed"))?;

    // Aggregate response (same as OpenAI!)
    let response = CompletionResponse::from_annotated_stream(stream.into())
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Failed to process response"))?;

    Ok(Json(response).into_response())
}

/// Router for token completion endpoint
pub fn token_completions_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/tokens/completion".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(token_completions))
        .with_state(state);
    (vec![doc], router)
}
