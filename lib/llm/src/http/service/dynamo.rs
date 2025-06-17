use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::post,
    Json, Router,
};
use futures::StreamExt;
use std::sync::Arc;

use super::{
    openai::{
        check_ready, monitor_for_disconnects, process_event_converter, ErrorResponse,
        EventConverter,
    },
    service_v2, RouteDoc,
};
use crate::protocols::dynamo::DynamoTokenCompletionRequest;

/// Dynamo Token Completions Handler - mirrors OpenAI completions exactly
#[tracing::instrument(skip_all)]
async fn token_completions(
    State(state): State<Arc<service_v2::State>>,
    Json(request): Json<DynamoTokenCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    check_ready(&state)?;

    let request_id = uuid::Uuid::new_v4().to_string();
    let streaming = request.stream.unwrap_or(false);
    let model = request.model.clone();

    // Always enable streaming internally (same as OpenAI handler)
    let mut request = request;
    request.stream = Some(true);

    // Get completions engine - using Backend models since they accept CompletionRequest
    let engine = state
        .manager()
        .get_completions_engine(&model)
        .map_err(|_| ErrorResponse::model_not_found())?;

    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &model,
        crate::http::service::metrics::Endpoint::Completions,
        streaming,
    );

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // Convert to CompletionRequest with TokenIds (this is the key conversion)
    let completion_request: crate::protocols::common::CompletionRequest = request
        .try_into()
        .map_err(|e| ErrorResponse::internal_server_error(&format!("Invalid request: {}", e)))?;

    // Create OpenAI-compatible request for the engine
    let openai_request = crate::protocols::openai::completions::NvCreateCompletionRequest {
        inner: async_openai::types::CreateCompletionRequest {
            model: model.clone(),
            prompt: async_openai::types::Prompt::String("".to_string()), // Dummy prompt since we use tokens
            stream: Some(true),
            max_tokens: completion_request.stop_conditions.max_tokens,
            temperature: completion_request.sampling_options.temperature,
            top_p: completion_request.sampling_options.top_p,
            n: completion_request.sampling_options.n.map(|n| n as u8),
            frequency_penalty: completion_request.sampling_options.frequency_penalty,
            presence_penalty: completion_request.sampling_options.presence_penalty,
            best_of: completion_request.sampling_options.best_of.map(|b| b as u8),
            ..Default::default()
        },
        nvext: None,
    };

    // Setup context with OpenAI format
    let request = dynamo_runtime::pipeline::Context::with_id(openai_request, request_id.clone());

    // Generate - this goes to existing completions engine
    let openai_stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Failed to generate completions"))?;

    let ctx = openai_stream.context();

    // The engine returns CompletionResponse stream already - no conversion needed
    let completion_stream: crate::protocols::DataStream<
        crate::protocols::Annotated<crate::protocols::openai::completions::CompletionResponse>,
    > = openai_stream.into();

    if streaming {
        let stream = completion_stream.map(move |response| {
            process_event_converter(EventConverter::from(response), &mut response_collector)
        });
        let stream = monitor_for_disconnects(stream.boxed(), ctx, inflight_guard).await;

        let mut sse_stream = Sse::new(stream);
        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        // Use the OpenAI completion aggregator for non-streaming
        let response =
            crate::protocols::openai::completions::CompletionResponse::from_annotated_stream(
                completion_stream,
            )
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to fold token completions stream for {}: {:?}",
                    request_id,
                    e
                );
                ErrorResponse::internal_server_error("Failed to fold token completions stream")
            })?;

        inflight_guard.mark_ok();
        Ok(Json(response).into_response())
    }
}

pub fn token_completions_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/dynamo/completions".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(token_completions))
        .with_state(state);
    (vec![doc], router)
}
