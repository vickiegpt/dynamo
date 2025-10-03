// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use axum::{
    Json, Router,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
use std::sync::Arc;

pub const DYNAMIC_ENDPOINT_PATH: &str = "dynamic_endpoint";

pub fn dynamic_endpoint_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let wildcard_path = "/{*path}";
    let path = path.unwrap_or_else(|| wildcard_path.to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::POST, &path)];

    let router = Router::new()
        .route(&path, post(dynamic_endpoint_handler))
        .with_state(state);

    (docs, router)
}

async fn dynamic_endpoint_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    let mut dynamic_endpoints: Vec<String> = Vec::new();

    let Some(etcd_client) = state.etcd_client() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "message": "Failed to get etcd client"
            })),
        );
    };

    let kvs = match etcd_client
        .kv_get_prefix(format!("{}/", DYNAMIC_ENDPOINT_PATH))
        .await
    {
        Ok(kvs) => kvs,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "message": "Failed to get dynamic endpoints"
                })),
            );
        }
    };

    for kv in kvs {
        match serde_json::from_slice::<String>(kv.value()) {
            Ok(path) => dynamic_endpoints.push(path),
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "message": "Failed to parse dynamic endpoint"
                    })),
                );
            }
        }
    }

    return (
        StatusCode::OK,
        Json(serde_json::json!({
            "dynamic_endpoints": dynamic_endpoints
        })),
    );
}
