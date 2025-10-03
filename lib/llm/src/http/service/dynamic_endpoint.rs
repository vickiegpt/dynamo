// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use axum::{
    Json, Router,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
use dynamo_runtime::instances::list_all_instances;
use dynamo_runtime::{pipeline::PushRouter, stream::StreamExt};
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
    axum::extract::Path(path): axum::extract::Path<String>,
) -> impl IntoResponse {
    let Some(etcd_client) = state.etcd_client() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "message": "Failed to get etcd client"
            })),
        );
    };

    let instances = match list_all_instances(etcd_client).await {
        Ok(instances) => instances,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "message": "Failed to get instances"
                })),
            );
        }
    };

    let dynamic_endpoints = instances
        .iter()
        .filter_map(|instance| instance.http_endpoint_path.clone())
        .collect::<Vec<String>>();

    let path = format!("/{}", &path);
    if dynamic_endpoints.contains(&path) {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "message": "Dynamic endpoint found"
            })),
        );
    } else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "message": "Dynamic endpoint not found"
            })),
        );
    }
}
