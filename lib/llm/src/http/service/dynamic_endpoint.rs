// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use crate::types::Annotated;
use axum::{
    Json, Router,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
use dynamo_runtime::instances::list_all_instances;
use dynamo_runtime::{DistributedRuntime, Runtime, component::Client};
use dynamo_runtime::{pipeline::PushRouter, stream::StreamExt};
use std::sync::Arc;

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

async fn inner_dynamic_endpoint_handler(
    state: Arc<service_v2::State>,
    path: String,
) -> Result<impl IntoResponse, &'static str> {
    let etcd_client = state
        .etcd_client()
        .ok_or_else(|| "Failed to get etcd client")?;

    let instances = list_all_instances(etcd_client)
        .await
        .map_err(|_| "Failed to get instances")?;

    let dynamic_endpoints = instances
        .iter()
        .filter_map(|instance| instance.http_endpoint_path.clone())
        .collect::<Vec<String>>();

    let fmt_path = format!("/{}", &path);
    if !dynamic_endpoints.contains(&fmt_path) {
        return Err("Dynamic endpoint not found");
    }

    let rt = Runtime::from_current().map_err(|_| "Failed to get runtime")?;
    let drt = DistributedRuntime::from_settings(rt)
        .await
        .map_err(|_| "Failed to get distributed runtime")?;

    let target_instances = instances
        .iter()
        .filter(|instance| instance.http_endpoint_path == Some(fmt_path.clone()))
        .collect::<Vec<_>>();

    let mut target_clients: Vec<Client> = Vec::new();
    for instance in target_instances {
        let ns = drt
            .namespace(instance.namespace.clone())
            .map_err(|_| "Failed to get namespace")?;
        let c = ns
            .component(instance.component.clone())
            .map_err(|_| "Failed to get component")?;
        let ep = c.endpoint(path.clone());
        let client = ep.client().await.map_err(|_| "Failed to get client")?;
        target_clients.push(client);
    }

    let mut all_responses = Vec::new();
    for client in target_clients {
        let router =
            PushRouter::<(), Annotated<serde_json::Value>>::from_client(client, Default::default())
                .await
                .map_err(|_| "Failed to get router")?;

        let mut stream = router
            .round_robin(().into())
            .await
            .map_err(|_| "Failed to route")?;

        while let Some(resp) = stream.next().await {
            all_responses.push(resp);
        }
    }

    Ok(Json(serde_json::json!({
        "responses": all_responses
    })))
}

async fn dynamic_endpoint_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    axum::extract::Path(path): axum::extract::Path<String>,
) -> impl IntoResponse {
    inner_dynamic_endpoint_handler(state, path)
        .await
        .map_err(|err_string| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "message": err_string
                })),
            )
        })
}
