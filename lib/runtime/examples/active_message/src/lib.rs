// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use dynamo_runtime::active_message::{
    client::ActiveMessageClient,
    handler::{ActiveMessage, ResponseHandler},
};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Serialize, Deserialize)]
pub struct ComputeRequest {
    pub x: i32,
    pub y: i32,
    pub operation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComputeResponse {
    pub result: i32,
}

#[derive(Debug)]
pub struct ComputeHandler;

#[async_trait]
impl ResponseHandler for ComputeHandler {
    type Response = ComputeResponse;

    async fn handle(
        &self,
        message: ActiveMessage,
        _client: &dyn ActiveMessageClient,
    ) -> Result<Self::Response> {
        let request: ComputeRequest = message.deserialize()?;

        let result = match request.operation.as_str() {
            "add" => request.x + request.y,
            "multiply" => request.x * request.y,
            _ => {
                info!("Unknown operation: {}", request.operation);
                anyhow::bail!("Unknown operation: {}", request.operation);
            }
        };

        info!(
            "Computed {} {} {} = {}",
            request.x, request.operation, request.y, result
        );

        Ok(ComputeResponse { result })
    }

    fn name(&self) -> &str {
        "compute"
    }
}
