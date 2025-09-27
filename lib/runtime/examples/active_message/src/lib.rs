// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::active_message::{
    handler::{ActiveMessageContext, ResponseHandler},
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
    async fn handle(
        &self,
        input: Bytes,
        _ctx: ActiveMessageContext,
    ) -> Result<Bytes> {
        let request: ComputeRequest = serde_json::from_slice(&input)?;

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

        let response = ComputeResponse { result };
        Ok(Bytes::from(serde_json::to_vec(&response)?))
    }

    fn name(&self) -> &str {
        "compute"
    }
}
