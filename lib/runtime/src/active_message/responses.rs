// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response types for built-in ActiveMessage handlers.

use serde::{Deserialize, Serialize};

/// Response from the _register_service handler
#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterServiceResponse {
    pub registered: bool,
    pub instance_id: String,
    pub endpoint: String,
}

/// Response from the _list_handlers handler
#[derive(Debug, Serialize, Deserialize)]
pub struct ListHandlersResponse {
    pub handlers: Vec<String>,
}

/// Response from the _wait_for_handler handler
#[derive(Debug, Serialize, Deserialize)]
pub struct WaitForHandlerResponse {
    pub handler_name: String,
    pub available: bool,
}

/// Response from the _health_check handler
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub status: String,
    pub timestamp: u64,
}
