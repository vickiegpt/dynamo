// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::broadcast;

use super::client::ActiveMessageClient;
use super::handler::{HandlerEvent, HandlerId, HandlerType};
use crate::utils::tasks::tracker::TaskTracker;

#[derive(Default)]
pub struct HandlerConfig {
    pub task_tracker: Option<TaskTracker>,
    pub max_concurrent_messages: Option<usize>,
}

impl HandlerConfig {
    pub fn with_task_tracker(mut self, tracker: TaskTracker) -> Self {
        self.task_tracker = Some(tracker);
        self
    }

    pub fn with_max_concurrent_messages(mut self, max: usize) -> Self {
        self.max_concurrent_messages = Some(max);
        self
    }
}

#[async_trait]
pub trait ActiveMessageManager: Send + Sync {
    fn client(&self) -> Arc<dyn ActiveMessageClient>;

    /// Register a handler using the handler type system
    async fn register_handler_typed(
        &self,
        handler: HandlerType,
        config: Option<HandlerConfig>,
    ) -> Result<()>;

    async fn deregister_handler(&self, name: &str) -> Result<()>;

    async fn list_handlers(&self) -> Vec<HandlerId>;

    fn handler_events(&self) -> broadcast::Receiver<HandlerEvent>;

    async fn shutdown(&self) -> Result<()>;
}
