// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../docs/active_message.md")]

pub mod builder;
pub mod client;
pub mod handler;
pub mod manager;
pub mod response;
pub mod responses;
pub mod status;
pub mod transport;
pub(crate) mod utils;

pub use builder::MessageBuilder;
pub use client::ActiveMessageClient;
pub use handler::{AckHandler, ActiveMessage, HandlerType, NoReturnHandler, ResponseHandler};
pub use manager::ActiveMessageManager;
pub use response::SingleResponseSender;
pub use responses::{
    HealthCheckResponse, JoinCohortResponse, ListHandlersResponse, RegisterServiceResponse,
    RemoveServiceResponse, RequestShutdownResponse, WaitForHandlerResponse,
};
pub use status::{DetachedConfirm, MessageStatus, SendAndConfirm, WithResponse};
pub use transport::Transport;

pub mod zmq;
