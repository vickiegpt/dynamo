// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block Manager V2 - EXPERIMENTAL
//!
//! A completely redesigned block management system with:
//! - Type-safe state transitions (Reset → Complete → Registered)
//! - Async batched return processing with controllable stepping
//! - Compile-time prevention of accessing registered mutable blocks
//! - Comprehensive testing support for race conditions
//!
//! NOTE: This module is currently experimental and under development.
//! It implements a simplified Block<T, State> API that differs from the
//! main codebase's Block<Storage, LocalityProvider, Metadata> API.

// Core modules
// pub mod block;
// pub mod blocks;
// pub mod free_list;
// pub mod registry;

// V2 implementation modules - now standalone
// #pub mod async_pool;
// pub mod builder;
// pub mod manager;
pub mod pools;

pub mod manager;
// pub mod progress_engine;
// pub mod state;
// pub mod wrappers;

// // Test module
// #[cfg(test)]
// pub mod tests;

// // Public exports
// pub use crate::tokens::SequenceHash;
// pub use block::BlockId;

// // Re-export key types from the new implementation
// pub use blocks::block::{Block, BlockId};
// pub use blocks::registry::{BlockRegistrationHandle, BlockRegistry};
// pub use pools::{ImmutableBlock, MutableBlock, MutableBlockState, RegisteredPool};
