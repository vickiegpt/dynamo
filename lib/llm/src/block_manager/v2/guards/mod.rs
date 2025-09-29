// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard types for type-safe block management
//!
//! This module provides type-safe RAII guards that ensure automatic resource cleanup:
//! - `MutableBlock`: Guards blocks in Reset state
//! - `CompleteBlock`: Guards blocks in Complete state
//! - `ImmutableBlock`: Guards registered blocks with upgrade capability
//! - `WeakBlock`: Weak references to registered blocks
//! - `PrimaryBlock`, `DuplicateBlock`: Internal registered block types

use super::pools::{block::BlockId, registry::BlockRegistrationHandle};
use crate::tokens::SequenceHash;

pub mod complete;
pub mod immutable;
pub mod mutable;
pub mod registered;

pub use complete::CompleteBlock;
pub use immutable::{ImmutableBlock, WeakBlock};
pub use mutable::MutableBlock;
pub(crate) use registered::{DuplicateBlock, PrimaryBlock};

/// Trait for types that can be registered and provide block information
pub trait RegisteredBlock<T>: Send + Sync {
    /// Get the block ID
    fn block_id(&self) -> BlockId;

    /// Get the sequence hash
    fn sequence_hash(&self) -> SequenceHash;

    /// Get the registration handle
    fn registration_handle(&self) -> &BlockRegistrationHandle;
}
