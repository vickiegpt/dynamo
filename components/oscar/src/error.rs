// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

/// Result type for Oscar operations.
pub type OscarResult<T> = Result<T, OscarError>;

/// Errors that can occur during Oscar operations.
#[derive(Error, Debug)]
pub enum OscarError {
    #[error("Object too large: {size} bytes (max {max} bytes)")]
    ObjectTooLarge { size: usize, max: usize },

    #[error("Storage error: {0}")]
    Storage(#[from] dynamo_runtime::storage::key_value_store::StorageError),

    #[error("Hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("Object not found: {hash}")]
    ObjectNotFound { hash: String },

    #[error("Invalid operation: {reason}")]
    InvalidOperation { reason: String },

    #[error("Concurrency error: {reason}")]
    Concurrency { reason: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}