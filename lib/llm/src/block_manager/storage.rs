// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Storage management for block manager.
//!
//! This module provides traits and implementations for managing storage of KV blocks.
//! It handles system memory, pinned memory, device memory, and remote (NIXL) storage,
//! with a focus on safety and performance.

use std::{fmt::Debug, sync::Arc};
use thiserror::Error;

// Re-export cudarc types we use
use cudarc::driver::{sys, CudaContext, DriverError};

/// Result type for storage operations
pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// Errors that can occur during storage operations
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Storage allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Storage not accessible: {0}")]
    NotAccessible(String),

    #[error("Invalid storage configuration: {0}")]
    InvalidConfig(String),

    #[error("Storage operation failed: {0}")]
    OperationFailed(String),

    #[error("CUDA error: {0}")]
    Cuda(#[from] DriverError),
}

/// Result type for storage operations
pub type Result<T> = std::result::Result<T, StorageError>;

/// Core storage trait that provides access to memory regions
pub trait Storage: Debug + Send + Sync + 'static {
    /// Returns the total size of the storage in bytes
    fn size(&self) -> usize;

    /// Returns true if the storage is accessible by the host/cpu portion
    /// of the application.
    fn is_host_accessible(&self) -> bool;

    /// Get a raw pointer to the storage
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the storage is dropped
    /// - Access patterns respect the storage's thread safety model
    unsafe fn as_ptr(&self) -> Option<*const u8>;

    /// Get a raw mutable pointer to the storage
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the storage is dropped
    /// - No other references exist while the pointer is in use
    /// - Access patterns respect the storage's thread safety model
    unsafe fn as_mut_ptr(&mut self) -> Option<*mut u8>;
}

/// System memory storage implementation using a Vec<u8>
#[derive(Debug)]
pub struct SystemStorage {
    data: Vec<u8>,
}

impl SystemStorage {
    /// Create a new system storage with the given size
    pub fn new(size: usize) -> Result<Self> {
        let mut data = Vec::with_capacity(size);
        // Initialize to zero to ensure consistent behavior
        data.resize(size, 0);
        Ok(Self { data })
    }
}

impl Storage for SystemStorage {
    fn size(&self) -> usize {
        self.data.len()
    }

    fn is_host_accessible(&self) -> bool {
        true
    }

    unsafe fn as_ptr(&self) -> Option<*const u8> {
        Some(self.data.as_ptr())
    }

    unsafe fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        Some(self.data.as_mut_ptr())
    }
}

/// Pinned host memory storage using CUDA page-locked memory
#[derive(Debug)]
pub struct PinnedStorage {
    ptr: u64,
    size: usize,
}

impl PinnedStorage {
    /// Create a new pinned storage with the given size
    pub fn new(ctx: &Arc<CudaContext>, size: usize) -> Result<Self> {
        unsafe {
            ctx.bind_to_thread().map_err(StorageError::Cuda)?;

            let ptr = cudarc::driver::result::malloc_host(size, sys::CU_MEMHOSTALLOC_WRITECOMBINED)
                .map_err(StorageError::Cuda)?;

            let ptr = ptr as *mut u8;
            assert!(!ptr.is_null(), "Failed to allocate pinned memory");
            assert!(ptr.is_aligned(), "Pinned memory is not aligned");
            assert!(size < isize::MAX as usize);

            let ptr = ptr as u64;
            Ok(Self { ptr, size })
        }
    }
}

impl Drop for PinnedStorage {
    fn drop(&mut self) {
        unsafe { cudarc::driver::result::free_host(self.ptr as _) }.unwrap();
    }
}

impl Storage for PinnedStorage {
    fn size(&self) -> usize {
        self.size
    }

    fn is_host_accessible(&self) -> bool {
        true
    }

    unsafe fn as_ptr(&self) -> Option<*const u8> {
        Some(self.ptr as *const u8)
    }

    unsafe fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        Some(self.ptr as *mut u8)
    }
}

/// CUDA device memory storage
#[derive(Debug)]
pub struct DeviceStorage {
    ptr: u64,
    size: usize,
    ctx: Arc<CudaContext>,
}

impl DeviceStorage {
    /// Create a new device storage with the given size
    pub fn new(ctx: Arc<CudaContext>, size: usize) -> Result<Self> {
        let ptr = unsafe { cudarc::driver::result::malloc_sync(size).map_err(StorageError::Cuda)? };

        Ok(Self { ptr, size, ctx })
    }

    /// Get the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl Storage for DeviceStorage {
    fn size(&self) -> usize {
        self.size
    }

    fn is_host_accessible(&self) -> bool {
        // Device memory is not directly accessible from host
        false
    }

    unsafe fn as_ptr(&self) -> Option<*const u8> {
        Some(self.ptr as *const u8)
    }

    unsafe fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        Some(self.ptr as *mut u8)
    }
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        unsafe { cudarc::driver::result::free_sync(self.ptr as _) }.unwrap();
    }
}

/// Remote storage implementation using NIXL
#[derive(Debug)]
pub struct NixlStorage {
    metadata: Arc<NixlMetadata>,
    descriptor: NixlDescriptor,
    size: usize,
}

impl NixlStorage {
    /// Create a new NIXL storage from metadata
    pub fn from_metadata(metadata: Arc<NixlMetadata>) -> Result<Self> {
        let descriptor = metadata.get_descriptor().map_err(|e| {
            StorageError::InvalidConfig(format!("Failed to get NIXL descriptor: {}", e))
        })?;

        let size = metadata.size();

        Ok(Self {
            metadata,
            descriptor,
            size,
        })
    }
}

impl Storage for NixlStorage {
    fn size(&self) -> usize {
        self.size
    }

    fn is_host_accessible(&self) -> bool {
        false
    }

    unsafe fn as_ptr(&self) -> Option<*const u8> {
        unimplemented!()
    }

    unsafe fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        unimplemented!()
    }
}

/// Mock types for compilation - these would be defined elsewhere
#[derive(Debug)]
pub struct NixlMetadata {
    // Implementation details...
}

impl NixlMetadata {
    fn get_descriptor(&self) -> std::result::Result<NixlDescriptor, Box<dyn std::error::Error>> {
        unimplemented!()
    }

    fn size(&self) -> usize {
        unimplemented!()
    }
}

#[derive(Debug)]
pub struct NixlDescriptor {
    // Implementation details...
}

#[derive(Debug, Default)]
pub struct NullStorage {}

impl Storage for NullStorage {
    fn size(&self) -> usize {
        0
    }

    fn is_host_accessible(&self) -> bool {
        false
    }

    unsafe fn as_ptr(&self) -> Option<*const u8> {
        None
    }

    unsafe fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        None
    }
}
