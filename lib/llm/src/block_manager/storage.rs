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

pub mod nixl;

use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    collections::HashMap,
    fmt::Debug,
    ptr::NonNull,
    sync::Arc,
};
use thiserror::Error;

// Re-export cudarc types we use
use cudarc::driver::{sys, CudaContext, DriverError};

/// Result type for storage operations
pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// Represents the type of storage used for a block
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageType {
    Device(Arc<CudaContext>),
    Pinned,
    System,
    Null,

    Nixl,
}

pub enum StorageLocality {
    Local,

    // todo: add a nixl agent/bytes identifier
    // perhaps this is an enum. other options could be a etcd path to a
    // keyval object with nixl metadata
    Remote,
}

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

    #[error("Registration key already exists: {0}")]
    RegistrationKeyExists(String),

    #[error("Handle not found for key: {0}")]
    HandleNotFound(String),
}

/// Core storage trait that provides access to memory regions
pub trait Storage: Debug + Send + Sync + 'static {
    /// Returns the type of storage
    fn storage_type(&self) -> StorageType;

    /// Returns the address of the storage
    fn addr(&self) -> u64;

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

pub trait RegistationHandle: std::any::Any + Send + Sync + 'static {
    /// Release the [RegistationHandle].
    /// This should be called when the external registration of this storage
    /// is no longer needed.
    ///
    /// Note: All [RegistrationHandle]s should be explicitly released before
    /// the [Storage] is dropped.
    fn release(&mut self);
}

pub trait RegisterableStorage: Storage + Send + Sync + 'static {
    /// Register a handle with a key
    /// If a handle with the same key already exists, an error is returned
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError>;

    /// Check if a handle is registered with a key
    fn is_registered(&self, key: &str) -> bool;

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle>;
}

#[derive(Default)]
pub struct RegistrationHandles {
    handles: HashMap<String, Box<dyn RegistationHandle>>,
}

impl RegistrationHandles {
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        let key = key.to_string();
        if self.handles.contains_key(&key) {
            return Err(StorageError::RegistrationKeyExists(key));
        }
        self.handles.insert(key, handle);
        Ok(())
    }

    fn release(&mut self) {
        for handle in self.handles.values_mut() {
            handle.release();
        }
        self.handles.clear();
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.contains_key(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.get(key).map(|h| h.as_ref())
    }
}

impl std::fmt::Debug for RegistrationHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RegistrationHandles {{ count: {:?} }}",
            self.handles.len()
        )
    }
}

impl Drop for RegistrationHandles {
    fn drop(&mut self) {
        if !self.handles.is_empty() {
            panic!("RegistrationHandles dropped with {} handles remaining; RegistrationHandles::release() needs to be explicitly called", self.handles.len());
        }
    }
}

/// Trait for types that can allocate specific Storage implementations.
pub trait StorageAllocator<S: Storage>: Send + Sync {
    /// Allocate storage of the specific type `S` with the given size in bytes.
    fn allocate(&self, size: usize) -> Result<S, StorageError>;
}

/// System memory storage implementation using pinned memory
#[derive(Debug)]
pub struct SystemStorage {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
    handles: RegistrationHandles,
}

unsafe impl Send for SystemStorage {}
unsafe impl Sync for SystemStorage {}

impl SystemStorage {
    /// Create a new system storage with the given size
    ///
    /// # Safety
    /// This function allocates memory that will be freed when the SystemStorage is dropped.
    pub fn new(size: usize) -> Result<Self, StorageError> {
        // Create layout for the allocation, ensuring proper alignment
        let layout =
            Layout::array::<u8>(size).map_err(|e| StorageError::AllocationFailed(e.to_string()))?;

        // Allocate zeroed memory
        let ptr = unsafe {
            NonNull::new(alloc_zeroed(layout))
                .ok_or_else(|| StorageError::AllocationFailed("memory allocation failed".into()))?
        };

        Ok(Self {
            ptr,
            layout,
            len: size,
            handles: RegistrationHandles::new(),
        })
    }
}

impl Drop for SystemStorage {
    fn drop(&mut self) {
        self.handles.release();
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

impl Storage for SystemStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::System
    }

    fn addr(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn size(&self) -> usize {
        self.len
    }

    fn is_host_accessible(&self) -> bool {
        true
    }

    unsafe fn as_ptr(&self) -> Option<*const u8> {
        Some(self.ptr.as_ptr())
    }

    unsafe fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        Some(self.ptr.as_ptr())
    }
}

impl RegisterableStorage for SystemStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

/// Allocator for SystemStorage
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemAllocator;

impl StorageAllocator<SystemStorage> for SystemAllocator {
    fn allocate(&self, size: usize) -> Result<SystemStorage, StorageError> {
        SystemStorage::new(size)
    }
}

/// Pinned host memory storage using CUDA page-locked memory
#[derive(Debug)]
pub struct PinnedStorage {
    ptr: u64,
    size: usize,
    handles: RegistrationHandles,
}

impl PinnedStorage {
    /// Create a new pinned storage with the given size
    pub fn new(ctx: &Arc<CudaContext>, size: usize) -> Result<Self, StorageError> {
        unsafe {
            ctx.bind_to_thread().map_err(StorageError::Cuda)?;

            let ptr = cudarc::driver::result::malloc_host(size, sys::CU_MEMHOSTALLOC_WRITECOMBINED)
                .map_err(StorageError::Cuda)?;

            let ptr = ptr as *mut u8;
            assert!(!ptr.is_null(), "Failed to allocate pinned memory");
            assert!(ptr.is_aligned(), "Pinned memory is not aligned");
            assert!(size < isize::MAX as usize);

            let ptr = ptr as u64;
            Ok(Self {
                ptr,
                size,
                handles: RegistrationHandles::new(),
            })
        }
    }
}

impl Drop for PinnedStorage {
    fn drop(&mut self) {
        self.handles.release();
        unsafe { cudarc::driver::result::free_host(self.ptr as _) }.unwrap();
    }
}

impl Storage for PinnedStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Pinned
    }

    fn addr(&self) -> u64 {
        self.ptr
    }

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

impl RegisterableStorage for PinnedStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

/// Allocator for PinnedStorage
pub struct PinnedAllocator {
    ctx: Arc<CudaContext>,
}

impl Default for PinnedAllocator {
    fn default() -> Self {
        Self {
            ctx: CudaContext::new(0).expect("Failed to create CUDA context"),
        }
    }
}

impl PinnedAllocator {
    pub fn try_new(device_id: usize) -> Result<Self, StorageError> {
        Ok(Self {
            ctx: CudaContext::new(device_id).map_err(StorageError::Cuda)?,
        })
    }
}

impl StorageAllocator<PinnedStorage> for PinnedAllocator {
    fn allocate(&self, size: usize) -> Result<PinnedStorage, StorageError> {
        PinnedStorage::new(&self.ctx, size)
    }
}

/// CUDA device memory storage
#[derive(Debug)]
pub struct DeviceStorage {
    ptr: u64,
    size: usize,
    ctx: Arc<CudaContext>,
    handles: RegistrationHandles,
}

impl DeviceStorage {
    /// Create a new device storage with the given size
    pub fn new(ctx: &Arc<CudaContext>, size: usize) -> Result<Self, StorageError> {
        ctx.bind_to_thread().map_err(StorageError::Cuda)?;
        let ptr = unsafe { cudarc::driver::result::malloc_sync(size).map_err(StorageError::Cuda)? };

        Ok(Self {
            ptr,
            size,
            ctx: ctx.clone(),
            handles: RegistrationHandles::new(),
        })
    }

    /// Get the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl Storage for DeviceStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Device(self.ctx.clone())
    }

    fn addr(&self) -> u64 {
        self.ptr
    }

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
        self.handles.release();
        unsafe { cudarc::driver::result::free_sync(self.ptr as _) }.unwrap();
    }
}

impl RegisterableStorage for DeviceStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

pub struct DeviceAllocator {
    ctx: Arc<CudaContext>,
}

impl Default for DeviceAllocator {
    fn default() -> Self {
        Self {
            ctx: CudaContext::new(0).expect("Failed to create CUDA context"),
        }
    }
}

impl DeviceAllocator {
    pub fn try_new(device_id: usize) -> Result<Self, StorageError> {
        Ok(Self {
            ctx: CudaContext::new(device_id).map_err(StorageError::Cuda)?,
        })
    }
}

impl StorageAllocator<DeviceStorage> for DeviceAllocator {
    fn allocate(&self, size: usize) -> Result<DeviceStorage, StorageError> {
        DeviceStorage::new(&self.ctx, size)
    }
}

pub mod tests {
    use super::*;

    #[derive(Debug)]
    pub struct NullDeviceStorage {
        size: u64,
    }

    impl NullDeviceStorage {
        pub fn new(size: u64) -> Self {
            Self { size }
        }
    }

    impl Storage for NullDeviceStorage {
        fn storage_type(&self) -> StorageType {
            StorageType::Null
        }

        fn addr(&self) -> u64 {
            0
        }

        fn size(&self) -> usize {
            self.size as usize
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

    pub struct NullDeviceAllocator;

    impl StorageAllocator<NullDeviceStorage> for NullDeviceAllocator {
        fn allocate(&self, size: usize) -> Result<NullDeviceStorage, StorageError> {
            Ok(NullDeviceStorage::new(size as u64))
        }
    }

    #[derive(Debug)]
    pub struct NullHostStorage {
        size: u64,
    }

    impl NullHostStorage {
        pub fn new(size: u64) -> Self {
            Self { size }
        }
    }

    impl Storage for NullHostStorage {
        fn storage_type(&self) -> StorageType {
            StorageType::Null
        }

        fn addr(&self) -> u64 {
            0
        }

        fn size(&self) -> usize {
            self.size as usize
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

    pub struct NullHostAllocator;

    impl StorageAllocator<NullHostStorage> for NullHostAllocator {
        fn allocate(&self, size: usize) -> Result<NullHostStorage, StorageError> {
            Ok(NullHostStorage::new(size as u64))
        }
    }
}
