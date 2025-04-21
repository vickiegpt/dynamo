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

use super::*;

use cudarc::driver::sys;

pub use cudarc::driver::{CudaContext, DriverError};

/// Pinned host memory storage using CUDA page-locked memory
#[derive(Debug)]
pub struct PinnedStorage {
    ptr: u64,
    size: usize,
}

impl PinnedStorage {
    /// Create a new pinned storage with the given size
    pub fn new(size: usize) -> Result<Self> {
        unsafe {
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

/// Pinned host memory allocator
#[derive(Debug, Default)]
pub struct PinnedAllocator {}

impl PinnedAllocator {
    /// Create a new pinned allocator
    pub fn try_new(_: usize) -> Result<Self> {
        Ok(Self {})
    }
}

impl StorageAllocator<PinnedStorage> for PinnedAllocator {
    fn allocate(&self, size: usize) -> Result<PinnedStorage> {
        PinnedStorage::new(size)
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
    pub fn new(ctx: &Arc<CudaContext>, size: usize) -> Result<Self> {
        ctx.bind_to_thread().map_err(StorageError::Cuda)?;
        let ptr = unsafe { cudarc::driver::result::malloc_sync(size).map_err(StorageError::Cuda)? };

        Ok(Self {
            ptr,
            size,
            ctx: ctx.clone(),
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
        unsafe { cudarc::driver::result::free_sync(self.ptr as _) }.unwrap();
    }
}

/// CUDA device memory allocator
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
    /// Create a new device allocator for the given device id
    pub fn try_new(device_id: usize) -> Result<Self> {
        Ok(Self {
            ctx: CudaContext::new(device_id).map_err(StorageError::Cuda)?,
        })
    }
}

impl StorageAllocator<DeviceStorage> for DeviceAllocator {
    fn allocate(&self, size: usize) -> Result<DeviceStorage> {
        DeviceStorage::new(&self.ctx, size)
    }
}
