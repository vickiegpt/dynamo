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

//! Block storage management.
//!
//! This module provides the implementation for managing collections of blocks
//! and their storage. It handles the relationship between storage, layout,
//! and individual blocks.

use super::{BlockError, BlockStorage, Storage, StorageError};

/// Storage view that provides safe access to a region of storage
#[derive(Debug)]
pub struct BlockView<'a, S: Storage> {
    block_storage: &'a BlockStorage<S>,
    offset: usize,
    size: usize,
}

impl<'a, S> BlockView<'a, S>
where
    S: Storage,
{
    /// Create a new storage view
    ///
    /// # Safety
    /// The caller must ensure:
    /// - offset + size <= storage.size()
    /// - The view does not outlive the storage
    pub unsafe fn new(
        block_storage: &'a BlockStorage<S>,
        offset: usize,
        size: usize,
    ) -> Result<Self, BlockError> {
        if offset + size > block_storage.storage.size() {
            return Err(BlockError::Storage(StorageError::InvalidConfig(
                "View extends beyond storage bounds".into(),
            )));
        }

        Ok(Self {
            block_storage,
            offset,
            size,
        })
    }

    /// Get a raw pointer to the view's data
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the view is dropped
    /// - Access patterns respect the storage's thread safety model
    pub unsafe fn as_ptr(&self) -> *const u8 {
        let base = self
            .block_storage
            .storage
            .as_ptr()
            .expect("Storage became inaccessible");
        base.add(self.offset)
    }

    /// Size of the view in bytes
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Mutable storage view that provides exclusive access to a region of storage
#[derive(Debug)]
pub struct BlockViewMut<'a, S: Storage> {
    block_storage: &'a mut BlockStorage<S>,
    offset: usize,
    size: usize,
}

impl<'a, S: Storage> BlockViewMut<'a, S> {
    /// Create a new mutable storage view
    ///
    /// # Safety
    /// The caller must ensure:
    /// - offset + size <= storage.size()
    /// - The view does not outlive the storage
    /// - No other views exist for this region
    pub unsafe fn new(
        block_storage: &'a mut BlockStorage<S>,
        offset: usize,
        size: usize,
    ) -> Result<Self, BlockError> {
        if offset + size > block_storage.storage.size() {
            return Err(BlockError::Storage(StorageError::InvalidConfig(
                "View extends beyond storage bounds".into(),
            )));
        }

        Ok(Self {
            block_storage,
            offset,
            size,
        })
    }

    /// Get a raw mutable pointer to the view's data
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the view is dropped
    /// - No other references exist while the pointer is in use
    /// - Access patterns respect the storage's thread safety model
    pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe {
            let base: *mut u8 = self
                .block_storage
                .storage
                .as_ptr()
                .expect("Storage became inaccessible") as *mut u8;
            base.add(self.offset)
        }
    }

    /// Size of the view in bytes
    pub fn size(&self) -> usize {
        self.size
    }
}
