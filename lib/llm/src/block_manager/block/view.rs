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

use super::{BlockData, BlockError, BlockLayout};

/// Storage view that provides safe access to a region of storage
#[derive(Debug)]
pub struct BlockView<'a, L: BlockLayout> {
    _block_data: &'a BlockData<L>,
    addr: usize,
    size: usize,
}

impl<'a, L> BlockView<'a, L>
where
    L: BlockLayout,
{
    /// Create a new storage view
    ///
    /// # Safety
    /// The caller must ensure:
    /// - addr + size <= storage.size()
    /// - The view does not outlive the storage
    pub(crate) unsafe fn new(
        _block_data: &'a BlockData<L>,
        addr: usize,
        size: usize,
    ) -> Result<Self, BlockError> {
        Ok(Self {
            _block_data,
            addr,
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
        self.addr as *const u8
    }

    /// Size of the view in bytes
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Mutable storage view that provides exclusive access to a region of storage
#[derive(Debug)]
pub struct BlockViewMut<'a, L: BlockLayout> {
    _block_data: &'a mut BlockData<L>,
    addr: usize,
    size: usize,
}

impl<'a, L: BlockLayout> BlockViewMut<'a, L> {
    /// Create a new mutable storage view
    ///
    /// # Safety
    /// The caller must ensure:
    /// - addr + size <= storage.size()
    /// - The view does not outlive the storage
    /// - No other views exist for this region
    pub(crate) unsafe fn new(
        _block_data: &'a mut BlockData<L>,
        addr: usize,
        size: usize,
    ) -> Result<Self, BlockError> {
        Ok(Self {
            _block_data,
            addr,
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
        self.addr as *mut u8
    }

    /// Size of the view in bytes
    pub fn size(&self) -> usize {
        self.size
    }
}

mod nixl {
    use super::*;

    pub use nixl_sys::{
        Agent as NixlAgent, MemType, MemoryRegion, NixlDescriptor, OptArgs,
        RegistrationHandle as NixlRegistrationHandle,
    };

    impl<'a, L: BlockLayout> MemoryRegion for BlockView<'a, L> {
        unsafe fn as_ptr(&self) -> *const u8 {
            self.addr as *const u8
        }

        fn size(&self) -> usize {
            self.size()
        }
    }

    impl<'a, L> NixlDescriptor for BlockView<'a, L>
    where
        L: BlockLayout,
        L::StorageType: NixlDescriptor,
    {
        fn mem_type(&self) -> MemType {
            self._block_data
                .layout
                .storage()
                .first()
                .expect("no storage")
                .mem_type()
        }

        fn device_id(&self) -> u64 {
            self._block_data
                .layout
                .storage()
                .first()
                .expect("no storage")
                .device_id()
        }
    }

    impl<'a, L: BlockLayout> MemoryRegion for BlockViewMut<'a, L> {
        unsafe fn as_ptr(&self) -> *const u8 {
            self.addr as *const u8
        }

        fn size(&self) -> usize {
            self.size()
        }
    }

    impl<'a, L> NixlDescriptor for BlockViewMut<'a, L>
    where
        L: BlockLayout,
        L::StorageType: NixlDescriptor,
    {
        fn mem_type(&self) -> MemType {
            self._block_data
                .layout
                .storage()
                .first()
                .expect("no storage")
                .mem_type()
        }

        fn device_id(&self) -> u64 {
            self._block_data
                .layout
                .storage()
                .first()
                .expect("no storage")
                .device_id()
        }
    }
}
