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

pub use nixl_sys::{
    Agent as NixlAgent, MemType, MemoryRegion, NixlDescriptor, OptArgs,
    RegistrationHandle as NixlRegistrationHandle,
};

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

use super::{
    DeviceStorage, PinnedStorage, RegistationHandle, RegisterableStorage, Remote, Storage,
    StorageType, SystemStorage,
};

use anyhow::Result;

pub trait NixlAccessible {}

impl RegistationHandle for NixlRegistrationHandle {
    fn release(&mut self) {
        if let Err(e) = self.deregister() {
            tracing::error!("Failed to deregister Nixl storage: {}", e);
        }
    }
}

pub trait NixlEnabledStorage: RegisterableStorage + NixlDescriptor + Sized {
    /// Register the storage with the NIXL agent.
    fn nixl_register(&mut self, agent: &NixlAgent, opt_args: Option<&OptArgs>) -> Result<()> {
        let handle = Box::new(agent.register_memory(self, opt_args)?);
        // Assuming PinnedStorage has `handles: RegistrationHandles`
        Ok(self.register("nixl", handle)?)
    }

    /// Check if the storage is registered with the NIXL agent.
    fn is_nixl_registered(&self) -> bool {
        self.is_registered("nixl")
    }

    fn nixl_agent_name(&self) -> Option<String> {
        // Get the registration handle associated with "nixl".
        self.registration_handle("nixl")
            // If a handle exists, attempt to downcast it.
            .and_then(|handle_box| {
                // Cast the trait object &dyn RegistationHandle to &dyn Any
                // then attempt to downcast to the concrete NixlRegistrationHandle type.
                // Note: This requires RegistationHandle: Any + 'static
                (handle_box as &dyn std::any::Any)
                    .downcast_ref::<NixlRegistrationHandle>()
                    // If downcast succeeds, get the agent name.
                    .map(|nixl_handle| nixl_handle.agent_name())
            })?
    }

    /// If the underlying storage is NIXL-compatible, return descriptions of the NIXL memory regions.
    /// This is used for serialization/deserialization of NIXL-specific layouts.
    fn get_nixl_descriptors(&self) -> Option<NixlStorage> {
        if self.is_nixl_registered() {
            Some(NixlStorage {
                addr: self.addr(),
                size: MemoryRegion::size(self),
                mem_type: self.mem_type(),
                device_id: self.device_id(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Getters)]
pub struct NixlStorage {
    addr: u64,
    size: usize,
    mem_type: MemType,
    device_id: u64,
}

impl Remote for NixlStorage {}
impl NixlAccessible for NixlStorage {}

impl Storage for NixlStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Nixl
    }

    fn addr(&self) -> u64 {
        self.addr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn is_host_accessible(&self) -> bool {
        false
    }

    unsafe fn as_ptr(&self) -> Option<*const u8> {
        Some(self.addr as *const u8)
    }

    unsafe fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        Some(self.addr as *mut u8)
    }
}

impl MemoryRegion for NixlStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl NixlDescriptor for NixlStorage {
    fn mem_type(&self) -> MemType {
        self.mem_type
    }

    fn device_id(&self) -> u64 {
        self.device_id
    }
}

impl NixlEnabledStorage for SystemStorage {}

impl MemoryRegion for SystemStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn size(&self) -> usize {
        self.len
    }
}

impl NixlDescriptor for SystemStorage {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }

    fn device_id(&self) -> u64 {
        0
    }
}

impl NixlAccessible for PinnedStorage {}
impl NixlEnabledStorage for PinnedStorage {}

impl MemoryRegion for PinnedStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl NixlDescriptor for PinnedStorage {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }

    fn device_id(&self) -> u64 {
        0
    }
}

impl NixlAccessible for DeviceStorage {}
impl NixlEnabledStorage for DeviceStorage {}

impl MemoryRegion for DeviceStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl NixlDescriptor for DeviceStorage {
    fn mem_type(&self) -> MemType {
        MemType::Vram
    }

    fn device_id(&self) -> u64 {
        self.ctx.cu_device() as u64
    }
}
