// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{check_cuda, sys::CUevent, CudaEvent};

use ::cudarc::driver::sys::{cuEventCreate, cuEventDestroy_v2, CUevent_flags};

use derive_builder::Builder;

#[derive(Clone, Builder)]
#[builder(pattern = "owned")]
pub struct CudaEventOptions {
    #[builder(default = "false")]
    pub enable_blocking: bool,

    #[builder(default = "false")]
    pub enable_timing: bool,

    #[builder(default = "false")]
    pub enable_ipc: bool,
}

impl CudaEventOptions {
    pub fn new() -> Self {
        Self::builder().build().unwrap()
    }

    pub fn builder() -> CudaEventOptionsBuilder {
        CudaEventOptionsBuilder::default()
    }
}

impl Default for CudaEventOptions {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

pub struct OwnedCudaEvent {
    event: CUevent,
}

impl OwnedCudaEvent {
    pub fn new(options: CudaEventOptions) -> anyhow::Result<Self> {
        // todo/assert: validate we are in a valid CUDA context

        let mut flags = CUevent_flags::CU_EVENT_DEFAULT as u32;

        if options.enable_blocking {
            flags |= CUevent_flags::CU_EVENT_BLOCKING_SYNC as u32;
        }

        if !options.enable_timing {
            flags |= CUevent_flags::CU_EVENT_DISABLE_TIMING as u32;
        }

        if options.enable_ipc {
            flags |= CUevent_flags::CU_EVENT_INTERPROCESS as u32;
        }

        let event = unsafe {
            let mut event: CUevent = std::ptr::null_mut();
            check_cuda(cuEventCreate(&mut event, flags))?;
            event
        };

        Ok(Self { event })
    }
}

impl CudaEvent for OwnedCudaEvent {
    unsafe fn cu_event(&self) -> CUevent {
        self.event
    }
}

impl AsRef<dyn CudaEvent> for OwnedCudaEvent {
    fn as_ref(&self) -> &(dyn CudaEvent + 'static) {
        self
    }
}

impl From<OwnedCudaEvent> for Box<dyn CudaEvent> {
    fn from(event: OwnedCudaEvent) -> Self {
        Box::new(event)
    }
}

impl Drop for OwnedCudaEvent {
    fn drop(&mut self) {
        let result = unsafe { cuEventDestroy_v2(self.event) };
        check_cuda(result).expect("Failed to destroy CUDA event");
    }
}
