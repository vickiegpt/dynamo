// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Silence warnings about unused code and deprecated features
// These are kept for API compatibility and potential future use
#![allow(dead_code)]
#![allow(deprecated)]

use std::{ptr::NonNull, ffi::c_void};
use pyo3::{IntoPy, Python, PyObject};

/// Raw FFI bindings for DLPack.
pub mod ffi {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum DeviceType {
        /// CPU device
        Cpu         = 1,
        /// CUDA GPU device
        Cuda        = 2,
        /// Pinned CUDA CPU memory by cudaMallocHost
        CudaHost    = 3,
        /// OpenCL devices.
        OpenCl      = 4,
        /// Vulkan buffer for next generation graphics.
        Vulkan      = 7,
        /// Metal for Apple GPU.
        Metal       = 8,
        /// Verilog simulator buffer
        Vpi         = 9,
        /// ROCm GPUs for AMD GPUs
        Rocm        = 10,
        /// Pinned ROCm CPU memory allocated by hipMallocHost
        RocmHost    = 11,
        /// Reserved extension device type,
        /// used for quickly test extension device
        /// The semantics can differ depending on the implementation.
        ExtDev      = 12,
        /// CUDA managed/unified memory allocated by cudaMallocManaged
        CudaManaged = 13,
        /// Unified shared memory allocated on a oneAPI non-partititioned
        /// device. Call to oneAPI runtime is required to determine the device
        /// type, the USM allocation type and the sycl context it is bound to.
        OneApi      = 14,
        /// GPU support for next generation WebGPU standard.
        WebGpu      = 15,
        /// Qualcomm Hexagon DSP
        Hexagon     = 16,
    }

    /// A Device for Tensor and operator.
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Device {
        /// The device type used in the device.
        pub device_type: DeviceType,
        /// The device index.
        /// For vanilla CPU memory, pinned memory, or managed memory, this is set to
        /// 0.
        pub device_id: i32,
    }

    // Device constants
    impl Device {
        pub const CPU: Self = Self {
            device_type: DeviceType::Cpu,
            device_id: 0,
        };
    }

    #[repr(u8)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum DataTypeCode {
        /// signed integer
        Int          = 0,
        /// unsigned integer
        UInt         = 1,
        /// IEEE floating point
        Float        = 2,
        /// Opaque handle type, reserved for testing purposes.
        /// Frameworks need to agree on the handle data type for the exchange to be
        /// well-defined.
        OpaqueHandle = 3,
        /// bfloat16
        Bfloat       = 4,
        /// complex number
        /// (C/C++/Python layout: compact struct per complex number)
        Complex      = 5,
        /// boolean
        Bool         = 6,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct DataType {
        /// Type code of base types.
        pub code: DataTypeCode,
        /// Number of bits, common choices are 8, 16, 32.
        pub bits: u8,
        /// Number of lanes in the type, used for vector types.
        pub lanes: u16,
    }

    impl DataType {
        // Bfloat
        pub const BF16: Self = Self {
            code: DataTypeCode::Bfloat,
            bits: 16,
            lanes: 1,
        };
        // Bool
        pub const BOOL: Self = Self {
            code: DataTypeCode::Bool,
            bits: 8,
            lanes: 1,
        };
        // Float
        pub const F16: Self = Self {
            code: DataTypeCode::Float,
            bits: 16,
            lanes: 1,
        };
        pub const F32: Self = Self {
            code: DataTypeCode::Float,
            bits: 32,
            lanes: 1,
        };
        pub const F64: Self = Self {
            code: DataTypeCode::Float,
            bits: 64,
            lanes: 1,
        };
        pub const I128: Self = Self {
            code: DataTypeCode::Int,
            bits: 128,
            lanes: 1,
        };
        pub const I16: Self = Self {
            code: DataTypeCode::Int,
            bits: 16,
            lanes: 1,
        };
        pub const I32: Self = Self {
            code: DataTypeCode::Int,
            bits: 32,
            lanes: 1,
        };
        pub const I64: Self = Self {
            code: DataTypeCode::Int,
            bits: 64,
            lanes: 1,
        };
        // Int
        pub const I8: Self = Self {
            code: DataTypeCode::Int,
            bits: 8,
            lanes: 1,
        };
        pub const U128: Self = Self {
            code: DataTypeCode::UInt,
            bits: 128,
            lanes: 1,
        };
        pub const U16: Self = Self {
            code: DataTypeCode::UInt,
            bits: 16,
            lanes: 1,
        };
        pub const U32: Self = Self {
            code: DataTypeCode::UInt,
            bits: 32,
            lanes: 1,
        };
        pub const U64: Self = Self {
            code: DataTypeCode::UInt,
            bits: 64,
            lanes: 1,
        };
        // Uint
        pub const U8: Self = Self {
            code: DataTypeCode::UInt,
            bits: 8,
            lanes: 1,
        };

        /// Calculate `DataType` size as (bits * lanes + 7) // 8
        pub fn size(&self) -> usize {
            ((self.bits as u32 * self.lanes as u32 + 7) / 8) as usize
        }
    }

    /// DLTensor structs
    #[repr(C)]
    pub struct DLTensor {
        pub data: *mut c_void,
        pub device: Device,
        pub ndim: i32,
        pub dtype: DataType,
        pub shape: *mut i64,
        pub strides: *mut i64,
        pub byte_offset: u64,
    }

    /// DLManagedTensor struct
    #[repr(C)]
    pub struct DLManagedTensor {
        pub dl_tensor: DLTensor,
        pub manager_ctx: *mut c_void,
        pub deleter: Option<unsafe extern "C" fn(*mut Self)>,
    }
}

/// Shape and strides for a tensor
pub enum ShapeAndStrides {
    Contiguous(Box<[i64]>),  // Shape only
    WithStrides(Box<[i64]>), // [Shape | Strides]
    Borrowed {
        shape: NonNull<i64>,
        strides: Option<NonNull<i64>>,
        len: usize,
    },
}

impl ShapeAndStrides {
    pub fn new_contiguous<'a, I>(shape: I) -> Self
    where
        I: IntoIterator<Item = &'a i64>,
    {
        let buf: Vec<i64> = shape.into_iter().copied().collect();
        Self::Contiguous(buf.into_boxed_slice())
    }

    pub fn shape(&self) -> &[i64] {
        match self {
            Self::Contiguous(ref v) => v.as_ref(),
            Self::WithStrides(ref v) => &v[0..self.len()],
            Self::Borrowed { shape, len, .. } => unsafe {
                std::slice::from_raw_parts(shape.as_ptr(), *len)
            },
        }
    }

    pub fn strides(&self) -> Option<&[i64]> {
        match self {
            Self::Contiguous(_) => None,
            Self::WithStrides(ref v) => Some(&v[self.len()..]),
            Self::Borrowed { strides, len, .. } => {
                strides.map(|s| unsafe { std::slice::from_raw_parts(s.as_ptr(), *len) })
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Contiguous(ref v) => v.len(),
            Self::WithStrides(ref v) => v.len() / 2,
            Self::Borrowed { len, .. } => *len,
        }
    }

    pub fn ndim(&self) -> i32 {
        self.len() as i32
    }

    pub(crate) fn shape_ptr(&self) -> *mut i64 {
        match self {
            Self::Contiguous(ref v) => v.as_ptr() as *mut i64,
            Self::WithStrides(ref v) => v.as_ptr() as *mut i64,
            Self::Borrowed { shape, .. } => shape.as_ptr(),
        }
    }

    pub(crate) fn strides_ptr(&self) -> *mut i64 {
        match self {
            Self::Contiguous(_) => std::ptr::null_mut(),
            Self::WithStrides(ref v) => &v[self.len()] as *const i64 as *mut i64,
            Self::Borrowed { strides, .. } => match strides {
                Some(strides) => strides.as_ptr(),
                None => std::ptr::null_mut(),
            },
        }
    }
}

/// DLPack is a data structure that can be used to describe tensor data.
pub type DLPack = NonNull<ffi::DLManagedTensor>;

/// User-implemented trait for tensor-like objects
pub trait ToTensor {
    fn data_ptr(&self) -> *mut c_void;
    fn shape_and_strides(&self) -> ShapeAndStrides;
    fn device(&self) -> ffi::Device;
    fn dtype(&self) -> ffi::DataType;
    fn byte_offset(&self) -> u64;
}

/// Convert to DLPack
pub trait IntoDLPack {
    fn into_dlpack(self) -> DLPack;
}

/// ManagerCtx for tensor data
pub struct ManagerCtx<T> {
    inner: T,
    shape_and_strides: ShapeAndStrides,
    tensor: Option<ffi::DLManagedTensor>,
}

unsafe extern "C" fn deleter_fn<T>(dl_managed_tensor: *mut ffi::DLManagedTensor) {
    // Reconstruct pointer and destroy it.
    let ctx = (*dl_managed_tensor).manager_ctx as *mut T;
    // Use from_raw to clean it.
    unsafe { let _ = Box::from_raw(ctx); };
}

impl<T> ManagerCtx<T>
where
    T: ToTensor,
{
    pub fn new(inner: T) -> Self {
        let shape_and_strides = inner.shape_and_strides();
        Self {
            inner,
            shape_and_strides,
            tensor: None,
        }
    }

    pub(crate) fn into_dl_managed_tensor(self) -> NonNull<ffi::DLManagedTensor> {
        // Move self to heap and get it's pointer.
        // We leak the data here and let deleter handle its memory.
        let ctx = Box::leak(Box::new(self));
        let tensor: ffi::DLManagedTensor = ffi::DLManagedTensor {
            dl_tensor: ctx.make_dl_tensor(),
            manager_ctx: ctx as *mut Self as *mut c_void,
            deleter: Some(deleter_fn::<Self>),
        };
        // Hold the data so it can be dropped when ctx dropped.
        ctx.tensor = Some(tensor);
        // Take the address of DLManagedTensor
        NonNull::from(ctx.tensor.as_ref().unwrap())
    }

    fn make_dl_tensor(&self) -> ffi::DLTensor {
        ffi::DLTensor {
            data: self.inner.data_ptr(),
            device: self.inner.device(),
            ndim: self.shape_and_strides.ndim(),
            dtype: self.inner.dtype(),
            shape: self.shape_and_strides.shape_ptr(),
            strides: self.shape_and_strides.strides_ptr(),
            byte_offset: self.inner.byte_offset(),
        }
    }

    pub fn into_dlpack(self) -> DLPack {
        self.into_dl_managed_tensor()
    }
}

// Constants for Python capsule support
const DLPACK_CAPSULE_NAME: &[u8] = b"dltensor\0";
const DLPACK_CAPSULE_USED_NAME: &[u8] = b"used_dltensor\0";

// DLPack to Python capsule conversion
pub fn dlpack_to_py_capsule(dlpack: DLPack) -> *mut pyo3::ffi::PyObject {
    unsafe {
        pyo3::ffi::PyCapsule_New(
            dlpack.as_ptr().cast(),
            DLPACK_CAPSULE_NAME.as_ptr().cast(),
            Some(dlpack_capsule_deleter),
        )
    }
}

// Python capsule deleter callback
unsafe extern "C" fn dlpack_capsule_deleter(capsule: *mut pyo3::ffi::PyObject) {
    if pyo3::ffi::PyCapsule_IsValid(capsule, DLPACK_CAPSULE_USED_NAME.as_ptr() as *const _) == 1 {
        return;
    }

    let mut exc_type = std::ptr::null_mut();
    let mut exc_value = std::ptr::null_mut();
    let mut exc_trace = std::ptr::null_mut();
    pyo3::ffi::PyErr_Fetch(&mut exc_type, &mut exc_value, &mut exc_trace);

    let managed = pyo3::ffi::PyCapsule_GetPointer(capsule, DLPACK_CAPSULE_NAME.as_ptr() as *const _)
        as *mut ffi::DLManagedTensor;

    if managed.is_null() {
        pyo3::ffi::PyErr_WriteUnraisable(capsule);
        pyo3::ffi::PyErr_Restore(exc_type, exc_value, exc_trace);
        return;
    }

    if let Some(del_fn) = (*managed).deleter {
        del_fn(managed);
        assert!(pyo3::ffi::PyErr_Occurred().is_null());
    }

    pyo3::ffi::PyErr_Restore(exc_type, exc_value, exc_trace);
}

// Implementation of IntoPy for ManagerCtx
impl<T> IntoPy<PyObject> for ManagerCtx<T>
where
    T: ToTensor,
{
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dlpack = self.into_dlpack();
        let capsule = dlpack_to_py_capsule(dlpack);
        unsafe { PyObject::from_owned_ptr(py, capsule) }
    }
}
