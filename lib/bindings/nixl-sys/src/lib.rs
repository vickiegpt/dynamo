//! Raw FFI bindings to the NIXL library
//!
//! This crate provides low-level bindings to the NIXL C++ library.
//! It is not meant to be used directly, but rather through the higher-level
//! `nixl` crate.

use std::ffi::{CStr, CString};
use std::fmt;
use std::ptr;
use std::ptr::NonNull;
use libc::uintptr_t;
use thiserror::Error;

// Include the generated bindings
mod bindings {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// Re-export types from the included bindings
pub use bindings::{
    nixl_capi_agent_t, nixl_capi_backend_t, nixl_capi_create_agent, nixl_capi_create_backend,
    nixl_capi_create_opt_args, nixl_capi_create_reg_dlist, nixl_capi_create_xfer_dlist,
    nixl_capi_destroy_agent, nixl_capi_destroy_backend, nixl_capi_destroy_mem_list,
    nixl_capi_destroy_opt_args, nixl_capi_destroy_params, nixl_capi_destroy_reg_dlist,
    nixl_capi_destroy_string_list, nixl_capi_destroy_xfer_dlist, nixl_capi_get_available_plugins,
    nixl_capi_get_backend_params, nixl_capi_get_plugin_params, nixl_capi_mem_list_get,
    nixl_capi_mem_list_is_empty, nixl_capi_mem_list_size, nixl_capi_mem_list_t,
    nixl_capi_mem_type_t, nixl_capi_mem_type_to_string, nixl_capi_opt_args_add_backend,
    nixl_capi_opt_args_t, nixl_capi_params_create_iterator, nixl_capi_params_destroy_iterator,
    nixl_capi_params_is_empty, nixl_capi_params_iterator_next, nixl_capi_params_t,
    nixl_capi_reg_dlist_add_desc, nixl_capi_reg_dlist_clear, nixl_capi_reg_dlist_has_overlaps,
    nixl_capi_reg_dlist_len, nixl_capi_reg_dlist_resize, nixl_capi_reg_dlist_t,
    nixl_capi_string_list_get, nixl_capi_string_list_size, nixl_capi_string_list_t,
    nixl_capi_xfer_dlist_add_desc, nixl_capi_xfer_dlist_clear, nixl_capi_xfer_dlist_has_overlaps,
    nixl_capi_xfer_dlist_len, nixl_capi_xfer_dlist_resize, nixl_capi_xfer_dlist_t,
};

// Re-export status codes
pub use bindings::{
    nixl_capi_status_t_NIXL_CAPI_ERROR_BACKEND as NIXL_CAPI_ERROR_BACKEND,
    nixl_capi_status_t_NIXL_CAPI_ERROR_INVALID_PARAM as NIXL_CAPI_ERROR_INVALID_PARAM,
    nixl_capi_status_t_NIXL_CAPI_SUCCESS as NIXL_CAPI_SUCCESS,
};

/// Errors that can occur when using NIXL
#[derive(Error, Debug)]
pub enum NixlError {
    #[error("Invalid parameter provided to NIXL")]
    InvalidParam,
    #[error("Backend error occurred")]
    BackendError,
    #[error("Failed to create CString from input: {0}")]
    StringConversionError(#[from] std::ffi::NulError),
    #[error("Index out of bounds")]
    IndexOutOfBounds,
}

/// Memory types supported by NIXL
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemType {
    Unknown,
    Dram,
    Gpu,
}

impl From<nixl_capi_mem_type_t> for MemType {
    fn from(mem_type: nixl_capi_mem_type_t) -> Self {
        match mem_type {
            1 => MemType::Dram,
            2 => MemType::Gpu,
            _ => MemType::Unknown,
        }
    }
}

impl fmt::Display for MemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: We know the memory type is valid and the string will be available
        let mut str_ptr = ptr::null();
        unsafe {
            let mem_type = match self {
                MemType::Unknown => 0,
                MemType::Dram => 1,
                MemType::Gpu => 2,
            };
            nixl_capi_mem_type_to_string(mem_type, &mut str_ptr);
            let c_str = CStr::from_ptr(str_ptr);
            write!(f, "{}", c_str.to_str().unwrap())
        }
    }
}

/// A safe wrapper around a list of strings from NIXL
pub struct StringList {
    inner: NonNull<bindings::nixl_capi_string_list_s>,
}

impl StringList {
    /// Returns the number of strings in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;
        let status = unsafe { nixl_capi_string_list_size(self.inner.as_ptr(), &mut size) };

        match status {
            0 => Ok(size),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list contains no strings
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        Ok(self.len()? == 0)
    }

    /// Returns the string at the given index
    pub fn get(&self, index: usize) -> Result<&str, NixlError> {
        let mut str_ptr = ptr::null();
        let status = unsafe { nixl_capi_string_list_get(self.inner.as_ptr(), index, &mut str_ptr) };

        match status {
            0 => {
                // SAFETY: If status is 0, str_ptr points to a valid null-terminated string
                let c_str = unsafe { CStr::from_ptr(str_ptr) };
                Ok(c_str.to_str().unwrap()) // Safe because NIXL strings are valid UTF-8
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the strings in the list
    pub fn iter(&self) -> StringListIterator<'_> {
        StringListIterator {
            list: self,
            index: 0,
            length: self.len().unwrap_or(0),
        }
    }
}

impl Drop for StringList {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_string_list(self.inner.as_ptr());
        }
    }
}

/// An iterator over the strings in a StringList
pub struct StringListIterator<'a> {
    list: &'a StringList,
    index: usize,
    length: usize,
}

impl<'a> Iterator for StringListIterator<'a> {
    type Item = Result<&'a str, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let result = self.list.get(self.index);
            self.index += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

/// A safe wrapper around NIXL memory list
pub struct MemList {
    inner: NonNull<bindings::nixl_capi_mem_list_s>,
}

impl Drop for MemList {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_mem_list(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around NIXL parameters
pub struct Params {
    inner: NonNull<bindings::nixl_capi_params_s>,
}

impl Drop for Params {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_params(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around a NIXL agent
pub struct Agent {
    // NonNull ensures we always have a valid pointer and enables optimizations
    inner: NonNull<bindings::nixl_capi_agent_s>,
}

// SAFETY: Agent can be sent between threads safely as the underlying C++ nixlAgent is thread-safe
unsafe impl Send for Agent {}
// SAFETY: Agent can be shared between threads safely as the underlying C++ nixlAgent is thread-safe
unsafe impl Sync for Agent {}

impl Agent {
    /// Creates a new NIXL agent with the given name
    ///
    /// # Arguments
    /// * `name` - The name to give the agent
    ///
    /// # Returns
    /// A new Agent instance
    ///
    /// # Errors
    /// Returns a NixlError if:
    /// * The name contains interior nul bytes
    /// * The agent creation fails
    pub fn new(name: &str) -> Result<Self, NixlError> {
        // Convert the name to a C string
        let name = CString::new(name)?;

        let mut agent = ptr::null_mut();

        // SAFETY: We ensure the CString is valid and properly null-terminated
        let status = unsafe { nixl_capi_create_agent(name.as_ptr(), &mut agent) };

        match status {
            0 => {
                // SAFETY: If status is 0, agent was successfully created and is non-null
                let inner = unsafe { NonNull::new_unchecked(agent) };
                Ok(Self { inner })
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Gets the list of available plugins
    ///
    /// # Returns
    /// A list of available plugin names
    ///
    /// # Errors
    /// Returns a NixlError if the operation fails
    pub fn get_available_plugins(&self) -> Result<StringList, NixlError> {
        let mut plugins = ptr::null_mut();

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_get_available_plugins(self.inner.as_ptr(), &mut plugins) };

        match status {
            0 => {
                // SAFETY: If status is 0, plugins was successfully created and is non-null
                let inner = unsafe { NonNull::new_unchecked(plugins) };
                Ok(StringList { inner })
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Gets the parameters for a plugin
    ///
    /// # Arguments
    /// * `plugin_name` - The name of the plugin
    ///
    /// # Returns
    /// The plugin's memory list and parameters
    ///
    /// # Errors
    /// Returns a NixlError if:
    /// * The plugin name contains interior nul bytes
    /// * The operation fails
    pub fn get_plugin_params(&self, plugin_name: &str) -> Result<(MemList, Params), NixlError> {
        let plugin_name = CString::new(plugin_name)?;
        let mut mems = ptr::null_mut();
        let mut params = ptr::null_mut();

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_get_plugin_params(
                self.inner.as_ptr(),
                plugin_name.as_ptr(),
                &mut mems,
                &mut params,
            )
        };

        match status {
            0 => {
                // SAFETY: If status is 0, both pointers were successfully created and are non-null
                let mems_inner = unsafe { NonNull::new_unchecked(mems) };
                let params_inner = unsafe { NonNull::new_unchecked(params) };
                Ok((
                    MemList { inner: mems_inner },
                    Params {
                        inner: params_inner,
                    },
                ))
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Creates a new backend for the given plugin
    ///
    /// # Arguments
    /// * `plugin_name` - The name of the plugin to create a backend for
    /// * `params` - The parameters to initialize the backend with
    ///
    /// # Returns
    /// A new Backend instance
    ///
    /// # Errors
    /// Returns a NixlError if:
    /// * The plugin name contains interior nul bytes
    /// * The backend creation fails
    pub fn create_backend(&self, plugin_name: &str, params: &Params) -> Result<Backend, NixlError> {
        let plugin_name = CString::new(plugin_name)?;
        let mut backend = ptr::null_mut();

        // SAFETY: self.inner and params.inner are guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_create_backend(
                self.inner.as_ptr(),
                plugin_name.as_ptr(),
                params.inner.as_ptr(),
                &mut backend,
            )
        };

        match status {
            0 => {
                // SAFETY: If status is 0, backend was successfully created and is non-null
                let inner = unsafe { NonNull::new_unchecked(backend) };
                Ok(Backend { inner })
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Gets the parameters and memory types for a backend after initialization
    pub fn get_backend_params(&self, backend: &Backend) -> Result<(MemList, Params), NixlError> {
        let mut mem_list = ptr::null_mut();
        let mut params = ptr::null_mut();

        let status = unsafe {
            nixl_capi_get_backend_params(
                self.inner.as_ptr(),
                backend.inner.as_ptr(),
                &mut mem_list,
                &mut params,
            )
        };

        if status != NIXL_CAPI_SUCCESS {
            return Err(NixlError::BackendError);
        }

        // SAFETY: If status is NIXL_CAPI_SUCCESS, both pointers are non-null
        unsafe {
            Ok((
                MemList {
                    inner: NonNull::new_unchecked(mem_list),
                },
                Params {
                    inner: NonNull::new_unchecked(params),
                },
            ))
        }
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_agent(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around a NIXL backend
pub struct Backend {
    inner: NonNull<bindings::nixl_capi_backend_s>,
}

impl Drop for Backend {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_backend(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around NIXL optional arguments
pub struct OptArgs {
    inner: NonNull<bindings::nixl_capi_opt_args_s>,
}

impl OptArgs {
    /// Creates a new empty optional arguments struct
    pub fn new() -> Result<Self, NixlError> {
        let mut args = ptr::null_mut();

        let status = unsafe { nixl_capi_create_opt_args(&mut args) };

        match status {
            0 => {
                // SAFETY: If status is 0, args was successfully created and is non-null
                let inner = unsafe { NonNull::new_unchecked(args) };
                Ok(Self { inner })
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Adds a backend to the optional arguments
    pub fn add_backend(&mut self, backend: &Backend) -> Result<(), NixlError> {
        let status =
            unsafe { nixl_capi_opt_args_add_backend(self.inner.as_ptr(), backend.inner.as_ptr()) };

        match status {
            0 => Ok(()),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }
}

impl Drop for OptArgs {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_opt_args(self.inner.as_ptr());
        }
    }
}

/// A key-value pair in the parameters
#[derive(Debug)]
pub struct ParamPair<'a> {
    pub key: &'a str,
    pub value: &'a str,
}

/// An iterator over parameter key-value pairs
pub struct ParamIterator<'a> {
    iter: NonNull<bindings::nixl_capi_param_iter_s>,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Iterator for ParamIterator<'a> {
    type Item = Result<ParamPair<'a>, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut key_ptr = ptr::null();
        let mut value_ptr = ptr::null();
        let mut has_next = false;

        // SAFETY: self.iter is guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_params_iterator_next(
                self.iter.as_ptr(),
                &mut key_ptr,
                &mut value_ptr,
                &mut has_next,
            )
        };

        match status {
            0 if !has_next => None,
            0 => {
                // SAFETY: If status is 0, both pointers are valid null-terminated strings
                let result = unsafe {
                    let key = CStr::from_ptr(key_ptr).to_str().unwrap();
                    let value = CStr::from_ptr(value_ptr).to_str().unwrap();
                    Ok(ParamPair { key, value })
                };
                Some(result)
            }
            -1 => Some(Err(NixlError::InvalidParam)),
            _ => Some(Err(NixlError::BackendError)),
        }
    }
}

impl<'a> Drop for ParamIterator<'a> {
    fn drop(&mut self) {
        // SAFETY: self.iter is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_params_destroy_iterator(self.iter.as_ptr());
        }
    }
}

impl Params {
    /// Returns true if the parameters are empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        let mut is_empty = false;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_params_is_empty(self.inner.as_ptr(), &mut is_empty) };

        match status {
            0 => Ok(is_empty),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the parameter key-value pairs
    pub fn iter(&self) -> Result<ParamIterator<'_>, NixlError> {
        let mut iter = ptr::null_mut();

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_params_create_iterator(self.inner.as_ptr(), &mut iter) };

        match status {
            0 => {
                // SAFETY: If status is 0, iter was successfully created and is non-null
                let iter = unsafe { NonNull::new_unchecked(iter) };
                Ok(ParamIterator {
                    iter,
                    _phantom: std::marker::PhantomData,
                })
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }
}

impl MemList {
    /// Returns true if the memory list is empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        let mut is_empty = false;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_mem_list_is_empty(self.inner.as_ptr(), &mut is_empty) };

        match status {
            0 => Ok(is_empty),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of memory types in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_mem_list_size(self.inner.as_ptr(), &mut size) };

        match status {
            0 => Ok(size),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the memory type at the given index
    pub fn get(&self, index: usize) -> Result<MemType, NixlError> {
        let mut mem_type = 0;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_mem_list_get(self.inner.as_ptr(), index, &mut mem_type) };

        match status {
            0 => Ok(MemType::from(mem_type)),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the memory types
    pub fn iter(&self) -> MemListIterator<'_> {
        MemListIterator {
            list: self,
            index: 0,
            length: self.len().unwrap_or(0),
        }
    }
}

/// An iterator over memory types in a MemList
pub struct MemListIterator<'a> {
    list: &'a MemList,
    index: usize,
    length: usize,
}

impl<'a> Iterator for MemListIterator<'a> {
    type Item = Result<MemType, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let result = self.list.get(self.index);
            self.index += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

/// A safe wrapper around a NIXL transfer descriptor list
pub struct XferDescList {
    inner: NonNull<bindings::nixl_capi_xfer_dlist_s>,
}

impl XferDescList {
    /// Creates a new transfer descriptor list for the given memory type
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        let mut dlist = ptr::null_mut();
        let status =
            unsafe { nixl_capi_create_xfer_dlist(mem_type as nixl_capi_mem_type_t, &mut dlist) };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, dlist is non-null
                let inner = unsafe { NonNull::new_unchecked(dlist) };
                Ok(Self { inner })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Adds a descriptor to the list
    pub fn add_desc(&mut self, addr: usize, len: usize, dev_id: u32) -> Result<(), NixlError> {
        let status = unsafe {
            nixl_capi_xfer_dlist_add_desc(self.inner.as_ptr(), addr as uintptr_t, len, dev_id)
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of descriptors in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut len = 0;
        let status = unsafe { nixl_capi_xfer_dlist_len(self.inner.as_ptr(), &mut len) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(len),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if any descriptors in the list overlap
    pub fn has_overlaps(&self) -> Result<bool, NixlError> {
        let mut has_overlaps = false;
        let status =
            unsafe { nixl_capi_xfer_dlist_has_overlaps(self.inner.as_ptr(), &mut has_overlaps) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(has_overlaps),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Clears all descriptors from the list
    pub fn clear(&mut self) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_xfer_dlist_clear(self.inner.as_ptr()) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Resizes the list to the given size
    pub fn resize(&mut self, new_size: usize) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_xfer_dlist_resize(self.inner.as_ptr(), new_size) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }
}

impl Drop for XferDescList {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_xfer_dlist(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around a NIXL registration descriptor list
pub struct RegDescList {
    inner: NonNull<bindings::nixl_capi_reg_dlist_s>,
}

impl RegDescList {
    /// Creates a new registration descriptor list for the given memory type
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        let mut dlist = ptr::null_mut();
        let status =
            unsafe { nixl_capi_create_reg_dlist(mem_type as nixl_capi_mem_type_t, &mut dlist) };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, dlist is non-null
                let inner = unsafe { NonNull::new_unchecked(dlist) };
                Ok(Self { inner })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Adds a descriptor to the list
    pub fn add_desc(&mut self, addr: usize, len: usize, dev_id: u32) -> Result<(), NixlError> {
        let status = unsafe {
            nixl_capi_reg_dlist_add_desc(self.inner.as_ptr(), addr as uintptr_t, len, dev_id)
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of descriptors in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut len = 0;
        let status = unsafe { nixl_capi_reg_dlist_len(self.inner.as_ptr(), &mut len) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(len),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if any descriptors in the list overlap
    pub fn has_overlaps(&self) -> Result<bool, NixlError> {
        let mut has_overlaps = false;
        let status =
            unsafe { nixl_capi_reg_dlist_has_overlaps(self.inner.as_ptr(), &mut has_overlaps) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(has_overlaps),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Clears all descriptors from the list
    pub fn clear(&mut self) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_reg_dlist_clear(self.inner.as_ptr()) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Resizes the list to the given size
    pub fn resize(&mut self, new_size: usize) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_reg_dlist_resize(self.inner.as_ptr(), new_size) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }
}

impl Drop for RegDescList {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_reg_dlist(self.inner.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let agent = Agent::new("test_agent").expect("Failed to create agent");
        drop(agent);
    }

    #[test]
    fn test_agent_invalid_name() {
        let result = Agent::new("test\0agent");
        assert!(matches!(result, Err(NixlError::StringConversionError(_))));
    }

    #[test]
    fn test_get_available_plugins() {
        let agent = Agent::new("test_agent").expect("Failed to create agent");
        let plugins = agent
            .get_available_plugins()
            .expect("Failed to get plugins");

        // Print available plugins
        for plugin in plugins.iter() {
            println!("Found plugin: {}", plugin.unwrap());
        }
    }

    #[test]
    fn test_get_plugin_params() {
        let agent = Agent::new("test_agent").expect("Failed to create agent");
        let (_mems, _params) = agent
            .get_plugin_params("UCX")
            .expect("Failed to get plugin params");
        // MemList and Params will be automatically dropped here
    }

    #[test]
    fn test_backend_creation() {
        let agent = Agent::new("test_agent").expect("Failed to create agent");
        let (mems, params) = agent
            .get_plugin_params("UCX")
            .expect("Failed to get plugin params");
        let backend = agent
            .create_backend("UCX", &params)
            .expect("Failed to create backend");

        let mut opt_args = OptArgs::new().expect("Failed to create opt args");
        opt_args
            .add_backend(&backend)
            .expect("Failed to add backend");
    }

    #[test]
    fn test_params_iteration() {
        let agent = Agent::new("test_agent").expect("Failed to create agent");
        let (mems, params) = agent
            .get_plugin_params("UCX")
            .expect("Failed to get plugin params");

        println!("Parameters:");
        if !params.is_empty().unwrap() {
            for param in params.iter().unwrap() {
                let param = param.unwrap();
                println!("  {} = {}", param.key, param.value);
            }
        } else {
            println!("  (empty)");
        }

        println!("Memory types:");
        if !mems.is_empty().unwrap() {
            for mem_type in mems.iter() {
                println!("  {}", mem_type.unwrap());
            }
        } else {
            println!("  (empty)");
        }
    }

    #[test]
    fn test_get_backend_params() {
        let agent = Agent::new("test_agent").unwrap();
        let plugins = agent.get_available_plugins().unwrap();
        assert!(plugins.is_empty().unwrap_or(false) == false);

        let plugin_name = plugins.get(0).unwrap();
        let (mems, params) = agent.get_plugin_params(&plugin_name).unwrap();
        let backend = agent.create_backend(&plugin_name, &params).unwrap();

        // Get backend params after initialization
        let (backend_mems, backend_params) = agent.get_backend_params(&backend).unwrap();

        // Verify we can access the parameters
        let param_iter = backend_params.iter().unwrap();
        for param in param_iter {
            let param = param.unwrap();
            println!("Backend param: {} = {}", param.key, param.value);
        }

        // Verify we can access the memory types
        for mem_type in backend_mems.iter() {
            println!("Backend memory type: {:?}", mem_type.unwrap());
        }
    }

    #[test]
    fn test_xfer_dlist() {
        let mut dlist = XferDescList::new(MemType::Dram).unwrap();

        // Add some descriptors
        dlist.add_desc(0x1000, 0x100, 0).unwrap();
        dlist.add_desc(0x2000, 0x200, 1).unwrap();

        // Check length
        assert_eq!(dlist.len().unwrap(), 2);

        // Check overlaps
        assert!(!dlist.has_overlaps().unwrap());

        // Add overlapping descriptor
        dlist.add_desc(0x1050, 0x100, 0).unwrap();
        assert!(dlist.has_overlaps().unwrap());

        // Clear list
        dlist.clear().unwrap();
        assert_eq!(dlist.len().unwrap(), 0);

        // Resize list
        dlist.resize(5).unwrap();
    }

    #[test]
    fn test_reg_dlist() {
        let mut dlist = RegDescList::new(MemType::Dram).unwrap();

        // Add some descriptors
        dlist.add_desc(0x1000, 0x100, 0).unwrap();
        dlist.add_desc(0x2000, 0x200, 1).unwrap();

        // Check length
        assert_eq!(dlist.len().unwrap(), 2);

        // Check overlaps
        assert!(!dlist.has_overlaps().unwrap());

        // Add overlapping descriptor
        dlist.add_desc(0x1050, 0x100, 0).unwrap();
        assert!(dlist.has_overlaps().unwrap());

        // Clear list
        dlist.clear().unwrap();
        assert_eq!(dlist.len().unwrap(), 0);

        // Resize list
        dlist.resize(5).unwrap();
    }
}
