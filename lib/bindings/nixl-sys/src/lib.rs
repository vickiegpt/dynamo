//! Raw FFI bindings to the NIXL library
//!
//! This crate provides low-level bindings to the NIXL C++ library.
//! It is not meant to be used directly, but rather through the higher-level
//! `nixl` crate.

use libc::uintptr_t;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::ptr::NonNull;
use std::sync::{Arc, RwLock};
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
    nixl_capi_deregister_mem, nixl_capi_destroy_agent, nixl_capi_destroy_backend,
    nixl_capi_destroy_mem_list, nixl_capi_destroy_opt_args, nixl_capi_destroy_params,
    nixl_capi_destroy_reg_dlist, nixl_capi_destroy_string_list, nixl_capi_destroy_xfer_dlist,
    nixl_capi_get_available_plugins, nixl_capi_get_backend_params, nixl_capi_get_local_md,
    nixl_capi_get_plugin_params, nixl_capi_invalidate_remote_md, nixl_capi_load_remote_md,
    nixl_capi_mem_list_get, nixl_capi_mem_list_is_empty, nixl_capi_mem_list_size,
    nixl_capi_mem_list_t, nixl_capi_mem_type_t, nixl_capi_mem_type_to_string,
    nixl_capi_opt_args_add_backend, nixl_capi_opt_args_t, nixl_capi_params_create_iterator,
    nixl_capi_params_destroy_iterator, nixl_capi_params_is_empty, nixl_capi_params_iterator_next,
    nixl_capi_params_t, nixl_capi_reg_dlist_add_desc, nixl_capi_reg_dlist_clear,
    nixl_capi_reg_dlist_has_overlaps, nixl_capi_reg_dlist_len, nixl_capi_reg_dlist_resize,
    nixl_capi_reg_dlist_t, nixl_capi_register_mem, nixl_capi_string_list_get,
    nixl_capi_string_list_size, nixl_capi_string_list_t, nixl_capi_xfer_dlist_add_desc,
    nixl_capi_xfer_dlist_clear, nixl_capi_xfer_dlist_has_overlaps, nixl_capi_xfer_dlist_len,
    nixl_capi_xfer_dlist_resize, nixl_capi_xfer_dlist_t,
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
    #[error("Invalid data pointer")]
    InvalidDataPointer,
    #[error("Failed to create XferRequest")]
    FailedToCreateXferRequest,
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

/// Inner state for an agent that manages the raw pointer
#[derive(Debug)]
struct AgentInner {
    handle: NonNull<bindings::nixl_capi_agent_s>,
    remotes: HashSet<String>,
}

unsafe impl Send for AgentInner {}
unsafe impl Sync for AgentInner {}

impl AgentInner {
    fn new(handle: NonNull<bindings::nixl_capi_agent_s>) -> Self {
        Self {
            handle,
            remotes: HashSet::new(),
        }
    }

    fn invalidate_remote_md(&mut self, remote_agent: &str) -> Result<(), NixlError> {
        unsafe {
            if self.remotes.remove(remote_agent) {
                nixl_capi_invalidate_remote_md(self.handle.as_ptr(), remote_agent.as_ptr().cast());
            } else {
                return Err(NixlError::InvalidParam);
            }
        }
        Ok(())
    }

    fn invalidate_all_remotes(&mut self) -> Result<(), NixlError> {
        unsafe {
            for remote in self.remotes.drain() {
                nixl_capi_invalidate_remote_md(self.handle.as_ptr(), remote.as_ptr().cast());
            }
        }
        Ok(())
    }
}

impl Drop for AgentInner {
    fn drop(&mut self) {
        unsafe {
            // invalidate all remotes
            for remote in self.remotes.iter() {
                nixl_capi_invalidate_remote_md(self.handle.as_ptr(), remote.as_ptr().cast());
            }

            nixl_capi_destroy_agent(self.handle.as_ptr());
        }
    }
}

/// A NIXL agent that can create backends and manage memory
#[derive(Debug, Clone)]
pub struct Agent {
    inner: Arc<RwLock<AgentInner>>,
}

impl Agent {
    /// Creates a new agent with the given name
    pub fn new(name: &str) -> Result<Self, NixlError> {
        let c_name = CString::new(name)?;
        let mut agent = ptr::null_mut();
        let status = unsafe { nixl_capi_create_agent(c_name.as_ptr(), &mut agent) };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, agent is non-null
                let handle = unsafe { NonNull::new_unchecked(agent) };
                Ok(Self {
                    inner: Arc::new(RwLock::new(AgentInner::new(handle))),
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
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
        let status = unsafe {
            nixl_capi_get_available_plugins(
                self.inner.write().unwrap().handle.as_ptr(),
                &mut plugins,
            )
        };

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
                self.inner.read().unwrap().handle.as_ptr(),
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

    /// Creates a new backend for the given plugin using the provided parameters
    pub fn create_backend(&self, plugin: &str, params: &Params) -> Result<Backend, NixlError> {
        let c_plugin = CString::new(plugin).map_err(|_| NixlError::InvalidParam)?;
        let mut backend = ptr::null_mut();
        let status = unsafe {
            nixl_capi_create_backend(
                self.inner.write().unwrap().handle.as_ptr(),
                c_plugin.as_ptr(),
                params.inner.as_ptr(),
                &mut backend,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, backend is non-null
                let inner = unsafe { NonNull::new_unchecked(backend) };
                Ok(Backend {
                    inner,
                    _agent: self.inner.clone(), // Keep agent alive while backend exists
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Gets the parameters and memory types for a backend after initialization
    pub fn get_backend_params(&self, backend: &Backend) -> Result<(MemList, Params), NixlError> {
        let mut mem_list = ptr::null_mut();
        let mut params = ptr::null_mut();

        let status = unsafe {
            nixl_capi_get_backend_params(
                self.inner.read().unwrap().handle.as_ptr(),
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

    pub fn register_memory(
        &self,
        descriptor: &dyn NixlDescriptor,
    ) -> Result<RegistrationHandle, NixlError> {
        let mut reg_dlist = RegDescList::new(descriptor.mem_type())?;
        unsafe {
            reg_dlist.add_storage_desc(descriptor)?;
            let _opt_args = OptArgs::new()?;
            nixl_capi_register_mem(
                self.inner.write().unwrap().handle.as_ptr(),
                reg_dlist.inner.as_ptr(),
                _opt_args.inner.as_ptr(),
            );
        }
        Ok(RegistrationHandle {
            agent: self.inner.clone(),
            ptr: unsafe { descriptor.as_ptr() }.ok_or(NixlError::InvalidParam)? as usize,
            size: descriptor.size(),
            dev_id: descriptor.device_id(),
            mem_type: descriptor.mem_type(),
        })
    }

    pub fn register_memory_with_args(
        &self,
        descriptor: &dyn NixlDescriptor,
        opt_args: &OptArgs,
    ) -> Result<RegistrationHandle, NixlError> {
        let mut reg_dlist = RegDescList::new(descriptor.mem_type())?;
        unsafe {
            reg_dlist.add_storage_desc(descriptor)?;
            nixl_capi_register_mem(
                self.inner.write().unwrap().handle.as_ptr(),
                reg_dlist.inner.as_ptr(),
                opt_args.inner.as_ptr(),
            );
        }
        Ok(RegistrationHandle {
            agent: self.inner.clone(),
            ptr: unsafe { descriptor.as_ptr() }.ok_or(NixlError::InvalidParam)? as usize,
            size: descriptor.size(),
            dev_id: descriptor.device_id(),
            mem_type: descriptor.mem_type(),
        })
    }

    /// Gets the local metadata for this agent as a byte array
    ///
    /// # Returns
    /// A Vec<u8> containing the serialized metadata
    ///
    /// # Notes
    /// This call will fail if no backends have been created.
    ///
    /// # Errors
    /// Returns a NixlError if the operation fails
    pub fn get_local_md(&self) -> Result<Vec<u8>, NixlError> {
        let mut data = std::ptr::null_mut();
        let mut len = 0;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_get_local_md(
                self.inner.write().unwrap().handle.as_ptr(),
                &mut data as *mut *mut _ as *mut *mut std::ffi::c_void,
                &mut len,
            )
        };

        let data = data as *const u8;

        // Check if the data pointer is valid
        if data.is_null() {
            return Err(NixlError::InvalidDataPointer);
        }

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, data points to valid memory of size len
                let bytes = unsafe {
                    let slice = std::slice::from_raw_parts(data, len);
                    let vec = slice.to_vec();
                    libc::free(data as *mut libc::c_void);
                    vec
                };
                Ok(bytes)
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Loads remote metadata from a byte slice and returns the remote agent's name
    ///
    /// # Arguments
    /// * `metadata` - The serialized metadata as a byte slice
    ///
    /// # Returns
    /// The name of the remote agent
    ///
    /// # Errors
    /// Returns a NixlError if the operation fails
    pub fn load_remote_md(&self, metadata: &[u8]) -> Result<String, NixlError> {
        let mut agent_name = std::ptr::null_mut();

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_load_remote_md(
                self.inner.write().unwrap().handle.as_ptr(),
                metadata.as_ptr() as *const std::ffi::c_void,
                metadata.len(),
                &mut agent_name,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, agent_name points to a valid null-terminated string
                let name = unsafe {
                    let c_str = std::ffi::CStr::from_ptr(agent_name);
                    let s = c_str.to_str().unwrap().to_string();
                    libc::free(agent_name as *mut libc::c_void);
                    s
                };
                self.inner.write().unwrap().remotes.insert(name.clone());
                Ok(name)
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    pub fn invalidate_remote_md(&self, remote_agent: &str) -> Result<(), NixlError> {
        self.inner
            .write()
            .unwrap()
            .invalidate_remote_md(remote_agent)
    }

    pub fn invalidate_all_remotes(&self) -> Result<(), NixlError> {
        self.inner.write().unwrap().invalidate_all_remotes()
    }

    /// Creates a transfer request between local and remote descriptors
    ///
    /// # Arguments
    /// * `operation` - The transfer operation (read or write)
    /// * `local_descs` - The local descriptor list
    /// * `remote_descs` - The remote descriptor list
    /// * `remote_agent` - The name of the remote agent
    /// * `opt_args` - Optional arguments for the transfer
    ///
    /// # Returns
    /// A handle to the transfer request
    ///
    /// # Errors
    /// Returns a NixlError if the operation fails
    pub fn create_xfer_req(
        &self,
        operation: XferOp,
        local_descs: &XferDescList,
        remote_descs: &XferDescList,
        remote_agent: &str,
        opt_args: Option<&OptArgs>,
    ) -> Result<XferRequest, NixlError> {
        let remote_agent = CString::new(remote_agent)?;
        let mut req = std::ptr::null_mut();

        // SAFETY: All pointers are guaranteed to be valid
        let status = unsafe {
            bindings::nixl_capi_create_xfer_req(
                self.inner.read().unwrap().handle.as_ptr(),
                operation as bindings::nixl_capi_xfer_op_t,
                local_descs.inner.as_ptr(),
                remote_descs.inner.as_ptr(),
                remote_agent.as_ptr(),
                &mut req,
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, req is guaranteed to be non-null
                let inner = NonNull::new(req).ok_or(NixlError::FailedToCreateXferRequest)?;
                Ok(XferRequest {
                    inner,
                    agent: self.inner.clone(),
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::FailedToCreateXferRequest),
        }
    }
}

#[derive(Debug)]
pub struct RegistrationHandle {
    agent: Arc<RwLock<AgentInner>>,
    ptr: usize,
    size: usize,
    dev_id: u32,
    mem_type: MemType,
}

impl RegistrationHandle {
    pub fn deregister(&mut self) -> Result<(), NixlError> {
        let mut reg_dlist = RegDescList::new(self.mem_type)?;
        unsafe {
            reg_dlist.add_desc(self.ptr, self.size, self.dev_id)?;
            let _opt_args = OptArgs::new().unwrap();
            nixl_capi_deregister_mem(
                self.agent.write().unwrap().handle.as_ptr(),
                reg_dlist.inner.as_ptr(),
                _opt_args.inner.as_ptr(),
            );
        }
        Ok(())
    }
}

impl Drop for RegistrationHandle {
    fn drop(&mut self) {
        if let Err(e) = self.deregister() {
            tracing::debug!("Failed to deregister memory: {:?}", e);
        }
    }
}

/// A NIXL backend that can be used for data transfer
#[derive(Debug)]
pub struct Backend {
    inner: NonNull<bindings::nixl_capi_backend_s>,
    _agent: Arc<RwLock<AgentInner>>, // Ensures agent outlives backend
}

impl Drop for Backend {
    fn drop(&mut self) {
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

    /// Add a backend to the optional arguments
    pub fn add_backend(&mut self, backend: &Backend) -> Result<(), NixlError> {
        let status =
            unsafe { nixl_capi_opt_args_add_backend(self.inner.as_ptr(), backend.inner.as_ptr()) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
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
pub struct XferDescList<'a> {
    inner: NonNull<bindings::nixl_capi_xfer_dlist_s>,
    _phantom: PhantomData<&'a dyn NixlDescriptor>,
}

impl<'a> XferDescList<'a> {
    /// Creates a new transfer descriptor list for the given memory type
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        let mut dlist = ptr::null_mut();
        let status =
            unsafe { nixl_capi_create_xfer_dlist(mem_type as nixl_capi_mem_type_t, &mut dlist) };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, dlist is non-null
                let inner = unsafe { NonNull::new_unchecked(dlist) };
                Ok(Self {
                    inner,
                    _phantom: PhantomData,
                })
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

    /// Add a descriptor from a type implementing NixlDescriptor
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The descriptor remains valid for the lifetime of the list
    /// - The memory region pointed to by the descriptor remains valid
    pub fn add_storage_desc<D: NixlDescriptor + 'a>(
        &mut self,
        desc: &'a D,
    ) -> Result<(), NixlError> {
        // Validate memory type matches
        let desc_mem_type = desc.mem_type();
        let list_mem_type = unsafe {
            // Get the memory type from the list by checking first descriptor
            let mut len = 0;
            match nixl_capi_xfer_dlist_len(self.inner.as_ptr(), &mut len) {
                0 => Ok(()),
                -1 => Err(NixlError::InvalidParam),
                _ => Err(NixlError::BackendError),
            }?;
            if len > 0 {
                // TODO: Add API to get descriptor memory type
                MemType::Unknown
            } else {
                desc_mem_type
            }
        };

        if desc_mem_type != list_mem_type && list_mem_type != MemType::Unknown {
            return Err(NixlError::InvalidParam);
        }

        // Get descriptor details
        let addr = unsafe { desc.as_ptr() }.ok_or(NixlError::InvalidParam)? as usize;
        let len = desc.size();
        let dev_id = desc.device_id();

        // Add to list
        self.add_desc(addr, len, dev_id)
    }
}

impl<'a> Drop for XferDescList<'a> {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_xfer_dlist(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around a NIXL registration descriptor list
pub struct RegDescList<'a> {
    inner: NonNull<bindings::nixl_capi_reg_dlist_s>,
    _phantom: PhantomData<&'a dyn NixlDescriptor>,
}

impl<'a> RegDescList<'a> {
    /// Creates a new registration descriptor list for the given memory type
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        let mut dlist = ptr::null_mut();
        let status =
            unsafe { nixl_capi_create_reg_dlist(mem_type as nixl_capi_mem_type_t, &mut dlist) };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, dlist is non-null
                let inner = unsafe { NonNull::new_unchecked(dlist) };
                Ok(Self {
                    inner,
                    _phantom: PhantomData,
                })
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

    /// Add a descriptor from a type implementing NixlDescriptor
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The descriptor remains valid for the lifetime of the list
    /// - The memory region pointed to by the descriptor remains valid
    pub fn add_storage_desc(&mut self, desc: &'a dyn NixlDescriptor) -> Result<(), NixlError> {
        // Validate memory type matches
        let desc_mem_type = desc.mem_type();
        let list_mem_type = unsafe {
            // Get the memory type from the list by checking first descriptor
            let mut len = 0;
            match nixl_capi_reg_dlist_len(self.inner.as_ptr(), &mut len) {
                0 => Ok(()),
                -1 => Err(NixlError::InvalidParam),
                _ => Err(NixlError::BackendError),
            }?;
            if len > 0 {
                // TODO: Add API to get descriptor memory type
                MemType::Unknown
            } else {
                desc_mem_type
            }
        };

        if desc_mem_type != list_mem_type && list_mem_type != MemType::Unknown {
            return Err(NixlError::InvalidParam);
        }

        // Get descriptor details
        let addr = unsafe { desc.as_ptr() }.ok_or(NixlError::InvalidParam)? as usize;
        let len = desc.size();
        let dev_id = desc.device_id();

        // Add to list
        self.add_desc(addr, len, dev_id)
    }
}

impl<'a> Drop for RegDescList<'a> {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_reg_dlist(self.inner.as_ptr());
        }
    }
}

/// A trait for storage types that can be used with NIXL
pub trait MemoryRegion: std::fmt::Debug + Send + Sync + 'static {
    /// Get a raw pointer to the storage
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the storage is dropped
    /// - Access patterns respect the storage's thread safety model
    unsafe fn as_ptr(&self) -> Option<*const u8>;

    /// Returns the total size of the storage in bytes
    fn size(&self) -> usize;
}

/// A trait for types that can be added to NIXL descriptor lists
pub trait NixlDescriptor: MemoryRegion {
    /// Get the memory type for this descriptor
    fn mem_type(&self) -> MemType;

    /// Get the device ID for this memory region
    fn device_id(&self) -> u32;

    /// Is registered
    fn is_registered(&self) -> bool;

    /// Registration Handle
    fn handle(&self) -> Option<&RegistrationHandle>;
}

/// A trait for types that can be registered with NIXL
pub trait NixlRegistration: NixlDescriptor {
    fn register(&mut self, agent: &Agent) -> Result<(), NixlError>;
    fn register_with_args(&mut self, agent: &Agent, opt_args: &OptArgs) -> Result<(), NixlError>;
}

/// System memory storage implementation using a Vec<u8>
#[derive(Debug)]
pub struct SystemStorage {
    data: Vec<u8>,
    handle: Option<RegistrationHandle>,
}

impl SystemStorage {
    /// Create a new system storage with the given size
    pub fn new(size: usize) -> Result<Self, NixlError> {
        let mut data = Vec::with_capacity(size);
        // Initialize to zero to ensure consistent behavior
        data.resize(size, 0);
        Ok(Self { data, handle: None })
    }

    /// Fill the storage with a specific byte value
    pub fn memset(&mut self, value: u8) {
        self.data.fill(value);
    }

    /// Get a slice of the underlying data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

impl MemoryRegion for SystemStorage {
    fn size(&self) -> usize {
        self.data.len()
    }

    unsafe fn as_ptr(&self) -> Option<*const u8> {
        Some(self.data.as_ptr())
    }
}

impl NixlDescriptor for SystemStorage {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }

    fn device_id(&self) -> u32 {
        0
    }

    fn is_registered(&self) -> bool {
        self.handle.is_some()
    }

    fn handle(&self) -> Option<&RegistrationHandle> {
        self.handle.as_ref()
    }
}

impl NixlRegistration for SystemStorage {
    fn register(&mut self, agent: &Agent) -> Result<(), NixlError> {
        let handle = agent.register_memory(self)?;
        self.handle = Some(handle);
        Ok(())
    }

    fn register_with_args(&mut self, agent: &Agent, opt_args: &OptArgs) -> Result<(), NixlError> {
        let handle = agent.register_memory_with_args(self, opt_args)?;
        self.handle = Some(handle);
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum XferOp {
    Write = 0,
    Read = 1,
}

/// A handle to a transfer request
pub struct XferRequest {
    inner: NonNull<bindings::nixl_capi_xfer_req_s>,
    agent: Arc<RwLock<AgentInner>>,
}

// SAFETY: XferRequest can be sent between threads safely
unsafe impl Send for XferRequest {}
// SAFETY: XferRequest can be shared between threads safely
unsafe impl Sync for XferRequest {}

impl Drop for XferRequest {
    fn drop(&mut self) {
        unsafe {
            bindings::nixl_capi_release_xfer_req(
                self.agent.write().unwrap().handle.as_ptr(),
                self.inner.as_ptr(),
            );

            bindings::nixl_capi_destroy_xfer_req(self.inner.as_ptr());
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
        let (_mems, params) = agent
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
        let (_mems, params) = agent.get_plugin_params(&plugin_name).unwrap();
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

        // add descriptors with overlaps
        dlist.add_desc(0x1000, 0x100, 0).unwrap();
        dlist.add_desc(0x1050, 0x100, 0).unwrap();
        assert!(dlist.has_overlaps().unwrap());
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

    #[test]
    fn test_storage_descriptor_lifetime() {
        // Create storage that outlives the descriptor list
        let storage = SystemStorage::new(1024).unwrap();

        {
            // Create a descriptor list with shorter lifetime
            let mut dlist = XferDescList::new(MemType::Dram).unwrap();
            dlist.add_storage_desc(&storage).unwrap();
            assert_eq!(dlist.len().unwrap(), 1);
            // dlist is dropped here, but storage is still valid
        }

        // MemoryRegion is still valid here
        assert_eq!(<SystemStorage as MemoryRegion>::size(&storage), 1024);
    }

    #[test]
    fn test_multiple_storage_descriptors() {
        let storage1 = SystemStorage::new(1024).unwrap();
        let storage2 = SystemStorage::new(2048).unwrap();

        let mut dlist = XferDescList::new(MemType::Dram).unwrap();

        // Add multiple descriptors
        dlist.add_storage_desc(&storage1).unwrap();
        dlist.add_storage_desc(&storage2).unwrap();

        assert_eq!(dlist.len().unwrap(), 2);
    }

    #[test]
    fn test_basic_agent_lifecycle() {
        // Create two agents
        let agent1 = Agent::new("A1").unwrap();
        let agent2 = Agent::new("A2").unwrap();

        // Get available plugins and print their names
        let plugins = agent1.get_available_plugins().unwrap();
        for plugin in plugins.iter() {
            println!("Found plugin: {}", plugin.unwrap());
        }

        // Get plugin parameters for both agents
        let (_mem_list1, params1) = agent1.get_plugin_params("UCX").unwrap();
        let (_mem_list2, params2) = agent2.get_plugin_params("UCX").unwrap();

        // Create backends for both agents
        let backend1 = agent1.create_backend("UCX", &params1).unwrap();
        let backend2 = agent2.create_backend("UCX", &params2).unwrap();

        // Create optional arguments and add backends
        let mut opt_args = OptArgs::new().unwrap();
        opt_args.add_backend(&backend1).unwrap();
        opt_args.add_backend(&backend2).unwrap();

        // Allocate and initialize memory regions
        let mut storage1 = SystemStorage::new(256).unwrap();
        let mut storage2 = SystemStorage::new(256).unwrap();

        // Initialize memory patterns
        storage1.memset(0xbb);
        storage2.memset(0x00);

        // Verify memory patterns
        assert!(storage1.as_slice().iter().all(|&x| x == 0xbb));
        assert!(storage2.as_slice().iter().all(|&x| x == 0x00));

        // Create registration descriptor lists
        storage1.register(&agent1).unwrap();
        storage2.register(&agent2).unwrap();

        // Mimic transferring metadata from agent2 to agent1
        let metadata = agent2.get_local_md().unwrap();
        let remote_name = agent1.load_remote_md(&metadata).unwrap();
        assert_eq!(remote_name, "A2");

        let mut local_xfer_dlist = XferDescList::new(MemType::Dram).unwrap();
        local_xfer_dlist.add_storage_desc(&storage1).unwrap();

        let mut remote_xfer_dlist = XferDescList::new(MemType::Dram).unwrap();
        remote_xfer_dlist.add_storage_desc(&storage2).unwrap();

        let xfer_args = OptArgs::new().unwrap();

        let xfer_req = agent1
            .create_xfer_req(
                XferOp::Write,
                &local_xfer_dlist,
                &remote_xfer_dlist,
                &remote_name,
                Some(&xfer_args),
            )
            .unwrap();

        drop(xfer_req);

        // Invalidate all remotes
        agent1.invalidate_all_remotes().unwrap();
        agent2.invalidate_all_remotes().unwrap();
    }

    #[test]
    fn test_memory_registration() {
        let agent = Agent::new("test_agent").unwrap();
        let mut storage = SystemStorage::new(1024).unwrap();

        // Test initial state
        assert!(!storage.is_registered());
        assert!(storage.handle().is_none());

        // Register memory
        storage.register(&agent).unwrap();

        // Verify registration
        assert!(storage.is_registered());
        assert!(storage.handle().is_some());

        let handle = storage.handle().unwrap();
        assert_eq!(handle.size, 1024);
        assert_eq!(handle.mem_type, MemType::Dram);
        assert_eq!(handle.dev_id, 0);

        // Verify we can still access the memory
        storage.memset(0xAA);
        assert!(storage.as_slice().iter().all(|&x| x == 0xAA));
    }

    #[test]
    fn test_registration_handle_drop() {
        let agent = Agent::new("test_agent").unwrap();
        let mut storage = SystemStorage::new(1024).unwrap();

        // Register memory
        storage.register(&agent).unwrap();
        assert!(storage.is_registered());

        // Drop the storage, which should trigger deregistration
        drop(storage);

        // Create new storage to verify we can register again
        let mut new_storage = SystemStorage::new(1024).unwrap();
        new_storage.register(&agent).unwrap();
        assert!(new_storage.is_registered());
    }

    #[test]
    fn test_multiple_registrations() {
        let agent = Agent::new("test_agent").unwrap();
        let mut storage1 = SystemStorage::new(1024).unwrap();
        let mut storage2 = SystemStorage::new(2048).unwrap();

        // Register both storages
        storage1.register(&agent).unwrap();
        storage2.register(&agent).unwrap();

        // Verify both are registered with correct sizes
        assert!(storage1.is_registered());
        assert!(storage2.is_registered());
        assert_eq!(storage1.handle().unwrap().size, 1024);
        assert_eq!(storage2.handle().unwrap().size, 2048);

        // Verify we can still access both memories
        storage1.memset(0xAA);
        storage2.memset(0xBB);
        assert!(storage1.as_slice().iter().all(|&x| x == 0xAA));
        assert!(storage2.as_slice().iter().all(|&x| x == 0xBB));
    }

    #[test]
    fn test_get_local_md() {
        let agent = Agent::new("test_agent").unwrap();

        // Get available plugins and print their names
        let plugins = agent.get_available_plugins().unwrap();
        for plugin in plugins.iter() {
            println!("Found plugin: {}", plugin.unwrap());
        }

        // Get plugin parameters for both agents
        let (_mem_list, params) = agent.get_plugin_params("UCX").unwrap();

        // Create backends for both agents
        let backend1 = agent.create_backend("UCX", &params).unwrap();

        let md = agent.get_local_md().unwrap();

        // Measure the size
        let initial_size = md.len();
        println!("Local metadata size: {}", initial_size);

        let mut opt_args = OptArgs::new().unwrap();
        opt_args.add_backend(&backend1).unwrap();

        let mut storages = Vec::new();

        for _i in 0..10 {
            // Register some memory regions
            let mut storage = SystemStorage::new(1024).unwrap();
            storage.register_with_args(&agent, &opt_args).unwrap();
            assert!(storage.is_registered());
            storages.push(storage);
        }

        let md = agent.get_local_md().unwrap();

        // Measure the size again
        let final_size = md.len();
        println!("Local metadata size: {}", final_size);

        // Check if the size has increased
        assert!(final_size > initial_size);
    }

    #[test]
    fn test_metadata_exchange() {
        // Create two agents
        let agent1 = Agent::new("agent1").unwrap();
        let agent2 = Agent::new("agent2").unwrap();

        // Get plugin parameters for both agents
        let (_mem_list, params) = agent1.get_plugin_params("UCX").unwrap();

        // Create backends for both agents
        let _backend1 = agent1.create_backend("UCX", &params).unwrap();
        let _backend2 = agent2.create_backend("UCX", &params).unwrap();

        // Get metadata from agent1
        let md = agent1.get_local_md().unwrap();

        // Load metadata into agent2
        let remote_name = agent2.load_remote_md(&md).unwrap();
        assert_eq!(remote_name, "agent1");
    }

    #[test]
    fn test_create_xfer_req() {
        let agent = Agent::new("test_agent").unwrap();

        // Create local and remote descriptor lists
        let local_descs = XferDescList::new(MemType::Dram).unwrap();
        let remote_descs = XferDescList::new(MemType::Dram).unwrap();

        // Create a transfer request
        let _req = agent
            .create_xfer_req(
                XferOp::Write,
                &local_descs,
                &remote_descs,
                "remote_agent",
                None,
            )
            .unwrap();
    }
}
