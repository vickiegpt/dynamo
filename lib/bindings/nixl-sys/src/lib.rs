//! Raw FFI bindings to the NIXL library
//!
//! This crate provides low-level bindings to the NIXL C++ library.
//! It is not meant to be used directly, but rather through the higher-level
//! `nixl` crate.

use libc::size_t;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::ptr::NonNull;
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
    nixl_capi_agent_t, nixl_capi_create_agent, nixl_capi_destroy_agent, nixl_capi_destroy_mem_list,
    nixl_capi_destroy_params, nixl_capi_destroy_string_list, nixl_capi_get_available_plugins,
    nixl_capi_get_plugin_params, nixl_capi_mem_list_t, nixl_capi_params_t,
    nixl_capi_string_list_get, nixl_capi_string_list_size, nixl_capi_string_list_t,
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
}

impl Drop for Agent {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_agent(self.inner.as_ptr());
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
}
