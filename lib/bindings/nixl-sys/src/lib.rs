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
    // include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
    include!("./bindings.rs");
}

// Re-export types from the included bindings
pub use bindings::{nixl_capi_agent_t, nixl_capi_create_agent, nixl_capi_destroy_agent};

/// Errors that can occur when using NIXL
#[derive(Error, Debug)]
pub enum NixlError {
    #[error("Invalid parameter provided to NIXL")]
    InvalidParam,
    #[error("Backend error occurred")]
    BackendError,
    #[error("Failed to create CString from input: {0}")]
    StringConversionError(#[from] std::ffi::NulError),
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
}
