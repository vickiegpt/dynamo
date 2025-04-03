//! Raw FFI bindings to the NIXL library
//!
//! This crate provides low-level bindings to the NIXL C++ library.
//! It is not meant to be used directly, but rather through the higher-level
//! `nixl` crate.

use std::ffi::{CStr, CString};
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
    nixlAgent, nixlAgentConfig, nixlBackendH, nixlBasicDesc, nixlBlobDesc, nixlXferReqH,
    nixl_b_params_t, nixl_create_agent, nixl_create_backend, nixl_create_xfer_req,
    nixl_deregister_mem, nixl_destroy_agent, nixl_get_avail_plugins, nixl_get_backend_params,
    nixl_get_local_md, nixl_get_notifs, nixl_get_plugin_params, nixl_get_xfer_status,
    nixl_invalidate_remote_md, nixl_load_remote_md, nixl_mem_list_t, nixl_mem_t, nixl_notifs_t,
    nixl_opt_args_t, nixl_post_xfer_req, nixl_reg_dlist_t, nixl_register_mem,
    nixl_release_xfer_req, nixl_status_t, nixl_xfer_dlist_t, nixl_xfer_op_t,
};

// Constants from the C header
pub const NIXL_SUCCESS: i32 = 1;
pub const NIXL_ERR_BACKEND: i32 = 4;
pub const NIXL_ERR_NOT_FOUND: i32 = 5;
pub const NIXL_ERR_NOT_SUPPORTED: i32 = 10;
pub const NIXL_ERR_INVALID_ARG: i32 = 3;
pub const NIXL_ERR_SYSTEM: i32 = -1; // Using -1 as a placeholder
pub const NIXL_ERR_BUSY: i32 = -2; // Using -2 as a placeholder
pub const NIXL_ERR_WOULD_BLOCK: i32 = -3; // Using -3 as a placeholder

/// Error type for NIXL operations
#[derive(Error, Debug)]
pub enum NixlError {
    #[error("Backend error: {0}")]
    Backend(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("System error: {0}")]
    System(String),
    #[error("Operation not supported")]
    NotSupported,
    #[error("Resource busy")]
    Busy,
    #[error("Resource not found")]
    NotFound,
    #[error("Operation would block")]
    WouldBlock,
    #[error("Unknown error code: {0}")]
    Unknown(i32),
}

impl From<i32> for NixlError {
    fn from(status: i32) -> Self {
        match status {
            NIXL_SUCCESS => unreachable!("Success is not an error"),
            NIXL_ERR_BACKEND => NixlError::Backend("Backend operation failed".into()),
            NIXL_ERR_INVALID_ARG => NixlError::InvalidArgument("Invalid argument provided".into()),
            NIXL_ERR_SYSTEM => NixlError::System("System error occurred".into()),
            NIXL_ERR_NOT_SUPPORTED => NixlError::NotSupported,
            NIXL_ERR_BUSY => NixlError::Busy,
            NIXL_ERR_NOT_FOUND => NixlError::NotFound,
            NIXL_ERR_WOULD_BLOCK => NixlError::WouldBlock,
            other => NixlError::Unknown(other),
        }
    }
}

/// Safe wrapper around NIXL agent configuration
#[derive(Debug)]
pub struct AgentConfig {
    inner: nixlAgentConfig,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            inner: nixlAgentConfig {
                useProgThread: true,
                pthrDelay: 1000, // Default delay of 1000us
            },
        }
    }
}

impl AgentConfig {
    /// Create a new agent configuration with custom settings
    pub fn new(use_prog_thread: bool, thread_delay_us: u64) -> Self {
        Self {
            inner: nixlAgentConfig {
                useProgThread: use_prog_thread,
                pthrDelay: thread_delay_us,
            },
        }
    }
}

/// Safe wrapper around NIXL agent
#[derive(Debug)]
pub struct Agent {
    inner: NonNull<nixlAgent>,
    name: String,
}

/// Wrapper type for NIXL status codes
#[derive(Debug, Clone, Copy)]
pub struct NixlStatus(nixl_status_t);

impl NixlStatus {
    /// Convert the status to a Result
    pub fn to_result(self) -> Result<(), NixlError> {
        match self.0 {
            NIXL_SUCCESS => Ok(()),
            NIXL_ERR_BACKEND => Err(NixlError::Backend("Backend operation failed".into())),
            NIXL_ERR_INVALID_ARG => Err(NixlError::InvalidArgument(
                "Invalid argument provided".into(),
            )),
            NIXL_ERR_SYSTEM => Err(NixlError::System("System error occurred".into())),
            NIXL_ERR_NOT_SUPPORTED => Err(NixlError::NotSupported),
            NIXL_ERR_BUSY => Err(NixlError::Busy),
            NIXL_ERR_NOT_FOUND => Err(NixlError::NotFound),
            NIXL_ERR_WOULD_BLOCK => Err(NixlError::WouldBlock),
            other => Err(NixlError::Unknown(other)),
        }
    }

    /// Create a new NixlStatus from a raw status code
    pub fn from_raw(status: nixl_status_t) -> Self {
        NixlStatus(status)
    }

    /// Get the raw status code
    pub fn as_raw(&self) -> nixl_status_t {
        self.0
    }
}

impl From<nixl_status_t> for NixlStatus {
    fn from(status: nixl_status_t) -> Self {
        NixlStatus(status)
    }
}

impl Agent {
    /// Create a new NIXL agent with the given name and configuration
    pub fn new(name: impl Into<String>, mut config: AgentConfig) -> Result<Self, NixlError> {
        let name = name.into();
        let c_name = CString::new(name.clone())
            .map_err(|_| NixlError::InvalidArgument("Name contains null bytes".into()))?;
        let mut agent_ptr = std::ptr::null_mut();

        // SAFETY: We ensure the name is valid and properly null-terminated
        unsafe {
            NixlStatus::from(nixl_create_agent(
                c_name.as_ptr(),
                &mut config.inner as *mut _,
                &mut agent_ptr,
            ))
            .to_result()?;

            Ok(Self {
                inner: NonNull::new(agent_ptr)
                    .ok_or(NixlError::System("Failed to create agent".into()))?,
                name,
            })
        }
    }

    /// Get available plugins for this agent
    pub fn available_plugins(&self) -> Result<Vec<String>, NixlError> {
        let mut plugins = Vec::new();
        let mut count = 0;
        let mut plugins_ptr = std::ptr::null_mut();

        // SAFETY: We own the agent and ensure proper cleanup
        unsafe {
            NixlStatus::from(nixl_get_avail_plugins(
                self.inner.as_ptr(),
                &mut plugins_ptr,
                &mut count,
            ))
            .to_result()?;

            // Convert raw plugin array to Vec<String>
            let plugins_slice = std::slice::from_raw_parts(plugins_ptr as *const *const i8, count);
            plugins.extend(
                plugins_slice
                    .iter()
                    .map(|&p| CStr::from_ptr(p).to_string_lossy().into_owned()),
            );

            // Free the plugin array
            libc::free(plugins_ptr as *mut _);
        }

        Ok(plugins)
    }

    /// Get plugin parameters
    pub fn plugin_params(
        &self,
        plugin_name: &str,
    ) -> Result<(nixl_mem_list_t, nixl_b_params_t), NixlError> {
        let c_name = CString::new(plugin_name)
            .map_err(|_| NixlError::InvalidArgument("Plugin name contains null bytes".into()))?;
        let mut mems = nixl_mem_list_t::default();
        let mut params = nixl_b_params_t::default();

        // SAFETY: We ensure the plugin name is valid and properly null-terminated
        unsafe {
            NixlStatus::from(nixl_get_plugin_params(
                self.inner.as_ptr(),
                c_name.as_ptr(),
                &mut mems,
                &mut params,
            ))
            .to_result()?;
        }

        Ok((mems, params))
    }

    /// Create a backend instance
    pub fn create_backend(
        &self,
        plugin_name: &str,
        params: &mut nixl_b_params_t,
    ) -> Result<NonNull<nixlBackendH>, NixlError> {
        let c_name = CString::new(plugin_name)
            .map_err(|_| NixlError::InvalidArgument("Plugin name contains null bytes".into()))?;
        let mut backend_ptr = std::ptr::null_mut();

        // SAFETY: We ensure the plugin name is valid and properly null-terminated
        unsafe {
            NixlStatus::from(nixl_create_backend(
                self.inner.as_ptr(),
                c_name.as_ptr(),
                params as *mut _,
                &mut backend_ptr,
            ))
            .to_result()?;

            Ok(NonNull::new(backend_ptr)
                .ok_or(NixlError::System("Failed to create backend".into()))?)
        }
    }

    /// Get local metadata
    pub fn local_metadata(&self) -> Result<String, NixlError> {
        let mut metadata_ptr = std::ptr::null_mut();

        // SAFETY: We own the agent and ensure proper cleanup
        unsafe {
            NixlStatus::from(nixl_get_local_md(self.inner.as_ptr(), &mut metadata_ptr))
                .to_result()?;
            let metadata = CStr::from_ptr(metadata_ptr).to_string_lossy().into_owned();
            libc::free(metadata_ptr as *mut _);
            Ok(metadata)
        }
    }

    /// Load remote metadata
    pub fn load_remote_metadata(&self, metadata: &str) -> Result<String, NixlError> {
        let c_metadata = CString::new(metadata)
            .map_err(|_| NixlError::InvalidArgument("Metadata contains null bytes".into()))?;
        let mut result_ptr = std::ptr::null_mut();

        // SAFETY: We ensure the metadata is valid and properly null-terminated
        unsafe {
            NixlStatus::from(nixl_load_remote_md(
                self.inner.as_ptr(),
                c_metadata.as_ptr(),
                &mut result_ptr,
            ))
            .to_result()?;
            let result = CStr::from_ptr(result_ptr).to_string_lossy().into_owned();
            libc::free(result_ptr as *mut _);
            Ok(result)
        }
    }

    /// Register memory with the backend
    pub fn register_memory(
        &self,
        dlist: &mut nixl_reg_dlist_t,
        extra_params: Option<&mut nixl_opt_args_t>,
    ) -> Result<(), NixlError> {
        // SAFETY: We own the agent and the dlist is valid
        unsafe {
            NixlStatus::from(nixl_register_mem(
                self.inner.as_ptr(),
                dlist as *mut _,
                extra_params.map_or(std::ptr::null_mut(), |p| p as *mut _),
            ))
            .to_result()
        }
    }

    /// Deregister memory from the backend
    pub fn deregister_memory(
        &self,
        dlist: &mut nixl_reg_dlist_t,
        extra_params: Option<&mut nixl_opt_args_t>,
    ) -> Result<(), NixlError> {
        // SAFETY: We own the agent and the dlist is valid
        unsafe {
            NixlStatus::from(nixl_deregister_mem(
                self.inner.as_ptr(),
                dlist as *mut _,
                extra_params.map_or(std::ptr::null_mut(), |p| p as *mut _),
            ))
            .to_result()
        }
    }

    /// Create a transfer request
    pub fn create_transfer_request(
        &self,
        op: nixl_xfer_op_t,
        src_descs: &mut nixl_xfer_dlist_t,
        dst_descs: &mut nixl_xfer_dlist_t,
        remote_agent: &str,
        extra_params: Option<&mut nixl_opt_args_t>,
    ) -> Result<NonNull<nixlXferReqH>, NixlError> {
        let c_remote = CString::new(remote_agent).map_err(|_| {
            NixlError::InvalidArgument("Remote agent name contains null bytes".into())
        })?;
        let mut req_handle = std::ptr::null_mut();

        // SAFETY: We ensure all parameters are valid and properly null-terminated
        unsafe {
            NixlStatus::from(nixl_create_xfer_req(
                self.inner.as_ptr(),
                op,
                src_descs as *mut _,
                dst_descs as *mut _,
                c_remote.as_ptr(),
                &mut req_handle,
                extra_params.map_or(std::ptr::null_mut(), |p| p as *mut _),
            ))
            .to_result()?;

            Ok(NonNull::new(req_handle).ok_or(NixlError::System(
                "Failed to create transfer request".into(),
            ))?)
        }
    }

    /// Post a transfer request
    pub fn post_transfer_request(
        &self,
        req_handle: NonNull<nixlXferReqH>,
    ) -> Result<(), NixlError> {
        // SAFETY: We own the agent and the request handle is valid
        unsafe {
            NixlStatus::from(nixl_post_xfer_req(self.inner.as_ptr(), req_handle.as_ptr()))
                .to_result()
        }
    }

    /// Get transfer request status
    pub fn get_transfer_status(
        &self,
        req_handle: NonNull<nixlXferReqH>,
    ) -> Result<nixl_status_t, NixlError> {
        // SAFETY: We own the agent and the request handle is valid
        unsafe {
            let status = nixl_get_xfer_status(self.inner.as_ptr(), req_handle.as_ptr());
            if status < 0 {
                Err(NixlError::from(status))
            } else {
                Ok(status)
            }
        }
    }

    /// Release a transfer request
    pub fn release_transfer_request(
        &self,
        req_handle: NonNull<nixlXferReqH>,
    ) -> Result<(), NixlError> {
        // SAFETY: We own the agent and the request handle is valid
        unsafe {
            NixlStatus::from(nixl_release_xfer_req(
                self.inner.as_ptr(),
                req_handle.as_ptr(),
            ))
            .to_result()
        }
    }

    /// Get notifications
    pub fn get_notifications(&self) -> Result<nixl_notifs_t, NixlError> {
        let mut notifs = nixl_notifs_t::default();

        // SAFETY: We own the agent and ensure proper cleanup
        unsafe {
            NixlStatus::from(nixl_get_notifs(self.inner.as_ptr(), &mut notifs)).to_result()?;
        }

        Ok(notifs)
    }

    /// Invalidate remote metadata
    pub fn invalidate_remote_metadata(&self, remote_agent: &str) -> Result<(), NixlError> {
        let c_remote = CString::new(remote_agent).map_err(|_| {
            NixlError::InvalidArgument("Remote agent name contains null bytes".into())
        })?;

        // SAFETY: We ensure the remote agent name is valid and properly null-terminated
        unsafe {
            NixlStatus::from(nixl_invalidate_remote_md(
                self.inner.as_ptr(),
                c_remote.as_ptr(),
            ))
            .to_result()
        }
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        // SAFETY: We own the agent and are dropping it
        unsafe {
            nixl_destroy_agent(self.inner.as_ptr());
        }
    }
}

/// Safe wrapper around memory buffer operations
#[derive(Debug)]
pub struct Buffer {
    ptr: NonNull<u8>,
    len: usize,
}

impl Buffer {
    /// Create a new buffer with the given size
    pub fn new(size: usize) -> Result<Self, NixlError> {
        // SAFETY: We use libc::calloc which is safe to use for allocation
        unsafe {
            let ptr = NonNull::new(libc::calloc(1, size) as *mut u8)
                .ok_or(NixlError::System("Failed to allocate memory".into()))?;
            Ok(Self { ptr, len: size })
        }
    }

    /// Fill the buffer with a specific value
    pub fn fill(&mut self, value: u8) {
        // SAFETY: We own the buffer and know its size
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), value, self.len);
        }
    }

    /// Get a slice of the buffer
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: We own the buffer and know its size
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: We own the buffer and know its size
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get the raw pointer to the buffer
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get the mutable raw pointer to the buffer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: We own the buffer and are dropping it
        unsafe {
            libc::free(self.ptr.as_ptr() as *mut libc::c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let config = AgentConfig::default();
        let agent = Agent::new("test_agent", config).expect("Failed to create agent");
        let plugins = agent.available_plugins().expect("Failed to get plugins");
        println!("Available plugins: {:?}", plugins);
    }
}
