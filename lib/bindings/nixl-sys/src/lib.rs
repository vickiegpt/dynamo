//! Raw FFI bindings to the NIXL library
//!
//! This crate provides low-level bindings to the NIXL C++ library.
//! It is not meant to be used directly, but rather through the higher-level
//! `nixl` crate.

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

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::*;

    #[test]
    fn test_agent_creation() {
        let name = CString::new("test_agent").unwrap();
        let mut agent = std::ptr::null_mut();

        unsafe {
            let status = nixl_capi_create_agent(name.as_ptr(), &mut agent);
            assert_eq!(status, 0); // NIXL_CAPI_SUCCESS

            let status = nixl_capi_destroy_agent(agent);
            assert_eq!(status, 0); // NIXL_CAPI_SUCCESS
        }
    }
}
