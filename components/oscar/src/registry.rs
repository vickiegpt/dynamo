// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    ContentHash, ObjectInfo, OscarError, OscarResult, SharedObject, StorageBackend,
};
use crate::v2::descriptors::CallerContext;
use dynamo_runtime::DistributedRuntime;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Object metadata stored in etcd for atomic registration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectMetadata {
    /// BLAKE3 hash as hex string
    pub hash: String,
    /// Size in bytes
    pub size: u64,
}

impl ObjectMetadata {
    /// Create new object metadata
    pub fn new(hash: ContentHash, size: u64) -> Self {
        Self {
            hash: hash.to_hex(),
            size,
        }
    }

    /// Serialize to JSON bytes for etcd storage
    pub fn to_json_bytes(&self) -> OscarResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(OscarError::Serialization)
    }

    /// Deserialize from JSON bytes
    pub fn from_json_bytes(bytes: &[u8]) -> OscarResult<Self> {
        serde_json::from_slice(bytes).map_err(OscarError::Serialization)
    }
}

/// Result of object registration operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistrationOutcome {
    /// New object was created and registered
    Created,
    /// Existing object was validated and new lease reference added
    Validated,
}

/// Registry for managing shared objects across leases.
pub struct ObjectRegistry {
    runtime: DistributedRuntime,
    storage: Box<dyn StorageBackend>,
}

impl ObjectRegistry {
    /// Create a new object registry.
    pub fn new(runtime: DistributedRuntime, storage: Box<dyn StorageBackend>) -> Self {
        Self { runtime, storage }
    }

    /// Register a new object or get existing one.
    pub async fn register_object(
        &self,
        object_name: String,
        data: bytes::Bytes,
        caller_context: CallerContext,
    ) -> OscarResult<RegistrationOutcome> {
        // 1. Validate content size and hash
        use crate::hash::ObjectHasher;
        ObjectHasher::validate_size(&data)?;
        let content_hash = ObjectHasher::hash(&data);

        // 2. Create object descriptor and generate keys
        let object_descriptor = crate::v2::descriptors::ObjectDescriptor::create(
            object_name, content_hash.clone(), caller_context
        ).map_err(|e| OscarError::InvalidOperation { 
            reason: format!("Failed to create object descriptor: {}", e) 
        })?;

        let metadata = ObjectMetadata::new(content_hash.clone(), data.len() as u64);
        let metadata_bytes = metadata.to_json_bytes()?;

        // 3. Generate etcd keys using v2 system
        let object_key = crate::v2::keys::OscarKeysV2::object_metadata_key(&object_descriptor)
            .map_err(|e| OscarError::InvalidOperation { 
                reason: format!("Failed to generate object key: {}", e) 
            })?;

        let lease_id = self.runtime.primary_lease()
            .ok_or_else(|| OscarError::InvalidOperation { 
                reason: "No primary lease available".to_string() 
            })?.id();
        let lease_ref_key = crate::v2::keys::OscarKeysV2::lease_reference_key(&object_descriptor, lease_id)
            .map_err(|e| OscarError::InvalidOperation { 
                reason: format!("Failed to generate lease reference key: {}", e) 
            })?;

        let lease_ref_value = content_hash.to_hex().into_bytes();

        // 4. Get etcd client
        let etcd_client = self.runtime.etcd_client()
            .ok_or_else(|| OscarError::InvalidOperation { 
                reason: "ETCD client not available".to_string() 
            })?;

        // 5. Try atomic create first (new object registration)
        let create_result = etcd_client.kv_create_and_put(
            object_key.clone(),
            metadata_bytes.clone(),
            None, // Use default lease for object metadata
            lease_ref_key.clone(),
            lease_ref_value.clone(),
            Some(lease_id),
        ).await;

        match create_result {
            Ok(()) => {
                // Successfully created new object
                Ok(RegistrationOutcome::Created)
            },
            Err(_) => {
                // Object already exists, try validation and attach
                let validate_result = etcd_client.kv_validate_and_put(
                    object_key,
                    metadata_bytes,
                    lease_ref_key,
                    lease_ref_value,
                    Some(lease_id),
                ).await;

                match validate_result {
                    Ok(()) => Ok(RegistrationOutcome::Validated),
                    Err(e) => Err(OscarError::Concurrency { 
                        reason: format!("Failed to validate and attach lease reference: {}", e) 
                    }),
                }
            }
        }
    }

    /// Get an object by hash and add reference.
    pub async fn get_object(&self, hash: &ContentHash, lease_id: Uuid) -> OscarResult<SharedObject> {
        // TODO: Implement get logic
        // 1. Check if object exists and available
        // 2. Add reference atomically
        // 3. Load data from storage backend
        todo!("Implementation in DYN-955")
    }

    /// Remove reference to an object.
    pub async fn remove_reference(
        &self,
        hash: &ContentHash,
        lease_id: Uuid,
    ) -> OscarResult<()> {
        // TODO: Implement reference removal
        // 1. Remove reference atomically
        // 2. If last reference, mark for deletion
        // 3. Background cleanup will handle actual deletion
        todo!("Implementation in DYN-956")
    }

    /// List objects for a lease.
    pub async fn list_objects(&self, lease_id: Uuid) -> OscarResult<Vec<ObjectInfo>> {
        // TODO: Implement listing
        // Query etcd for objects with references from this lease
        todo!("Implementation in DYN-956")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_metadata_serialization() {
        let hash = crate::hash::ObjectHasher::hash(b"test content");
        let metadata = ObjectMetadata::new(hash.clone(), 12);
        
        // Test serialization
        let json_bytes = metadata.to_json_bytes().unwrap();
        
        // Test deserialization
        let deserialized = ObjectMetadata::from_json_bytes(&json_bytes).unwrap();
        
        assert_eq!(metadata, deserialized);
        assert_eq!(deserialized.hash, hash.to_hex());
        assert_eq!(deserialized.size, 12);
    }

    #[test]
    fn test_registration_outcome_values() {
        // Test that our enum variants work as expected
        assert_eq!(RegistrationOutcome::Created, RegistrationOutcome::Created);
        assert_eq!(RegistrationOutcome::Validated, RegistrationOutcome::Validated);
        assert_ne!(RegistrationOutcome::Created, RegistrationOutcome::Validated);
    }

    #[test]
    fn test_object_metadata_creation() {
        let hash = crate::hash::ObjectHasher::hash(b"hello world");
        let metadata = ObjectMetadata::new(hash.clone(), 11);
        
        assert_eq!(metadata.hash, hash.to_hex());
        assert_eq!(metadata.size, 11);
    }

    // Note: Integration tests requiring DistributedRuntime would need etcd
    // and are not included here. They would test:
    // - register_object with CallerContext variations
    // - Atomic create vs validate scenarios  
    // - Error handling for invalid object names
    // - Size limit validation
}