// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global block registry for managing block registration handles.

use super::block::{Block, Registered};
use super::{CompleteBlock, SequenceHash};
use crate::block_manager::v2::pools::{BlockMetadata, RegisteredBlock};

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

/// Error types for attachment operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentError {
    /// Attempted to attach a type as unique when it's already registered as multiple
    TypeAlreadyRegisteredAsMultiple(TypeId),
    /// Attempted to attach a type as multiple when it's already registered as unique
    TypeAlreadyRegisteredAsUnique(TypeId),
}

impl std::fmt::Display for AttachmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttachmentError::TypeAlreadyRegisteredAsMultiple(type_id) => {
                write!(
                    f,
                    "Type {:?} is already registered as multiple attachment",
                    type_id
                )
            }
            AttachmentError::TypeAlreadyRegisteredAsUnique(type_id) => {
                write!(
                    f,
                    "Type {:?} is already registered as unique attachment",
                    type_id
                )
            }
        }
    }
}

impl std::error::Error for AttachmentError {}

/// Tracks how a type is registered in the attachment system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachmentMode {
    Unique,
    Multiple,
}

/// Storage for attachments on a BlockRegistrationHandle
#[derive(Debug)]
struct AttachmentStore {
    /// Unique attachments - only one value per TypeId
    unique_attachments: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Multiple attachments - multiple values per TypeId
    multiple_attachments: HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>,
    /// Track which types are registered and how
    type_registry: HashMap<TypeId, AttachmentMode>,
    /// Storage for weak block references - separate from generic attachments, keyed by TypeId
    weak_blocks: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl AttachmentStore {
    fn new() -> Self {
        Self {
            unique_attachments: HashMap::new(),
            multiple_attachments: HashMap::new(),
            type_registry: HashMap::new(),
            weak_blocks: HashMap::new(),
        }
    }
}

/// Typed accessor for attachments of a specific type
pub struct TypedAttachments<'a, T> {
    handle: &'a BlockRegistrationHandle,
    _phantom: PhantomData<T>,
}

/// Handle that represents a block registration in the global registry.
/// This handle is cloneable and can be shared across pools.
#[derive(Clone, Debug)]
pub struct BlockRegistrationHandle {
    inner: Arc<BlockRegistrationHandleInner>,
}

struct WeakBlock<T: BlockMetadata + Sync> {
    raw_block: Weak<Block<T, Registered>>,
    reg_block: Weak<super::PrimaryBlock<T>>,
}

#[derive(Debug)]
struct BlockRegistrationHandleInner {
    /// Sequence hash of the block
    seq_hash: SequenceHash,
    /// Attachments for the block
    attachments: Mutex<AttachmentStore>,
    /// Weak reference to the registry - allows us to remove the block from the registry on drop
    registry: Weak<Mutex<RegistryState>>,
}

impl Drop for BlockRegistrationHandleInner {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            let mut state = registry.lock();
            state.canonical_blocks.remove(&self.seq_hash);
        }
    }
}

impl BlockRegistrationHandle {
    pub fn seq_hash(&self) -> SequenceHash {
        self.inner.seq_hash
    }

    /// Get a typed accessor for attachments of type T
    pub fn get<T: Any + Send + Sync>(&self) -> TypedAttachments<'_, T> {
        TypedAttachments {
            handle: self,
            _phantom: PhantomData,
        }
    }

    /// Attach a unique value of type T to this handle.
    /// Only one value per type is allowed - subsequent calls will replace the previous value.
    /// Returns an error if type T is already registered as multiple attachment.
    pub fn attach_unique<T: Any + Send + Sync>(&self, value: T) -> Result<(), AttachmentError> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();

        // Check if this type is already registered as multiple
        if let Some(AttachmentMode::Multiple) = attachments.type_registry.get(&type_id) {
            return Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(type_id));
        }

        // Register/update as unique
        attachments
            .unique_attachments
            .insert(type_id, Box::new(value));
        attachments
            .type_registry
            .insert(type_id, AttachmentMode::Unique);

        Ok(())
    }

    /// Attach a value of type T to this handle.
    /// Multiple values per type are allowed - this will append to existing values.
    /// Returns an error if type T is already registered as unique attachment.
    pub fn attach<T: Any + Send + Sync>(&self, value: T) -> Result<(), AttachmentError> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();

        // Check if this type is already registered as unique
        if let Some(AttachmentMode::Unique) = attachments.type_registry.get(&type_id) {
            return Err(AttachmentError::TypeAlreadyRegisteredAsUnique(type_id));
        }

        // Register/update as multiple
        attachments
            .multiple_attachments
            .entry(type_id)
            .or_insert_with(Vec::new)
            .push(Box::new(value));
        attachments
            .type_registry
            .insert(type_id, AttachmentMode::Multiple);

        Ok(())
    }

    pub(crate) fn attach_block<T: BlockMetadata + Sync>(
        &self,
        block: super::PrimaryBlock<T>,
    ) -> super::ImmutableBlock<T> {
        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
        let mut attachments = self.inner.attachments.lock();

        if let Some(weak_any) = attachments.weak_blocks.get(&type_id) {
            if let Some(weak) = weak_any.downcast_ref::<WeakBlock<T>>() {
                debug_assert!(
                    weak.raw_block.upgrade().is_none(),
                    "Attempted to reattach block when raw block is still alive"
                );
                debug_assert!(
                    weak.reg_block.upgrade().is_none(),
                    "Attempted to reattach block when registered block is still alive"
                );
            }
        }

        let raw_weak = Arc::downgrade(block.block.as_ref().unwrap());
        let reg_arc = Arc::new(block);
        let reg_weak = Arc::downgrade(&reg_arc);

        attachments.weak_blocks.insert(
            type_id,
            Box::new(WeakBlock {
                raw_block: raw_weak,
                reg_block: reg_weak,
            }),
        );

        super::ImmutableBlock { block: reg_arc }
    }

    pub(crate) fn register_block<T: BlockMetadata + Sync>(
        &self,
        mut block: CompleteBlock<T>,
        duplication_policy: super::BlockDuplicationPolicy,
        pool_return_fn: Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>,
    ) -> super::ImmutableBlock<T> {
        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
        let block_id = block.block_id();

        // Take ownership of the inner block
        let inner_block = block.block.take().unwrap();
        let reset_return_fn = block.return_fn.clone();

        // Register the block to get it in Registered state
        let registered_block = inner_block.register(self.clone());

        let mut attachments = self.inner.attachments.lock();

        // Check for existing blocks with same sequence hash
        if let Some(weak_any) = attachments.weak_blocks.get(&type_id) {
            if let Some(weak_block) = weak_any.downcast_ref::<WeakBlock<T>>() {
                // Try to get the existing primary block
                if let Some(existing_primary) = weak_block.reg_block.upgrade() {
                    // Check if same block_id (shouldn't happen)
                    if existing_primary.block_id() == block_id {
                        panic!("Attempted to register block with same block_id as existing");
                    }

                    // Handle duplicate based on policy
                    match duplication_policy {
                        super::BlockDuplicationPolicy::Allow => {
                            // Create DuplicateBlock referencing the primary
                            let duplicate = super::DuplicateBlock::new(
                                registered_block,
                                existing_primary.clone(),
                                reset_return_fn,
                            );
                            return super::ImmutableBlock {
                                block: Arc::new(duplicate),
                            };
                        }
                        super::BlockDuplicationPolicy::Reject => {
                            // Return existing primary, discard new block
                            // The registered_block will be dropped and eventually returned to reset pool
                            return super::ImmutableBlock {
                                block: existing_primary,
                            };
                        }
                    }
                }

                // Primary couldn't be upgraded but raw block might exist
                // This is an edge case - for now, treat as creating a new primary
            }
        }

        // No existing block or couldn't upgrade - create new primary
        let primary = super::PrimaryBlock::new(Arc::new(registered_block), pool_return_fn);

        // Store weak references for future lookups
        let primary_arc = Arc::new(primary);
        let raw_weak = Arc::downgrade(primary_arc.block.as_ref().unwrap());
        let reg_weak = Arc::downgrade(&primary_arc);

        attachments.weak_blocks.insert(
            type_id,
            Box::new(WeakBlock {
                raw_block: raw_weak,
                reg_block: reg_weak,
            }),
        );

        drop(attachments); // Release lock

        super::ImmutableBlock {
            block: primary_arc as Arc<dyn super::RegisteredBlock<T>>,
        }
    }

    pub(crate) fn try_get_block<T: BlockMetadata + Sync>(
        &self,
        pool_return_fn: Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>,
    ) -> Option<super::ImmutableBlock<T>> {
        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
        let mut attachments = self.inner.attachments.lock();

        let weak_block = attachments
            .weak_blocks
            .get(&type_id)
            .and_then(|weak_any| weak_any.downcast_ref::<WeakBlock<T>>())?;

        // Try to upgrade the PrimaryBlock first
        if let Some(primary_arc) = weak_block.reg_block.upgrade() {
            return Some(super::ImmutableBlock {
                block: primary_arc as Arc<dyn super::RegisteredBlock<T>>,
            });
        }

        // Try to upgrade the raw Block<T, Registered>
        if let Some(raw_arc) = weak_block.raw_block.upgrade() {
            // Create new PrimaryBlock from the raw block
            let primary = super::PrimaryBlock::new(raw_arc, pool_return_fn);
            let primary_arc = Arc::new(primary);

            // Update the weak reference
            let new_weak = Arc::downgrade(&primary_arc);
            let weak_block_mut = WeakBlock {
                raw_block: weak_block.raw_block.clone(),
                reg_block: new_weak,
            };
            attachments
                .weak_blocks
                .insert(type_id, Box::new(weak_block_mut));

            return Some(super::ImmutableBlock {
                block: primary_arc as Arc<dyn super::RegisteredBlock<T>>,
            });
        }

        None
    }

    // /// Attach a weak reference to a registered block.
    // /// If a weak reference already exists for this type, asserts that its strong count is 0 before replacing it.
    // pub(crate) fn attach_weak_block<T: BlockMetadata + Sync>(
    //     &self,
    //     weak: Weak<Block<T, Registered>>,
    // ) {
    //     let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
    //     let mut attachments = self.inner.attachments.lock();

    //     // Check if we already have a weak reference for this type
    //     #[cfg(debug_assertions)]
    //     {
    //         if let Some(existing_weak_any) = attachments.weak_blocks.get(&type_id) {
    //             if let Some(existing_weak) =
    //                 existing_weak_any.downcast_ref::<Weak<Block<T, Registered>>>()
    //             {
    //                 // Assert that the existing weak reference has no strong references
    //                 assert_eq!(
    //                     existing_weak.strong_count(),
    //                     0,
    //                     "Attempted to attach weak block reference when existing weak reference still has strong references"
    //                 );
    //             }
    //         }
    //     }

    //     // Store the new weak reference keyed by type
    //     attachments.weak_blocks.insert(type_id, Box::new(weak));
    // }

    // /// Retrieve and upgrade the stored weak block reference for type T.
    // /// Returns None if no weak reference is stored for this type or if the upgrade fails.
    // pub(crate) fn try_get_block<T: BlockMetadata + Sync>(
    //     &self,
    // ) -> Option<Arc<Block<T, Registered>>> {
    //     let type_id = TypeId::of::<Weak<Block<T, Registered>>>();
    //     let attachments = self.inner.attachments.lock();

    //     if let Some(weak_any) = attachments.weak_blocks.get(&type_id) {
    //         if let Some(weak) = weak_any.downcast_ref::<Weak<Block<T, Registered>>>() {
    //             return weak.upgrade();
    //         }
    //     }

    //     None
    // }
}

impl<'a, T: Any + Send + Sync> TypedAttachments<'a, T> {
    /// Execute a closure with immutable access to the unique attachment of type T.
    pub fn with_unique<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let type_id = TypeId::of::<T>();
        let attachments = self.handle.inner.attachments.lock();
        attachments
            .unique_attachments
            .get(&type_id)?
            .downcast_ref::<T>()
            .map(f)
    }

    /// Execute a closure with mutable access to the unique attachment of type T.
    pub fn with_unique_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.handle.inner.attachments.lock();
        attachments
            .unique_attachments
            .get_mut(&type_id)?
            .downcast_mut::<T>()
            .map(f)
    }

    /// Execute a closure with immutable access to multiple attachments of type T.
    pub fn with_multiple<R>(&self, f: impl FnOnce(&[&T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let attachments = self.handle.inner.attachments.lock();

        let multiple_refs: Vec<&T> = attachments
            .multiple_attachments
            .get(&type_id)
            .map(|vec| vec.iter().filter_map(|v| v.downcast_ref::<T>()).collect())
            .unwrap_or_default();

        f(&multiple_refs)
    }

    /// Execute a closure with mutable access to multiple attachments of type T.
    pub fn with_multiple_mut<R>(&self, f: impl FnOnce(&mut [&mut T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.handle.inner.attachments.lock();

        let mut multiple_refs: Vec<&mut T> = attachments
            .multiple_attachments
            .get_mut(&type_id)
            .map(|vec| {
                vec.iter_mut()
                    .filter_map(|v| v.downcast_mut::<T>())
                    .collect()
            })
            .unwrap_or_default();

        f(&mut multiple_refs)
    }

    /// Execute a closure with immutable access to both unique and multiple attachments of type T.
    pub fn with_all<R>(&self, f: impl FnOnce(Option<&T>, &[&T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let attachments = self.handle.inner.attachments.lock();

        let unique = attachments
            .unique_attachments
            .get(&type_id)
            .and_then(|v| v.downcast_ref::<T>());

        let multiple_refs: Vec<&T> = attachments
            .multiple_attachments
            .get(&type_id)
            .map(|vec| vec.iter().filter_map(|v| v.downcast_ref::<T>()).collect())
            .unwrap_or_default();

        f(unique, &multiple_refs)
    }

    /// Execute a closure with mutable access to both unique and multiple attachments of type T.
    pub fn with_all_mut<R>(&self, f: impl FnOnce(Option<&mut T>, &mut [&mut T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.handle.inner.attachments.lock();

        // Check where this type is registered to avoid double mutable borrow
        match attachments.type_registry.get(&type_id) {
            Some(AttachmentMode::Unique) => {
                // Type is registered as unique - get mutable reference to unique only
                let unique = attachments
                    .unique_attachments
                    .get_mut(&type_id)
                    .and_then(|v| v.downcast_mut::<T>());
                let mut empty_vec: Vec<&mut T> = Vec::new();
                f(unique, &mut empty_vec)
            }
            Some(AttachmentMode::Multiple) => {
                // Type is registered as multiple - get mutable references to multiple only
                let mut multiple_refs: Vec<&mut T> = attachments
                    .multiple_attachments
                    .get_mut(&type_id)
                    .map(|vec| {
                        vec.iter_mut()
                            .filter_map(|v| v.downcast_mut::<T>())
                            .collect()
                    })
                    .unwrap_or_default();
                f(None, &mut multiple_refs)
            }
            None => {
                // Type not registered at all
                let mut empty_vec: Vec<&mut T> = Vec::new();
                f(None, &mut empty_vec)
            }
        }
    }
}

/// Global registry for managing block registrations.
/// Tracks canonical blocks and provides registration handles.
#[derive(Debug, Clone)]
pub struct BlockRegistry {
    state: Arc<Mutex<RegistryState>>,
}

#[derive(Debug)]
struct RegistryState {
    canonical_blocks: HashMap<SequenceHash, Weak<BlockRegistrationHandleInner>>,
}

impl BlockRegistry {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(RegistryState {
                canonical_blocks: HashMap::new(),
            })),
        }
    }

    /// Register a sequence hash and get a registration handle.
    /// If the sequence is already registered, returns the existing handle.
    /// Otherwise, creates a new canonical registration.
    pub fn register_sequence_hash(&self, seq_hash: SequenceHash) -> BlockRegistrationHandle {
        let mut state = self.state.lock();

        // Check if we already have a canonical block for this sequence hash
        if let Some(weak_handle) = state.canonical_blocks.get(&seq_hash) {
            if let Some(existing_handle) = weak_handle.upgrade() {
                // Return a clone of the existing canonical handle
                return BlockRegistrationHandle {
                    inner: existing_handle,
                };
            }
        }

        // Create a new canonical registration
        let inner = Arc::new(BlockRegistrationHandleInner {
            seq_hash,
            registry: Arc::downgrade(&self.state),
            attachments: Mutex::new(AttachmentStore::new()),
        });

        state
            .canonical_blocks
            .insert(seq_hash, Arc::downgrade(&inner));

        BlockRegistrationHandle { inner }
    }

    /// Match a sequence hash and return a registration handle.
    pub fn match_sequence_hash(&self, seq_hash: SequenceHash) -> Option<BlockRegistrationHandle> {
        let state = self.state.lock();
        state
            .canonical_blocks
            .get(&seq_hash)
            .and_then(|weak| weak.upgrade())
            .map(|inner| BlockRegistrationHandle { inner })
    }

    /// Check if a sequence is currently registered (has a canonical handle).
    pub fn is_registered(&self, seq_hash: SequenceHash) -> bool {
        let state = self.state.lock();
        state
            .canonical_blocks
            .get(&seq_hash)
            .map(|weak| weak.strong_count() > 0)
            .unwrap_or(false)
    }

    /// Get the current number of registered blocks.
    pub fn registered_count(&self) -> usize {
        let state = self.state.lock();
        state.canonical_blocks.len()
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_new_sequence() {
        let registry = BlockRegistry::new();
        let seq_hash = 42;
        let handle = registry.register_sequence_hash(seq_hash);

        assert_eq!(handle.seq_hash(), seq_hash);
        assert!(registry.is_registered(seq_hash));
        assert_eq!(registry.registered_count(), 1);
    }

    #[test]
    fn test_register_existing_sequence_returns_same_handle() {
        let registry = BlockRegistry::new();
        let seq_hash = 42;
        let handle1 = registry.register_sequence_hash(seq_hash);
        let handle2 = registry.register_sequence_hash(seq_hash);

        assert_eq!(handle1.seq_hash(), handle2.seq_hash());
        assert_eq!(registry.registered_count(), 1);
    }

    #[test]
    fn test_handle_drop_removes_registration() {
        let registry = BlockRegistry::new();
        let seq_hash = 42;
        {
            let _handle = registry.register_sequence_hash(seq_hash);
            assert!(registry.is_registered(seq_hash));
            assert_eq!(registry.registered_count(), 1);
        }

        // Handle should be dropped and registration removed
        assert!(!registry.is_registered(seq_hash));
        assert_eq!(registry.registered_count(), 0);
    }

    #[test]
    fn test_multiple_handles_same_sequence() {
        let registry = BlockRegistry::new();
        let seq_hash = 42;
        let handle1 = registry.register_sequence_hash(seq_hash);
        let handle2 = handle1.clone();

        drop(handle1);

        // Sequence should still be registered because handle2 exists
        assert!(registry.is_registered(seq_hash));
        assert_eq!(registry.registered_count(), 1);

        drop(handle2);

        // Now sequence should be unregistered
        assert!(!registry.is_registered(seq_hash));
        assert_eq!(registry.registered_count(), 0);
    }

    #[test]
    fn test_attach_unique() {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(42);

        // Attach a unique value
        handle.attach_unique("test_publisher".to_string()).unwrap();

        // Retrieve the value using the new API
        let value = handle.get::<String>().with_unique(|s| s.clone());
        assert_eq!(value, Some("test_publisher".to_string()));

        // Replace with a new value (should succeed)
        handle.attach_unique("new_publisher".to_string()).unwrap();
        let value = handle.get::<String>().with_unique(|s| s.clone());
        assert_eq!(value, Some("new_publisher".to_string()));
    }

    #[test]
    fn test_attach_multiple() {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(42);

        // Attach multiple values
        handle.attach("listener1".to_string()).unwrap();
        handle.attach("listener2".to_string()).unwrap();
        handle.attach("listener3".to_string()).unwrap();

        // Retrieve all values using the new API
        let listeners = handle
            .get::<String>()
            .with_multiple(|listeners| listeners.iter().map(|s| (*s).clone()).collect::<Vec<_>>());
        assert_eq!(listeners.len(), 3);
        assert!(listeners.contains(&"listener1".to_string()));
        assert!(listeners.contains(&"listener2".to_string()));
        assert!(listeners.contains(&"listener3".to_string()));

        // Also test with_all
        handle.get::<String>().with_all(|unique, multiple| {
            assert_eq!(unique, None);
            assert_eq!(multiple.len(), 3);
        });
    }

    #[test]
    fn test_type_tracking_enforcement() {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(42);

        // Test: attach unique first, then try multiple (should fail)
        handle
            .attach_unique("unique_publisher".to_string())
            .unwrap();

        let result = handle.attach("listener1".to_string());
        assert_eq!(
            result,
            Err(AttachmentError::TypeAlreadyRegisteredAsUnique(
                TypeId::of::<String>()
            ))
        );

        // Test with different types: attach multiple first, then try unique (should fail)
        handle.attach(42i32).unwrap();
        handle.attach(43i32).unwrap();

        let result = handle.attach_unique(44i32);
        assert_eq!(
            result,
            Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(
                TypeId::of::<i32>()
            ))
        );
    }

    #[test]
    fn test_with_unique_closure() {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(42);

        handle.attach_unique(42i32).unwrap();

        // Test with_unique closure using new API
        let result = handle.get::<i32>().with_unique(|value| *value * 2);
        assert_eq!(result, Some(84));

        // Test with non-existent type
        let result = handle.get::<u64>().with_unique(|value| *value * 2);
        assert_eq!(result, None);
    }

    #[test]
    fn test_with_all_closure() {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(42);

        // Use different types since we can't mix unique and multiple for same type
        handle.attach_unique(100i32).unwrap(); // unique i32
        handle.attach(1i64).unwrap(); // multiple i64
        handle.attach(2i64).unwrap();
        handle.attach(3i64).unwrap();

        // Test with_all closure for i32 (should have unique only) using new API
        let result = handle.get::<i32>().with_all(|unique, multiple| {
            let unique_sum = unique.unwrap_or(&0);
            let multiple_sum: i32 = multiple.iter().map(|&&x| x).sum();
            unique_sum + multiple_sum
        });
        assert_eq!(result, 100); // Only unique value

        // Test with_all closure for i64 (should have multiple only) using new API
        let result = handle.get::<i64>().with_all(|unique, multiple| {
            let unique_sum = unique.unwrap_or(&0);
            let multiple_sum: i64 = multiple.iter().map(|&&x| x).sum();
            unique_sum + multiple_sum
        });
        assert_eq!(result, 6); // 1 + 2 + 3

        // Test with non-existent type using new API
        let result = handle.get::<u64>().with_all(|unique, multiple| {
            let unique_sum = unique.unwrap_or(&0);
            let multiple_sum: u64 = multiple.iter().map(|&&x| x).sum();
            unique_sum + multiple_sum
        });
        assert_eq!(result, 0);
    }

    #[test]
    fn test_different_types_usage() {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(42);

        // Define some test types for better demonstration
        #[derive(Debug, Clone, PartialEq)]
        struct EventPublisher(String);

        #[derive(Debug, Clone, PartialEq)]
        struct EventListener(String);

        // Attach unique EventPublisher
        handle
            .attach_unique(EventPublisher("main_publisher".to_string()))
            .unwrap();

        // Attach multiple EventListeners
        handle
            .attach(EventListener("listener1".to_string()))
            .unwrap();
        handle
            .attach(EventListener("listener2".to_string()))
            .unwrap();

        // Retrieve by type using new API
        let publisher = handle.get::<EventPublisher>().with_unique(|p| p.clone());
        assert_eq!(
            publisher,
            Some(EventPublisher("main_publisher".to_string()))
        );

        let listeners = handle
            .get::<EventListener>()
            .with_multiple(|listeners| listeners.iter().map(|l| (*l).clone()).collect::<Vec<_>>());
        assert_eq!(listeners.len(), 2);
        assert!(listeners.contains(&EventListener("listener1".to_string())));
        assert!(listeners.contains(&EventListener("listener2".to_string())));

        // Test with_all for EventListener (should have no unique, only multiple)
        handle.get::<EventListener>().with_all(|unique, multiple| {
            assert_eq!(unique, None);
            assert_eq!(multiple.len(), 2);
        });

        // Attempting to register EventPublisher as multiple should fail
        let result = handle.attach(EventPublisher("another_publisher".to_string()));
        assert_eq!(
            result,
            Err(AttachmentError::TypeAlreadyRegisteredAsUnique(
                TypeId::of::<EventPublisher>()
            ))
        );

        // Attempting to register EventListener as unique should fail
        let result = handle.attach_unique(EventListener("unique_listener".to_string()));
        assert_eq!(
            result,
            Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(
                TypeId::of::<EventListener>()
            ))
        );
    }

    #[test]
    fn test_mutable_access() {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(42);

        #[derive(Debug, Clone, PartialEq)]
        struct UniqueCounter(i32);

        #[derive(Debug, Clone, PartialEq)]
        struct MultipleCounter(i32);

        impl UniqueCounter {
            fn increment(&mut self) {
                self.0 += 1;
            }
        }

        impl MultipleCounter {
            fn increment(&mut self) {
                self.0 += 1;
            }
        }

        // Test unique mutable access
        handle.attach_unique(UniqueCounter(0)).unwrap();

        handle.get::<UniqueCounter>().with_unique_mut(|counter| {
            counter.increment();
            counter.increment();
        });

        // Verify the change
        let value = handle
            .get::<UniqueCounter>()
            .with_unique(|counter| counter.0);
        assert_eq!(value, Some(2));

        // Test mutable access to multiple (different type)
        handle.attach(MultipleCounter(10)).unwrap();
        handle.attach(MultipleCounter(20)).unwrap();

        handle
            .get::<MultipleCounter>()
            .with_multiple_mut(|counters| {
                for counter in counters {
                    counter.increment();
                }
            });

        // Verify multiple were modified
        let total = handle
            .get::<MultipleCounter>()
            .with_multiple(|counters| counters.iter().map(|c| c.0).sum::<i32>());
        assert_eq!(total, 32); // 11 + 21
    }

    #[test]
    fn test_with_all_mut_unique() {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(42);

        #[derive(Debug, Clone, PartialEq)]
        struct UniqueValue(i32);

        impl UniqueValue {
            fn increment(&mut self) {
                self.0 += 1;
            }
        }

        // Attach unique value
        handle.attach_unique(UniqueValue(10)).unwrap();

        // Test with_all_mut for unique type
        handle
            .get::<UniqueValue>()
            .with_all_mut(|unique, multiple| {
                assert!(unique.is_some());
                assert_eq!(multiple.len(), 0);
                if let Some(val) = unique {
                    val.increment();
                }
            });

        // Verify the change
        let value = handle.get::<UniqueValue>().with_unique(|v| v.0);
        assert_eq!(value, Some(11));
    }

    #[test]
    fn test_with_all_mut_multiple() {
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(42);

        #[derive(Debug, Clone, PartialEq)]
        struct MultipleValue(i32);

        impl MultipleValue {
            fn increment(&mut self) {
                self.0 += 1;
            }
        }

        // Attach multiple values
        handle.attach(MultipleValue(1)).unwrap();
        handle.attach(MultipleValue(2)).unwrap();

        // Test with_all_mut for multiple type
        handle
            .get::<MultipleValue>()
            .with_all_mut(|unique, multiple| {
                assert!(unique.is_none());
                assert_eq!(multiple.len(), 2);
                for val in multiple {
                    val.increment();
                }
            });

        // Verify the changes
        let total = handle
            .get::<MultipleValue>()
            .with_multiple(|values| values.iter().map(|v| v.0).sum::<i32>());
        assert_eq!(total, 5); // 2 + 3
    }
}
