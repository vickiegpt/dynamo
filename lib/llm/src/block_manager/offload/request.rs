use std::cmp::Ordering;
use std::sync::Weak;

use crate::block_manager::block::{BlockMetadata, MutableBlock};
use crate::block_manager::storage::Storage;

#[derive(PartialEq, Eq)]
pub struct OffloadRequestKey {
    pub priority: u64,
    pub timestamp: u64,
}

/// Data needed to offload a block.
/// While the block is in the offload queue, we hold a weak reference to it.
/// This way, we don't prevent the block from being reused if needed.
pub struct OffloadRequest<S: Storage, M: BlockMetadata> {
    pub key: OffloadRequestKey,
    pub block: Weak<MutableBlock<S, M>>,
    pub sequence_hash: u64,
}

impl<S: Storage, M: BlockMetadata> PartialOrd for OffloadRequest<S, M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Order offload requests by priority, high to low.
impl<S: Storage, M: BlockMetadata> Ord for OffloadRequest<S, M> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order high to low.
        match other.key.priority.cmp(&self.key.priority) {
            Ordering::Equal => self.key.timestamp.cmp(&other.key.timestamp),
            ordering => ordering,
        }
    }
}

/// Equality is based on sequence hash, priority, and location.
impl<S: Storage, M: BlockMetadata> PartialEq for OffloadRequest<S, M> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<S: Storage, M: BlockMetadata> Eq for OffloadRequest<S, M> {}
