// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::storage::key_value_store::Key;

use super::{KeyValueBucket, KeyValueStore, StorageError, StorageOutcome};

#[derive(Clone)]
pub struct MemoryStorage {
    inner: Arc<MemoryStorageInner>,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

struct MemoryStorageInner {
    data: Mutex<HashMap<String, MemoryBucket>>,
    change_sender: UnboundedSender<(String, String)>,
    change_receiver: Mutex<UnboundedReceiver<(String, String)>>,
}

pub struct MemoryBucketRef {
    name: String,
    inner: Arc<MemoryStorageInner>,
}

struct MemoryBucket {
    data: HashMap<String, (u64, String)>,
}

impl MemoryBucket {
    fn new() -> Self {
        MemoryBucket {
            data: HashMap::new(),
        }
    }
}

impl MemoryStorage {
    pub fn new() -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        MemoryStorage {
            inner: Arc::new(MemoryStorageInner {
                data: Mutex::new(HashMap::new()),
                change_sender: tx,
                change_receiver: Mutex::new(rx),
            }),
        }
    }
}

#[async_trait]
impl KeyValueStore for MemoryStorage {
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        // MemoryStorage doesn't respect TTL yet
        _ttl: Option<Duration>,
    ) -> Result<Box<dyn KeyValueBucket>, StorageError> {
        let mut locked_data = self.inner.data.lock().await;
        // Ensure the bucket exists
        locked_data
            .entry(bucket_name.to_string())
            .or_insert_with(MemoryBucket::new);
        // Return an object able to access it
        Ok(Box::new(MemoryBucketRef {
            name: bucket_name.to_string(),
            inner: self.inner.clone(),
        }))
    }

    /// This operation cannot fail on MemoryStorage. Always returns Ok.
    async fn get_bucket(
        &self,
        bucket_name: &str,
    ) -> Result<Option<Box<dyn KeyValueBucket>>, StorageError> {
        let locked_data = self.inner.data.lock().await;
        match locked_data.get(bucket_name) {
            Some(_) => Ok(Some(Box::new(MemoryBucketRef {
                name: bucket_name.to_string(),
                inner: self.inner.clone(),
            }))),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl KeyValueBucket for MemoryBucketRef {
    async fn insert(
        &self,
        key: &Key,
        value: &str,
        revision: u64,
    ) -> Result<StorageOutcome, StorageError> {
        let mut locked_data = self.inner.data.lock().await;
        let mut b = locked_data.get_mut(&self.name);
        let Some(bucket) = b.as_mut() else {
            return Err(StorageError::MissingBucket(self.name.to_string()));
        };
        let outcome = match bucket.data.entry(key.to_string()) {
            Entry::Vacant(e) => {
                e.insert((revision, value.to_string()));
                let _ = self
                    .inner
                    .change_sender
                    .send((key.to_string(), value.to_string()));
                StorageOutcome::Created(revision)
            }
            Entry::Occupied(mut entry) => {
                let (rev, _v) = entry.get();
                if *rev == revision {
                    StorageOutcome::Exists(revision)
                } else {
                    entry.insert((revision, value.to_string()));
                    StorageOutcome::Created(revision)
                }
            }
        };
        Ok(outcome)
    }

    async fn get(&self, key: &Key) -> Result<Option<bytes::Bytes>, StorageError> {
        let locked_data = self.inner.data.lock().await;
        let Some(bucket) = locked_data.get(&self.name) else {
            return Ok(None);
        };
        Ok(bucket
            .data
            .get(&key.0)
            .map(|(_, v)| bytes::Bytes::from(v.clone())))
    }

    async fn delete(&self, key: &Key) -> Result<(), StorageError> {
        let mut locked_data = self.inner.data.lock().await;
        let Some(bucket) = locked_data.get_mut(&self.name) else {
            return Err(StorageError::MissingBucket(self.name.to_string()));
        };
        bucket.data.remove(&key.0);
        Ok(())
    }

    /// All current values in the bucket first, then block waiting for new
    /// values to be published.
    /// Caller takes the lock so only a single caller may use this at once.
    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = bytes::Bytes> + Send + 'life0>>, StorageError>
    {
        Ok(Box::pin(async_stream::stream! {
            // All the existing ones first
            let mut seen = HashSet::new();
            let data_lock = self.inner.data.lock().await;
            let Some(bucket) = data_lock.get(&self.name) else {
                tracing::error!(bucket_name = self.name, "watch: Missing bucket");
                return;
            };
            for (_rev, v) in bucket.data.values() {
                seen.insert(v.clone());
                yield bytes::Bytes::from(v.clone());
            }
            drop(data_lock);
            // Now any new ones
            let mut rcv_lock = self.inner.change_receiver.lock().await;
            loop {
                match rcv_lock.recv().await {
                    None => {
                        // Channel is closed, no more values coming
                        break;
                    },
                    Some((_k, v)) => {
                        if seen.contains(&v) {
                            continue;
                        }
                        yield bytes::Bytes::from(v.clone());
                    }
                }
            }
        }))
    }

    async fn entries(&self) -> Result<HashMap<String, bytes::Bytes>, StorageError> {
        let locked_data = self.inner.data.lock().await;
        match locked_data.get(&self.name) {
            Some(bucket) => Ok(bucket
                .data
                .iter()
                .map(|(k, (_rev, v))| (k.to_string(), bytes::Bytes::from(v.clone())))
                .collect()),
            None => Err(StorageError::MissingBucket(self.name.clone())),
        }
    }
}
