// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interface to a traditional key-value store such as etcd.
//! "key_value_store" spelt out because in AI land "KV" means something else.

use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::CancellationToken;
use crate::slug::Slug;
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

mod mem;
pub use mem::MemoryStorage;
mod nats;
pub use nats::NATSStorage;
mod etcd;
pub use etcd::EtcdStorage;

/// A key that is safe to use directly in the KV store.
#[derive(Debug, Clone, PartialEq)]
pub struct Key(String);

impl Key {
    pub fn new(s: &str) -> Key {
        Key(Slug::slugify(s).to_string())
    }

    /// Create a Key without changing the string, it is assumed already KV store safe.
    pub fn from_raw(s: String) -> Key {
        Key(s)
    }
}

impl From<&str> for Key {
    fn from(s: &str) -> Key {
        Key::new(s)
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for Key {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&Key> for String {
    fn from(k: &Key) -> String {
        k.0.clone()
    }
}

#[async_trait]
pub trait KeyValueStore: Send + Sync {
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        // auto-delete items older than this
        ttl: Option<Duration>,
    ) -> Result<Box<dyn KeyValueBucket>, StorageError>;

    async fn get_bucket(
        &self,
        bucket_name: &str,
    ) -> Result<Option<Box<dyn KeyValueBucket>>, StorageError>;
}

pub struct KeyValueStoreManager(Box<dyn KeyValueStore>);

impl KeyValueStoreManager {
    pub fn new(s: Box<dyn KeyValueStore>) -> KeyValueStoreManager {
        KeyValueStoreManager(s)
    }

    pub async fn load<T: for<'a> Deserialize<'a>>(
        &self,
        bucket: &str,
        key: &Key,
    ) -> Result<Option<T>, StorageError> {
        let Some(bucket) = self.0.get_bucket(bucket).await? else {
            // No bucket means no cards
            return Ok(None);
        };
        match bucket.get(key).await {
            Ok(Some(card_bytes)) => {
                let card: T = serde_json::from_slice(card_bytes.as_ref())?;
                Ok(Some(card))
            }
            Ok(None) => Ok(None),
            Err(err) => {
                // TODO look at what errors NATS can give us and make more specific wrappers
                Err(StorageError::NATSError(err.to_string()))
            }
        }
    }

    /// Returns a receiver that will receive all the existing keys, and
    /// then block and receive new keys as they are created.
    /// Starts a task that runs forever, watches the store.
    pub fn watch<T: for<'a> Deserialize<'a> + Send + 'static>(
        self: Arc<Self>,
        bucket_name: &str,
        bucket_ttl: Option<Duration>,
    ) -> (
        tokio::task::JoinHandle<Result<(), StorageError>>,
        tokio::sync::mpsc::UnboundedReceiver<T>,
    ) {
        let bucket_name = bucket_name.to_string();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let watch_task = tokio::spawn(async move {
            // Start listening for changes but don't poll this yet
            let bucket = self
                .0
                .get_or_create_bucket(&bucket_name, bucket_ttl)
                .await?;
            let mut stream = bucket.watch().await?;

            // Send all the existing keys
            for (_, card_bytes) in bucket.entries().await? {
                let card: T = serde_json::from_slice(card_bytes.as_ref())?;
                let _ = tx.send(card);
            }

            // Now block waiting for new entries
            while let Some(card_bytes) = stream.next().await {
                let card: T = serde_json::from_slice(card_bytes.as_ref())?;
                let _ = tx.send(card);
            }

            Ok::<(), StorageError>(())
        });
        (watch_task, rx)
    }

    pub async fn publish<T: Serialize + Versioned + Send + Sync>(
        &self,
        bucket_name: &str,
        bucket_ttl: Option<Duration>,
        key: &Key,
        obj: &mut T,
    ) -> anyhow::Result<StorageOutcome> {
        let obj_json = serde_json::to_string(obj)?;
        let bucket = self.0.get_or_create_bucket(bucket_name, bucket_ttl).await?;

        let outcome = bucket.insert(key, &obj_json, obj.revision()).await?;

        match outcome {
            StorageOutcome::Created(revision) | StorageOutcome::Exists(revision) => {
                obj.set_revision(revision);
            }
        }
        Ok(outcome)
    }
}

/// An online storage for key-value config values.
/// Usually backed by `nats-server`.
#[async_trait]
pub trait KeyValueBucket: Send {
    /// A bucket is a collection of key/value pairs.
    /// Insert a value into a bucket, if it doesn't exist already
    async fn insert(
        &self,
        key: &Key,
        value: &str,
        revision: u64,
    ) -> Result<StorageOutcome, StorageError>;

    /// Fetch an item from the key-value storage
    async fn get(&self, key: &Key) -> Result<Option<bytes::Bytes>, StorageError>;

    /// Delete an item from the bucket
    async fn delete(&self, key: &Key) -> Result<(), StorageError>;

    /// A stream of items inserted into the bucket.
    /// Every time the stream is polled it will either return a newly created entry, or block until
    /// such time.
    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = bytes::Bytes> + Send + 'life0>>, StorageError>;

    async fn entries(&self) -> Result<HashMap<String, bytes::Bytes>, StorageError>;
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum StorageOutcome {
    /// The operation succeeded and created a new entry with this revision.
    /// Note that "create" also means update, because each new revision is a "create".
    Created(u64),
    /// The operation did not do anything, the value was already present, with this revision.
    Exists(u64),
}
impl fmt::Display for StorageOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageOutcome::Created(revision) => write!(f, "Created at {revision}"),
            StorageOutcome::Exists(revision) => write!(f, "Exists at {revision}"),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("Could not find bucket '{0}'")]
    MissingBucket(String),

    #[error("Could not find key '{0}'")]
    MissingKey(String),

    #[error("Internal storage error: '{0}'")]
    ProviderError(String),

    #[error("Internal NATS error: {0}")]
    NATSError(String),

    #[error("Internal etcd error: {0}")]
    EtcdError(String),

    #[error("Key Value Error: {0} for bucket '{1}")]
    KeyValueError(String, String),

    #[error("Error decoding bytes: {0}")]
    JSONDecodeError(#[from] serde_json::error::Error),

    #[error("Race condition, retry the call")]
    Retry,
}

/// A trait allowing to get/set a revision on an object.
/// NATS uses this to ensure atomic updates.
pub trait Versioned {
    fn revision(&self) -> u64;
    fn set_revision(&mut self, r: u64);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use futures::{StreamExt, pin_mut};

    const BUCKET_NAME: &str = "mdc";

    /// Convert the value returned by `watch()` into a broadcast stream that multiple
    /// clients can listen to.
    #[allow(dead_code)]
    pub struct TappableStream {
        tx: tokio::sync::broadcast::Sender<bytes::Bytes>,
    }

    #[allow(dead_code)]
    impl TappableStream {
        async fn new<T>(stream: T, max_size: usize) -> Self
        where
            T: futures::Stream<Item = bytes::Bytes> + Send + 'static,
        {
            let (tx, _) = tokio::sync::broadcast::channel(max_size);
            let tx2 = tx.clone();
            tokio::spawn(async move {
                pin_mut!(stream);
                while let Some(x) = stream.next().await {
                    let _ = tx2.send(x);
                }
            });
            TappableStream { tx }
        }

        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<bytes::Bytes> {
            self.tx.subscribe()
        }
    }

    fn init() {
        crate::logging::init();
    }

    #[tokio::test]
    async fn test_memory_storage() -> anyhow::Result<()> {
        init();

        let s = Arc::new(MemoryStorage::new());
        let s2 = Arc::clone(&s);

        let bucket = s.get_or_create_bucket(BUCKET_NAME, None).await?;
        let res = bucket.insert(&"test1".into(), "value1", 0).await?;
        assert_eq!(res, StorageOutcome::Created(0));

        let (got_first_tx, got_first_rx) = tokio::sync::oneshot::channel();
        let ingress = tokio::spawn(async move {
            let b2 = s2.get_or_create_bucket(BUCKET_NAME, None).await?;
            let mut stream = b2.watch().await?;

            // Put in before starting the watch-all
            let v = stream.next().await.unwrap();
            assert_eq!(v, "value1".as_bytes());

            got_first_tx.send(()).unwrap();

            // Put in after
            let v = stream.next().await.unwrap();
            assert_eq!(v, "value2".as_bytes());
            let v = stream.next().await.unwrap();
            assert_eq!(v, "value3".as_bytes());

            Ok::<_, StorageError>(())
        });

        // MemoryStorage uses a HashMap with no inherent ordering, so we must ensure test1 is
        // fetched before test2 is inserted, otherwise they can come out in any order, and we
        // wouldn't be testing the watch behavior.
        got_first_rx.await?;

        let res = bucket.insert(&"test2".into(), "value2", 0).await?;
        assert_eq!(res, StorageOutcome::Created(0));

        // Repeat a key and revision. Ignored.
        let res = bucket.insert(&"test2".into(), "value2", 0).await?;
        assert_eq!(res, StorageOutcome::Exists(0));

        // Increment revision
        let res = bucket.insert(&"test2".into(), "value2", 1).await?;
        assert_eq!(res, StorageOutcome::Created(1));

        let res = bucket.insert(&"test3".into(), "value3", 0).await?;
        assert_eq!(res, StorageOutcome::Created(0));

        // ingress exits once it has received all values
        let _ = ingress.await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_stream() -> anyhow::Result<()> {
        init();

        let s: &'static _ = Box::leak(Box::new(MemoryStorage::new()));
        let bucket: &'static _ =
            Box::leak(Box::new(s.get_or_create_bucket(BUCKET_NAME, None).await?));

        let res = bucket.insert(&"test1".into(), "value1", 0).await?;
        assert_eq!(res, StorageOutcome::Created(0));

        let stream = bucket.watch().await?;
        let tap = TappableStream::new(stream, 10).await;

        let mut rx1 = tap.subscribe();
        let mut rx2 = tap.subscribe();

        let handle1 = tokio::spawn(async move {
            let b = rx1.recv().await.unwrap();
            assert_eq!(b, bytes::Bytes::from(vec![b'G', b'K']));
        });
        let handle2 = tokio::spawn(async move {
            let b = rx2.recv().await.unwrap();
            assert_eq!(b, bytes::Bytes::from(vec![b'G', b'K']));
        });

        bucket.insert(&"test1".into(), "GK", 1).await?;

        let _ = futures::join!(handle1, handle2);
        Ok(())
    }
}
