use super::storage::StorageType;
use crate::kv_router::{
    indexer::RouterEvent,
    protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    },
    KV_EVENT_SUBJECT,
};
use derive_getters::Dissolve;
use dynamo_runtime::traits::events::EventPublisher;
use dynamo_runtime::{
    component::{Component, Namespace},
    raise, Result,
};
use prometheus::local;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::tokens::{SequenceHash, TokenBlock};

pub type WorkerIdentifier = u64;

/// The [EventManager] is not responsible for managing the history of the blocks, nor what
/// events have been published.
///
/// The [EventManager] is only responsible for issuing events on state changes. In this case,
/// there are two states:
///
/// - Store: a dynamo event plane message will be published which defines the registration/storing
///   of the block. Details include, but are not limited to, the sequence/prefix hash, the local block
///   hash, the sequence position of the block, the block size, and the storage location/class which
///   the block is stored in.
///
/// - Remove: a dynamo event plane message will be published which defines the removal of the block
///   from the cache. This messasge will include enough information to identify the block within a
///   storage hierarchy; minmally, the sequence hash and the storage location/class.
///
/// The [RegistrationHandle] associated from [EventManager::block_register] call is an RAII object
/// which will trigger a `Remove` event on being dropped.
pub trait EventManager: Send + Sync + std::fmt::Debug {
    fn register_block(&self, token_block: &TokenBlock) -> Result<RegistrationHandle>;
    fn register_blocks(&self, token_block: &[TokenBlock]) -> Result<Vec<RegistrationHandle>>;
}

trait EventReleaseManager: Send + Sync {
    fn block_release(&self, sequence_hash: SequenceHash);
}

// Implementation notes:
//
// - Removable events are per blocks. I think we will want to leverage a task to collect drop/remove
//   events so that we can batch them together.
//
// - Registration events are can be batched by the nature of the [EventManager::register_blocks] call.

pub struct RegistrationHandle {
    sequence_hash: SequenceHash,
    // memory_type: MemoryType,
    release_manager: Option<Arc<dyn EventReleaseManager>>,
}

impl std::fmt::Debug for RegistrationHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RegistrationHandle {{ sequence_hash: {} }}",
            self.sequence_hash
        )
    }
}

impl Drop for RegistrationHandle {
    fn drop(&mut self) {
        if let Some(release_manager) = self.release_manager.take() {
            release_manager.block_release(self.sequence_hash)
        }
    }
}

enum Event {
    StoreSingle(RegisterBlockEvent),
    StoreMultiple(RegisterBlocksEvent),
    RemoveSingle(SequenceHash),
}

struct RegisterBlockEvent {
    block_hash: LocalBlockHash,
    sequence_hash: ExternalSequenceBlockHash,
    parent_hash: Option<ExternalSequenceBlockHash>,
}

struct RegisterBlocksEvent {
    hashes: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    parent_hash: Option<ExternalSequenceBlockHash>,
}

#[derive(Debug)]
pub struct NullEventManager {}

impl EventManager for NullEventManager {
    fn register_block(&self, token_block: &TokenBlock) -> Result<RegistrationHandle> {
        Ok(RegistrationHandle {
            sequence_hash: token_block.sequence_hash(),
            release_manager: None,
        })
    }

    fn register_blocks(&self, token_block: &[TokenBlock]) -> Result<Vec<RegistrationHandle>> {
        Ok(token_block
            .iter()
            .map(|block| RegistrationHandle {
                sequence_hash: block.sequence_hash(),
                release_manager: None,
            })
            .collect())
    }
}

impl EventReleaseManager for NullEventManager {
    fn block_release(&self, _sequence_hash: SequenceHash) {}
}

pub enum DynamoPublisher {
    Component(Component),
    Namespace(Namespace),
}

impl DynamoPublisher {
    pub async fn publish(&self, event: RouterEvent) -> Result<()> {
        match self {
            DynamoPublisher::Component(component) => {
                component.publish(KV_EVENT_SUBJECT, &event).await
            }
            DynamoPublisher::Namespace(namespace) => {
                namespace.publish(KV_EVENT_SUBJECT, &event).await
            }
        }
    }
}

struct EventChannel {
    tx: mpsc::UnboundedSender<Event>,
}

impl EventReleaseManager for EventChannel {
    // Generalize sequence_hash
    fn block_release(&self, sequence_hash: SequenceHash) {
        self.tx.send(Event::RemoveSingle(sequence_hash));
    }
}

pub struct NatsEventManager {
    event_channel: Arc<EventChannel>,
}

impl NatsEventManager {
    // todo - generalize identifier
    pub async fn new(publisher: DynamoPublisher, worker_identifier: u64) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let state = NatsEventsManagerState {
            rx,
            publisher,
            worker_identifier,
        };

        tokio::spawn(progress_engine(state));

        Self {
            event_channel: Arc::new(EventChannel { tx }),
        }
    }
}

impl std::fmt::Debug for NatsEventManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NatsEventManager")
    }
}

impl EventManager for NatsEventManager {
    fn register_block(&self, token_block: &TokenBlock) -> Result<RegistrationHandle> {
        let event = Event::StoreSingle(RegisterBlockEvent {
            block_hash: LocalBlockHash(token_block.block_hash()),
            sequence_hash: ExternalSequenceBlockHash(token_block.sequence_hash()),
            parent_hash: token_block
                .parent_sequence_hash()
                .map(ExternalSequenceBlockHash),
        });
        if self.event_channel.tx.send(event).is_err() {}
        Ok(RegistrationHandle {
            sequence_hash: token_block.sequence_hash(),
            release_manager: Some(self.event_channel.clone()),
        })
    }

    fn register_blocks(&self, token_blocks: &[TokenBlock]) -> Result<Vec<RegistrationHandle>> {
        unimplemented!()
    }
}

#[derive(Dissolve)]
struct NatsEventsManagerState {
    rx: mpsc::UnboundedReceiver<Event>,
    publisher: DynamoPublisher,
    worker_identifier: WorkerIdentifier,
}

async fn progress_engine(state: NatsEventsManagerState) {
    let (mut rx, publisher, worker_identifier) = state.dissolve();

    let mut event_id = 0;

    while let Some(event) = rx.recv().await {
        match event {
            Event::StoreSingle(event) => {
                let store_data = KvCacheStoreData {
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: event.sequence_hash,
                        tokens_hash: event.block_hash,
                    }],
                    parent_hash: event.parent_hash,
                };
                let data = KvCacheEventData::Stored(store_data);
                let event = KvCacheEvent { event_id, data };
                let event = RouterEvent::new(worker_identifier as i64, event);
                if publisher.publish(event).await.is_err() {
                    tracing::warn!("Failed to publish store event");
                }
            }
            Event::StoreMultiple(event) => {
                let store_data = KvCacheStoreData {
                    blocks: event
                        .hashes
                        .iter()
                        .map(|(local_hash, external_hash)| KvCacheStoredBlockData {
                            block_hash: *external_hash,
                            tokens_hash: *local_hash,
                        })
                        .collect(),
                    parent_hash: event.parent_hash,
                };
                let data = KvCacheEventData::Stored(store_data);
                let event = KvCacheEvent { event_id, data };
                let event = RouterEvent::new(worker_identifier as i64, event);
                if publisher.publish(event).await.is_err() {
                    tracing::warn!("Failed to publish store event");
                }
            }
            Event::RemoveSingle(sequence_hash) => {
                let remove_data = KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(sequence_hash)],
                };
                let data = KvCacheEventData::Removed(remove_data);
                let event = KvCacheEvent { event_id, data };
                let event = RouterEvent::new(worker_identifier as i64, event);
                if publisher.publish(event).await.is_err() {
                    tracing::warn!("Failed to publish remove event");
                }
            }
        }
        event_id += 1;
    }
}
