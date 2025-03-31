use dynamo_runtime::{
    component::{Component, Namespace},
    raise,
    traits::events::EventPublisher,
    Result,
};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::tokens::{SequenceHash, TokenBlock};

pub trait EventManager {
    fn block_register(&self, token_block: &TokenBlock) -> Result<RegisteredBlock>;
}

enum Event {
    StoreSingle(TokenBlock),
    StoreMultiple(Vec<TokenBlock>),
    RemoveSingle(SequenceHash),
    RemoveMultiple(Vec<SequenceHash>),
}

pub struct RegisteredBlock {
    token_block: TokenBlock,
    event_channel: Option<mpsc::UnboundedSender<Event>>,
}

impl Drop for RegisteredBlock {
    fn drop(&mut self) {
        if let Some(channel) = self.event_channel.take() {
            if channel
                .send(Event::RemoveSingle(self.token_block.sequence_hash()))
                .is_err()
            {
                tracing::warn!("failed to issue remove event for token block; channel closed");
            }
        }
    }
}

pub struct NullEventManager {}

impl EventManager for NullEventManager {
    fn block_register(&self, token_block: &TokenBlock) -> Result<RegisteredBlock> {
        Ok(RegisteredBlock {
            token_block: token_block.clone(),
            event_channel: None,
        })
    }
}

pub struct NatsEventManager {
    publisher: Arc<dyn EventPublisher>,
    event_channel: mpsc::UnboundedSender<Event>,
}

impl EventManager for NatsEventManager {
    fn block_register(&self, token_block: &TokenBlock) -> Result<RegisteredBlock> {
        let event = Event::Store(vec![token_block.clone()]);

        if self.event_channel.send(event).is_err() {
            raise!("failed to send store event for token block; channel closed");
        }

        Ok(RegisteredBlock {
            token_block: token_block.clone(),
            event_channel: Some(self.event_channel.clone()),
        })
    }
}
