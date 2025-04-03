// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # KV Block Available Pool
//!
//! The Available Pool manages KV blocks that are not actively in use but retain their previous state.
//!
//! ## Key Features:
//!
//! - **State Preservation**: Blocks in the pool maintain their previous state and can be reused.
//!
//! - **Priority-Based FIFO**: Blocks are returned in first-in, first-out order within their priority levels.
//!   Lower priority values are processed first, allowing important blocks to be retained longer.
//!
//! - **State Matching**: Blocks can be matched against their previous state instead of being taken randomly,
//!   enabling efficient reuse of blocks with specific sequence hashes.
//!
//! - **Priority Management**: Priorities can be applied to blocks based on their sequence hash,
//!   requiring some external knowledge of the block's characteristics.
//!
//! - **State Management**: Blocks can have their states wiped clean/reset individually or in groups.
//!   The entire pool can also be reset as needed.
//!
//! - **Synchronization**: Fence operations ensure all higher priority operations have completed
//!   before proceeding. Note that this is not a true fence - higher priority operations issued
//!   after the fence will still be processed before the fence completes.

use derive_getters::Dissolve;
use dynamo_runtime::{
    raise,
    utils::pool::{PoolExt, PoolItem, PoolValue, ReturnHandle},
    Result,
};
use std::{collections::BTreeSet, collections::HashMap, collections::VecDeque, sync::Arc};
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::JoinHandle,
};

use crate::{
    block_manager::block::BlockState,
    tokens::{SequenceHash, TokenBlock},
};

use super::block::{Block, BlockMetadata};
use super::storage::Storage;

pub type UniqueBlock<S: Storage, M: BlockMetadata> = PoolItem<Block<S, M>>;

pub struct BlockPool<T: Storage, M: BlockMetadata> {
    match_tx: mpsc::UnboundedSender<MatchRequest<T, M>>,
    control_tx: mpsc::UnboundedSender<ControlRequest<T, M>>,
    fence_tx: mpsc::UnboundedSender<oneshot::Sender<()>>,
    return_handle: Arc<ReturnHandleImpl<T, M>>,
    total_blocks_rx: watch::Receiver<u64>,
    available_blocks_rx: watch::Receiver<u64>,
    join_handle: JoinHandle<()>,
}

impl<T: Storage, M: BlockMetadata> BlockPool<T, M> {
    pub async fn new() -> Self {
        let (match_tx, match_rx) = mpsc::unbounded_channel();
        let (return_tx, return_rx) = mpsc::unbounded_channel();
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        let (fence_tx, fence_rx) = mpsc::unbounded_channel();

        let return_tx_clone = return_tx.clone();
        let return_handle = Arc::new(ReturnHandleImpl {
            return_tx: return_tx_clone,
        });

        let state = BlockPoolInner::new();

        let total_blocks_rx = state.total_blocks_watcher();
        let available_blocks_rx = state.available_blocks_watcher();

        let join_handle = tokio::spawn(progress_engine(
            match_rx, return_rx, control_rx, fence_rx, state,
        ));

        Self {
            match_tx,
            control_tx,
            fence_tx,
            return_handle,
            total_blocks_rx,
            available_blocks_rx,
            join_handle,
        }
    }

    pub fn total_blocks(&self) -> u64 {
        *self.total_blocks_rx.borrow()
    }

    pub fn available_blocks(&self) -> u64 {
        *self.available_blocks_rx.borrow()
    }

    pub fn total_blocks_watch(&self) -> watch::Receiver<u64> {
        self.total_blocks_rx.clone()
    }

    pub fn available_blocks_watch(&self) -> watch::Receiver<u64> {
        self.available_blocks_rx.clone()
    }

    pub fn is_active(&self) -> bool {
        !self.join_handle.is_finished()
    }

    pub async fn match_blocks(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> Result<Vec<PoolItem<Block<T, M>>>> {
        let (tx, rx) = oneshot::channel();
        if self
            .match_tx
            .send(MatchRequest::MatchMultiple(MatchMultiple {
                hashes,
                return_handle: self.return_handle.clone(),
                tx,
            }))
            .is_err()
        {
            raise!("failed to send match request; channel closed");
        }

        let matched_blocks = rx.await?;
        Ok(matched_blocks)
    }

    pub async fn match_token_blocks(
        &self,
        token_blocks: &[TokenBlock],
    ) -> Result<Vec<PoolItem<Block<T, M>>>> {
        let hashes: Vec<u64> = token_blocks.iter().map(|b| b.sequence_hash()).collect();
        self.match_blocks(hashes).await
    }

    pub async fn take_blocks(&self, count: u32) -> Result<Vec<PoolItem<Block<T, M>>>> {
        let (tx, rx) = oneshot::channel();
        if self
            .match_tx
            .send(MatchRequest::Take(Take {
                count,
                return_handle: self.return_handle.clone(),
                tx,
            }))
            .is_err()
        {
            raise!("failed to send take request; channel closed");
        }

        let matched_blocks = rx.await?;
        Ok(matched_blocks)
    }

    pub async fn insert(&self, block: Block<T, M>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::Insert(InsertControl { block, tx }))
            .is_err()
        {
            raise!("failed to send insert request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn update_single(&self, update: UpdateBlock<M>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::UpdateSingle(UpdateSingleControl {
                update,
                tx,
            }))
            .is_err()
        {
            raise!("failed to send update single request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn update_multiple(&self, updates: Vec<UpdateBlock<M>>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::UpdateMultiple(UpdateMultipleControl {
                updates,
                tx,
            }))
            .is_err()
        {
            raise!("failed to send update multiple request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn reset(&self, sequence_hashes: Vec<SequenceHash>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::Reset(ResetControl {
                sequence_hashes,
                tx,
                _phantom: std::marker::PhantomData,
            }))
            .is_err()
        {
            raise!("failed to send reset request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn reset_all(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::ResetAll(ResetAllControl {
                tx,
                _phantom: std::marker::PhantomData,
            }))
            .is_err()
        {
            raise!("failed to send reset all request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn fence(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self.fence_tx.send(tx).is_err() {
            raise!("failed to send fence request; channel closed");
        }
        rx.await?;
        Ok(())
    }
}

struct ReturnHandleImpl<T: Storage + 'static, M: BlockMetadata> {
    return_tx: mpsc::UnboundedSender<PoolValue<Block<T, M>>>,
}

impl<T: Storage, M: BlockMetadata> ReturnHandle<Block<T, M>> for ReturnHandleImpl<T, M> {
    fn return_to_pool(&self, value: PoolValue<Block<T, M>>) {
        if self.return_tx.send(value).is_err() {
            tracing::trace!("Failed to return block to pool");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PriorityKey<M: BlockMetadata> {
    metadata: M,
    sequence_hash: SequenceHash,
}

impl<M: BlockMetadata> PriorityKey<M> {
    fn new(metadata: M, sequence_hash: SequenceHash) -> Self {
        Self {
            metadata,
            sequence_hash,
        }
    }

    fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }

    fn metadata(&self) -> &M {
        &self.metadata
    }

    fn update_metadata(&mut self, metadata: M) {
        self.metadata = metadata;
    }
}

// customize ord and partial ord for to store first by priority (lowest to highest), then by return_tick (lowest to highest)
impl<M: BlockMetadata> PartialOrd for PriorityKey<M> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<M: BlockMetadata> Ord for PriorityKey<M> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.metadata
            .cmp(&other.metadata)
            .then(self.sequence_hash.cmp(&other.sequence_hash))
    }
}

// impl<T: Storage, M: BlockMetadata> From<&Block<T, M>> for PriorityKey<M> {
//     fn from(block: &Block<T, M>) -> Result<Self, BlockError> {
//         Self {
//             metadata: block.metadata().clone(),
//             sequence_hash: block.sequence_hash(),
//         }
//     }
// }

#[derive(Default)]
pub struct BlockPoolInner<T: Storage + 'static, M: BlockMetadata> {
    // Direct lookup by sequence_hash
    lookup_map: HashMap<SequenceHash, PoolValue<Block<T, M>>>,

    // Ordered by timestamp (oldest first)
    priority_set: BTreeSet<PriorityKey<M>>,

    // Fully Uninitialized
    uninitialized_set: VecDeque<PoolValue<Block<T, M>>>,

    // Return Tick
    return_tick: u64,

    // Total blocks
    total_blocks_tx: watch::Sender<u64>,

    // Available blocks
    available_blocks_tx: watch::Sender<u64>,
}

impl<T: Storage, M: BlockMetadata> BlockPoolInner<T, M> {
    fn new() -> Self {
        let (total_blocks_tx, _) = watch::channel(0);
        let (available_blocks_tx, _) = watch::channel(0);

        Self {
            lookup_map: HashMap::new(),
            priority_set: BTreeSet::new(),
            uninitialized_set: VecDeque::new(),
            return_tick: 0,
            total_blocks_tx,
            available_blocks_tx,
        }
    }

    pub fn total_blocks_watcher(&self) -> watch::Receiver<u64> {
        self.total_blocks_tx.subscribe()
    }

    pub fn available_blocks_watcher(&self) -> watch::Receiver<u64> {
        self.available_blocks_tx.subscribe()
    }

    fn insert_with_sequence_hash(
        &mut self,
        block: PoolValue<Block<T, M>>,
        sequence_hash: SequenceHash,
    ) {
        let priority_key = PriorityKey::new(block.metadata().clone(), sequence_hash);
        if self.priority_set.contains(&priority_key) {
            tracing::debug!("multiple entries with the same priority key, resetting block and inserting into uninitialized set");
            let mut block = block;
            block.reset();
            self.uninitialized_set.push_back(block);
        } else {
            if self.lookup_map.contains_key(&sequence_hash) {
                tracing::debug!("multiple entries in lookup map with the same sequence hash, inserting into uninitialized set");
                let mut block = block;
                block.reset();
                self.uninitialized_set.push_back(block);
            } else {
                tracing::debug!("inserting block to map and priority set");
                self.priority_set.insert(priority_key);
                self.lookup_map.insert(sequence_hash, block);
            }
        }
    }

    // Insert an item with a given key and sequence_hash
    fn insert(&mut self, block: PoolValue<Block<T, M>>) {
        tracing::debug!("inserting block into available blocks");

        // If we already have an entry for this sequence hash or the block is reset,
        // we need to move it to the uninitialized set
        match block.state() {
            BlockState::Reset => {
                tracing::debug!("inserted block to uninitialized set");
                self.uninitialized_set.push_back(block);
            }
            BlockState::Partial(_) => {
                tracing::debug!("inserted block to uninitialized set");
                self.uninitialized_set.push_back(block);
            }
            BlockState::Complete(state) => {
                tracing::debug!("inserting completed/unregistered block to map and priority set");
                let sequence_hash = state.token_block.sequence_hash();
                self.insert_with_sequence_hash(block, sequence_hash);
            }
            BlockState::Registered(state) => {
                tracing::debug!("inserting registered block to map and priority set");
                let sequence_hash = state.sequence_hash;
                self.insert_with_sequence_hash(block, sequence_hash);
            }
        }
    }

    fn take_with_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Option<PoolValue<Block<T, M>>> {
        match self.lookup_map.remove(&sequence_hash) {
            Some(block) => {
                // Remove from priority set
                let priority_key = PriorityKey::new(block.metadata().clone(), sequence_hash);
                self.priority_set.remove(&priority_key);
                Some(block)
            }
            None => None,
        }
    }

    fn match_hashes(
        &mut self,
        hashes: Vec<SequenceHash>,
        return_handle: Arc<ReturnHandleImpl<T, M>>,
    ) -> Vec<PoolItem<Block<T, M>>> {
        let mut matched_blocks = Vec::with_capacity(hashes.len());

        for hash in hashes {
            if let Some(block) = self.take_with_sequence_hash(hash) {
                matched_blocks.push(self.create_pool_item(block, return_handle.clone()));
            } else {
                break;
            }
        }

        let count = matched_blocks.len() as u64;
        self.available_blocks_tx
            .send_modify(|n| *n = n.saturating_sub(count));

        matched_blocks
    }

    fn handle_match_single(&mut self, match_single: MatchSingle<T, M>) {
        let (hash, return_handle, rx) = match_single.dissolve();

        let matched_blocks = self.match_hashes(vec![hash], return_handle);
        let optional_single = matched_blocks.into_iter().next();

        // Send the result back through the channel
        if rx.send(optional_single).is_err() {
            tracing::trace!("Failed to send matched block to requester");
        }
    }

    fn handle_match_multiple(&mut self, match_multiple: MatchMultiple<T, M>) {
        let (hashes, return_handle, rx) = match_multiple.dissolve();

        let matched_blocks = self.match_hashes(hashes, return_handle);

        // Send the matched blocks back through the channel
        if rx.send(matched_blocks).is_err() {
            tracing::trace!("Failed to send matched blocks to requester");
        }
    }

    fn take(&mut self) -> Option<PoolValue<Block<T, M>>> {
        // First try uninitialized blocks - these are often part of sequences
        // that have been arranged in the correct order
        if let Some(block) = self.uninitialized_set.pop_front() {
            return Some(block);
        }

        // if we have blocks in the priority set, pop the first (it's sorted by priority)
        // a fatal error will occur if the block is not found in the lookup map
        if let Some(key) = self.priority_set.pop_first() {
            let block = match self.lookup_map.remove(&key.sequence_hash()) {
                Some(block) => block,
                None => {
                    panic!("block from priority set not found in lookup map");
                }
            };

            return Some(block);
        }

        None
    }

    fn handle_take(&mut self, take: Take<T, M>) {
        let (count, return_handle, tx) = take.dissolve();

        let mut taken_blocks = Vec::with_capacity(count as usize);

        for _ in 0..count {
            if let Some(block) = self.take() {
                taken_blocks.push(self.create_pool_item(block, return_handle.clone()));
            } else {
                break;
            }
        }

        let count = taken_blocks.len() as u64;
        self.available_blocks_tx
            .send_modify(|n| *n = n.saturating_sub(count));

        // Send the result back through the channel
        if tx.send(taken_blocks).is_err() {
            tracing::trace!("Failed to send matched blocks to requester");
        }
    }

    fn handle_match_request(&mut self, match_request: MatchRequest<T, M>) {
        match match_request {
            MatchRequest::MatchSingle(match_single) => self.handle_match_single(match_single),
            MatchRequest::MatchMultiple(match_multiple) => {
                self.handle_match_multiple(match_multiple)
            }
            MatchRequest::Take(take) => self.handle_take(take),
        }
    }

    fn handle_control_request(&mut self, control_request: ControlRequest<T, M>) {
        match control_request {
            ControlRequest::Insert(insert) => {
                let (block, tx) = insert.dissolve();
                self.handle_insert(block);
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send insert ack; receiver dropped");
                }
            }
            ControlRequest::UpdateSingle(update_single) => {
                let (update, tx) = update_single.dissolve();
                self.handle_update_single(update);
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send update single ack; receiver dropped");
                }
            }
            ControlRequest::UpdateMultiple(update_multiple) => {
                let (updates, tx) = update_multiple.dissolve();
                self.handle_update_multiple(updates);
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send update multiple ack; receiver dropped");
                }
            }
            ControlRequest::Reset(reset) => {
                let (sequence_hashes, tx, _) = reset.dissolve();
                self.handle_reset(sequence_hashes);
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send reset ack; receiver dropped");
                }
            }
            ControlRequest::ResetAll(reset_all) => {
                let (tx, _) = reset_all.dissolve();
                self.handle_reset_all();
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send reset all ack; receiver dropped");
                }
            }
        }
    }

    fn handle_insert(&mut self, block: Block<T, M>) {
        self.available_blocks_tx.send_modify(|n| *n += 1);
        self.total_blocks_tx.send_modify(|n| *n += 1);
        self.return_tick += 1;

        self.insert(PoolValue::Direct(block));
    }

    fn handle_return(&mut self, block: PoolValue<Block<T, M>>) {
        self.available_blocks_tx.send_modify(|n| *n += 1);
        self.return_tick += 1;

        self.insert(block);
    }

    fn handle_update_single(&mut self, update: UpdateBlock<M>) {
        self.update_block(vec![update]);
    }

    fn handle_update_multiple(&mut self, updates: Vec<UpdateBlock<M>>) {
        for update in updates {
            if let Some(mut block) = self.take_with_sequence_hash(update.hash) {
                *block.metadata_mut() = update.metadata;
                self.insert(block);
            }
        }
    }

    fn update_block(&mut self, updates: Vec<UpdateBlock<M>>) {
        for update in updates {
            if let Some(mut block) = self.take_with_sequence_hash(update.hash) {
                *block.metadata_mut() = update.metadata;
                self.insert(block);
            }
        }
    }

    fn handle_reset(&mut self, sequence_hashes: Vec<SequenceHash>) {
        for hash in sequence_hashes {
            if let Some(mut block) = self.take_with_sequence_hash(hash) {
                // Reset metadata through deref
                block.metadata_mut().reset_metadata();
                self.insert(block);
            }
        }
    }

    fn handle_reset_all(&mut self) {
        while let Some(priority_key) = self.priority_set.pop_first() {
            if let Some(mut block) = self.lookup_map.remove(&priority_key.sequence_hash()) {
                // reset block -- both state and metadata
                block.reset();
                self.insert(block);
            } else {
                panic!("block from priority set not found in lookup map");
            }
        }
    }
}

impl<T: Storage, M: BlockMetadata> PoolExt<Block<T, M>> for BlockPoolInner<T, M> {}

#[derive(Dissolve)]
pub struct MatchSingle<T: Storage + 'static, M: BlockMetadata> {
    hash: SequenceHash,
    return_handle: Arc<ReturnHandleImpl<T, M>>,
    tx: oneshot::Sender<Option<UniqueBlock<T, M>>>,
}

#[derive(Dissolve)]
pub struct MatchMultiple<T: Storage + 'static, M: BlockMetadata> {
    hashes: Vec<SequenceHash>,
    return_handle: Arc<ReturnHandleImpl<T, M>>,
    tx: oneshot::Sender<Vec<UniqueBlock<T, M>>>,
}

#[derive(Dissolve)]
pub struct Take<T: Storage + 'static, M: BlockMetadata> {
    count: u32,
    return_handle: Arc<ReturnHandleImpl<T, M>>,
    tx: oneshot::Sender<Vec<UniqueBlock<T, M>>>,
}

pub enum MatchRequest<T: Storage + 'static, M: BlockMetadata> {
    MatchSingle(MatchSingle<T, M>),
    MatchMultiple(MatchMultiple<T, M>),
    Take(Take<T, M>),
}

pub struct UpdateBlock<M: BlockMetadata> {
    hash: SequenceHash,
    metadata: M,
}

impl<M: BlockMetadata> UpdateBlock<M> {
    pub fn new(hash: SequenceHash, metadata: M) -> Self {
        Self { hash, metadata }
    }
}

#[derive(Dissolve)]
pub struct InsertControl<T: Storage + 'static, M: BlockMetadata> {
    block: Block<T, M>,
    tx: oneshot::Sender<()>,
}

#[derive(Dissolve)]
pub struct UpdateSingleControl<M: BlockMetadata> {
    update: UpdateBlock<M>,
    tx: oneshot::Sender<()>,
}

#[derive(Dissolve)]
pub struct UpdateMultipleControl<M: BlockMetadata> {
    updates: Vec<UpdateBlock<M>>,
    tx: oneshot::Sender<()>,
}

#[derive(Dissolve)]
pub struct ResetControl<M: BlockMetadata> {
    sequence_hashes: Vec<SequenceHash>,
    tx: oneshot::Sender<()>,
    _phantom: std::marker::PhantomData<M>,
}

#[derive(Dissolve)]
pub struct ResetAllControl<M: BlockMetadata> {
    tx: oneshot::Sender<()>,
    _phantom: std::marker::PhantomData<M>,
}

pub enum ControlRequest<T: Storage + 'static, M: BlockMetadata> {
    Insert(InsertControl<T, M>),
    UpdateSingle(UpdateSingleControl<M>),
    UpdateMultiple(UpdateMultipleControl<M>),
    Reset(ResetControl<M>),
    ResetAll(ResetAllControl<M>),
}

pub async fn progress_engine<T: Storage + 'static, M: BlockMetadata>(
    match_rx: mpsc::UnboundedReceiver<MatchRequest<T, M>>,
    return_rx: mpsc::UnboundedReceiver<PoolValue<Block<T, M>>>,
    ctrl_rx: mpsc::UnboundedReceiver<ControlRequest<T, M>>,
    fence_rx: mpsc::UnboundedReceiver<oneshot::Sender<()>>,
    mut state: BlockPoolInner<T, M>,
) {
    let mut match_rx = match_rx;
    let mut return_rx = return_rx;
    let mut ctrl_rx = ctrl_rx;
    let mut fence_rx = fence_rx;

    loop {
        tokio::select! {
            biased;

            Some(match_req) = match_rx.recv(), if !match_rx.is_closed() => {
                state.handle_match_request(match_req);
            }

            Some(block) = return_rx.recv(), if !return_rx.is_closed() => {
                state.handle_return(block);
            }

            Some(req) = ctrl_rx.recv(), if !ctrl_rx.is_closed() => {
                state.handle_control_request(req);
            }

            Some(tx) = fence_rx.recv() => {
                if tx.send(()).is_err() {
                    tracing::trace!("Failed to send fence ack; receiver dropped");
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{
        block_manager::{
            block::{BlockStorage, BlockStorageCollection},
            layout::{BlockLayout, NullLayout},
            storage::NullStorage,
        },
        tokens::{Token, Tokens},
    };

    use super::*;

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
    pub struct TestMetadata {
        priority: u32,
        return_tick: u64,
    }

    impl BlockMetadata for TestMetadata {
        fn on_acquired(&mut self) {}

        fn on_returned(&mut self, tick: u64) {
            self.return_tick = tick;
        }

        fn is_reset(&self) -> bool {
            self.priority == 0 && self.return_tick == 0
        }

        fn reset_metadata(&mut self) {
            self.priority = 0;
            self.return_tick = 0;
        }
    }

    type TestPriorityKey = PriorityKey<TestMetadata>;

    fn make_priority_key(priority: u32, return_tick: u64) -> TestPriorityKey {
        TestPriorityKey {
            metadata: TestMetadata {
                priority,
                return_tick,
            },
            sequence_hash: SequenceHash::from(return_tick),
        }
    }

    #[test]
    fn test_priority_key_ord() {
        let mut map = BTreeSet::new();

        let hash1 = SequenceHash::from(1u64);
        let hash2 = SequenceHash::from(2u64);
        let hash3 = SequenceHash::from(3u64);

        map.insert(make_priority_key(0, 2));
        map.insert(make_priority_key(1, 1));
        map.insert(make_priority_key(0, 3));

        // Test popping from the map to verify ordering
        let first_key = map.pop_first().unwrap();
        assert_eq!(first_key.metadata.priority, 0);
        assert_eq!(first_key.metadata.return_tick, 2);
        assert_eq!(first_key.sequence_hash, hash1);

        let second_key = map.pop_first().unwrap();
        assert_eq!(second_key.metadata.priority, 0);
        assert_eq!(second_key.metadata.return_tick, 3);
        assert_eq!(second_key.sequence_hash, hash3);

        let third_key = map.pop_first().unwrap();
        assert_eq!(third_key.metadata.priority, 1);
        assert_eq!(third_key.metadata.return_tick, 1);
        assert_eq!(third_key.sequence_hash, hash2);

        // Map should now be empty
        assert!(map.is_empty());
    }

    // Helper function to create a sequence of tokens
    pub fn create_token_sequence(values: &[u32]) -> Tokens {
        let tokens: Vec<Token> = values.iter().map(|&v| Token::from(v)).collect();
        Tokens::from(tokens)
    }

    pub async fn create_block_pool(num_blocks: usize) -> BlockPool<NullStorage, TestMetadata> {
        let pool = BlockPool::new().await;

        let block_collection = BlockStorageCollection::<NullStorage, TestMetadata>::new(
            NullStorage::default(),
            NullLayout::new(num_blocks),
        )
        .unwrap();

        let blocks = block_collection.into_blocks().unwrap();

        for block in blocks {
            pool.insert(block).await.unwrap();
        }

        pool
    }

    // // Helper to create blocks from a sequence with given size
    // pub fn populate_block_pool(
    //     tokens: Tokens,
    //     block_size: usize,
    //     pool: &BlockPool<NullStorage, TestMetadata>,
    // ) -> Vec<Block<NullStorage, TestMetadata>> {
    //     let (token_blocks, partial_token_block) = tokens.into_sequence(block_size).into_parts();

    //     // pop off the last block for partial
    //     let partial_block = blocks.pop().unwrap();

    //     // zip token and blocks and initialize the blocks
    //     token_blocks
    //         .into_iter()
    //         .map(|token_block| {
    //             let mut block = pool.take_block(token_block.sequence_hash()).await.unwrap();
    //             block.set_sequence_hash(token_block.sequence_hash());
    //             block.set_block_hash(token_block.block_hash());
    //             block
    //         })
    //         .collect()
    // }

    // #[tokio::test]
    // async fn test_basic_sequence_matching() {
    //     let pool = BlockPool::new().await;

    //     // Create a sequence of 4 tokens split into blocks of 2
    //     let sequence = create_token_sequence(&[1, 2, 3, 4]);
    //     let blocks = create_blocks(sequence, 2);
    //     assert_eq!(blocks.len(), 2);

    //     // Match the blocks in sequence
    //     let hashes: Vec<_> = blocks
    //         .iter()
    //         .map(|b| b.token_block.sequence_hash())
    //         .collect();

    //     // Insert blocks into pool
    //     for block in blocks {
    //         pool.insert(block).await.unwrap();
    //     }

    //     pool.fence().await.unwrap();

    //     assert_eq!(pool.total_blocks(), 2);
    //     assert_eq!(pool.available_blocks(), 2);

    //     // Match the blocks in sequence
    //     let matched = pool.match_blocks(hashes.clone()).await.unwrap();
    //     assert_eq!(matched.len(), 2);

    //     assert_eq!(pool.total_blocks(), 2);
    //     assert_eq!(pool.available_blocks(), 0);

    //     // Validate the blocks are in the correct order and match the sequence hashes
    //     assert_eq!(matched[0].token_block.sequence_hash(), hashes[0]);
    //     assert_eq!(matched[1].token_block.sequence_hash(), hashes[1]);

    //     // Return blocks in reverse order (tail to root)
    //     for block in matched.into_iter().rev() {
    //         drop(block); // This will trigger return_to_pool
    //     }

    //     pool.fence().await.unwrap();

    //     assert_eq!(pool.total_blocks(), 2);
    //     assert_eq!(pool.available_blocks(), 2);
    // }

    // #[tokio::test]
    // async fn test_equal_priority_taking() {
    //     let pool = BlockPool::new().await;

    //     // Create two sequences with different priorities
    //     let seq1 = create_token_sequence(&[1, 2, 3, 4]);
    //     let seq2 = create_token_sequence(&[5, 6, 7, 8]);

    //     let mut blocks1 = create_blocks(seq1, 2);
    //     let mut blocks2 = create_blocks(seq2, 2);

    //     for block in blocks1.iter_mut() {
    //         block.priority = 1;
    //     }
    //     for block in blocks2.iter_mut() {
    //         block.priority = 1;
    //     }

    //     // If priorities were equal, first in, first out would apply

    //     // Insert Sequence 2
    //     for block in blocks2.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     // Insert Sequence 1
    //     for block in blocks1.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     pool.fence().await.unwrap();

    //     let blocks = pool.take_blocks(4).await.unwrap();
    //     assert_eq!(blocks.len(), 4);

    //     // Validate the blocks are in the correct order
    //     assert_eq!(blocks[0].token_block.tokens()[0], 7);
    //     assert_eq!(blocks[1].token_block.tokens()[0], 5);
    //     assert_eq!(blocks[2].token_block.tokens()[0], 3);
    //     assert_eq!(blocks[3].token_block.tokens()[0], 1);
    // }

    // #[tokio::test]
    // async fn test_priority_taking() {
    //     let pool = BlockPool::new().await;

    //     // Create two sequences with different priorities
    //     let seq1 = create_token_sequence(&[1, 2, 3, 4]);
    //     let seq2 = create_token_sequence(&[5, 6, 7, 8]);

    //     let mut blocks1 = create_blocks(seq1, 2);
    //     let mut blocks2 = create_blocks(seq2, 2);

    //     for block in blocks1.iter_mut() {
    //         block.priority = 1;
    //     }
    //     for block in blocks2.iter_mut() {
    //         block.priority = 2;
    //     }

    //     // If priorities were equal, first in, first out would apply
    //     // but here we have a higher priority block first (which are taken last)
    //     // returned first, but lower priority blocks inserted after
    //     // we expect the lower priority blocks to be taken first

    //     // Insert Sequence 2
    //     for block in blocks2.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     // Insert Sequence 1
    //     for block in blocks1.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     pool.fence().await.unwrap();

    //     let blocks = pool.take_blocks(4).await.unwrap();
    //     assert_eq!(blocks.len(), 4);

    //     // Validate the blocks are in the correct order
    //     assert_eq!(blocks[0].token_block.tokens()[0], 3);
    //     assert_eq!(blocks[1].token_block.tokens()[0], 1);
    //     assert_eq!(blocks[2].token_block.tokens()[0], 7);
    //     assert_eq!(blocks[3].token_block.tokens()[0], 5);
    // }

    // #[tokio::test]
    // async fn test_priority_taking_after_update() {
    //     let pool = BlockPool::new().await;

    //     // Create two sequences with different priorities
    //     let seq1 = create_token_sequence(&[1, 2, 3, 4]);
    //     let seq2 = create_token_sequence(&[5, 6, 7, 8]);

    //     let mut blocks1 = create_blocks(seq1, 2);
    //     let mut blocks2 = create_blocks(seq2, 2);

    //     for block in blocks1.iter_mut() {
    //         block.priority = 1;
    //     }
    //     for block in blocks2.iter_mut() {
    //         block.priority = 1;
    //     }

    //     // record hash of blocks 2
    //     // insert blocks 2, then blocks 1
    //     // update priority of blocks 2 to 2 using the update api
    //     // pull 4 blocks and test order

    //     let block_hashes = blocks2
    //         .iter()
    //         .map(|b| b.token_block.sequence_hash())
    //         .collect::<Vec<_>>();

    //     // Insert Sequence 2
    //     for block in blocks2.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     // Insert Sequence 1
    //     for block in blocks1.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     pool.fence().await.unwrap();

    //     // Update priority of blocks 2 to 2
    //     pool.update_multiple(
    //         block_hashes
    //             .into_iter()
    //             .map(|h| UpdateBlock {
    //                 hash: h,
    //                 priority: Some(2),
    //             })
    //             .collect(),
    //     )
    //     .await
    //     .unwrap();

    //     pool.fence().await.unwrap();

    //     let blocks = pool.take_blocks(4).await.unwrap();
    //     assert_eq!(blocks.len(), 4);

    //     // Validate the blocks are in the correct order
    //     assert_eq!(blocks[0].token_block.tokens()[0], 3);
    //     assert_eq!(blocks[1].token_block.tokens()[0], 1);
    //     assert_eq!(blocks[2].token_block.tokens()[0], 7);
    //     assert_eq!(blocks[3].token_block.tokens()[0], 5);
    // }

    // #[tokio::test]
    // async fn test_reset_all() {
    //     let pool = BlockPool::new().await;

    //     // Create two sequences with different priorities
    //     let seq1 = create_token_sequence(&[1, 2, 3, 4]);
    //     let seq2 = create_token_sequence(&[5, 6, 7, 8]);

    //     let mut blocks1 = create_blocks(seq1, 2);
    //     let mut blocks2 = create_blocks(seq2, 2);

    //     for block in blocks1.iter_mut() {
    //         block.priority = 1;
    //     }

    //     for block in blocks2.iter_mut() {
    //         block.priority = 1;
    //     }

    //     // record hash of blocks 2
    //     let block_hashes = blocks2
    //         .iter()
    //         .map(|b| b.token_block.sequence_hash())
    //         .collect::<Vec<_>>();

    //     // Insert Sequence 2
    //     for block in blocks2.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     // Insert Sequence 1
    //     for block in blocks1.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     // Reset All
    //     pool.reset_all().await.unwrap();
    //     pool.fence().await.unwrap();

    //     // Try to match from block 2 hashes, expect no matches
    //     let matched = pool.match_blocks(block_hashes).await.unwrap();
    //     assert_eq!(matched.len(), 0);
    // }

    // #[tokio::test]
    // async fn test_reset_block2() {
    //     let pool = BlockPool::new().await;

    //     // Create two sequences with different priorities
    //     let seq1 = create_token_sequence(&[1, 2, 3, 4]);
    //     let seq2 = create_token_sequence(&[5, 6, 7, 8]);

    //     let mut blocks1 = create_blocks(seq1, 2);
    //     let mut blocks2 = create_blocks(seq2, 2);

    //     for block in blocks1.iter_mut() {
    //         block.priority = 1;
    //     }

    //     for block in blocks2.iter_mut() {
    //         block.priority = 1;
    //     }

    //     // record hash of blocks 2
    //     let block2_hashes = blocks2
    //         .iter()
    //         .map(|b| b.token_block.sequence_hash())
    //         .collect::<Vec<_>>();

    //     let block1_hashes = blocks1
    //         .iter()
    //         .map(|b| b.token_block.sequence_hash())
    //         .collect::<Vec<_>>();

    //     // Insert Sequence 2
    //     for block in blocks2.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     // Insert Sequence 1
    //     for block in blocks1.into_iter().rev() {
    //         pool.insert(block).await.unwrap();
    //     }

    //     // Reset Block 2
    //     pool.reset(block2_hashes.clone()).await.unwrap();
    //     pool.fence().await.unwrap();

    //     // Try to match from block 2 hashes, expect no matches
    //     let matched = pool.match_blocks(block2_hashes).await.unwrap();
    //     assert_eq!(matched.len(), 0);

    //     let matched = pool.match_blocks(block1_hashes).await.unwrap();
    //     assert_eq!(matched.len(), 2);
    // }
}
