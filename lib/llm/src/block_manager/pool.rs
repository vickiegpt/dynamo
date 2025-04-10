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

//! # KV Block Pools
//!
//! The InactiveBlockPool manages KV blocks that are not actively in use but retain their previous state.
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
//!
//! The [ActiveBlockPool] manages KV blocks that are actively in use.

pub mod active;
mod inactive;

use derive_getters::Dissolve;
use dynamo_runtime::{
    raise,
    utils::pool::{PoolExt, PoolItem, PoolValue, ReturnHandle, SharedPoolItem},
    Result,
};
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    ops::Deref,
    sync::Arc,
};
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

pub type UniqueBlock<S, M> = PoolItem<Block<S, M>>;
pub type SharedBlock<S, M> = SharedPoolItem<Block<S, M>>;

pub struct InactiveBlockPool<T: Storage, M: BlockMetadata> {
    match_tx: mpsc::UnboundedSender<MatchRequest<T, M>>,
    control_tx: mpsc::UnboundedSender<ControlRequest<T, M>>,
    fence_tx: mpsc::UnboundedSender<oneshot::Sender<()>>,
    return_handle: Arc<ReturnHandleImpl<T, M>>,
    total_blocks_rx: watch::Receiver<u64>,
    available_blocks_rx: watch::Receiver<u64>,
    join_handle: JoinHandle<()>,
}

/// Concrete implementation of the ReturnHandle trait for returning
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
pub struct PriorityKey<M: BlockMetadata> {
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

    pub fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }

    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    pub fn update_metadata(&mut self, metadata: M) {
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

#[cfg(test)]
pub(crate) mod tests {
    use crate::{
        block_manager::{
            block::{state::CompleteState, BlockStorageCollection},
            layout::NullLayout,
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

    fn make_priority_key(
        priority: u32,
        return_tick: u64,
        sequence_hash: SequenceHash,
    ) -> TestPriorityKey {
        TestPriorityKey {
            metadata: TestMetadata {
                priority,
                return_tick,
            },
            sequence_hash,
        }
    }

    #[test]
    fn test_priority_key_ord() {
        let mut map = BTreeSet::new();

        let hash1 = SequenceHash::from(1u64);
        let hash2 = SequenceHash::from(2u64);
        let hash3 = SequenceHash::from(3u64);

        map.insert(make_priority_key(0, 2, hash1));
        map.insert(make_priority_key(1, 1, hash2));
        map.insert(make_priority_key(0, 3, hash3));

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

    pub async fn create_block_pool(
        num_blocks: usize,
    ) -> InactiveBlockPool<NullStorage, TestMetadata> {
        let pool = InactiveBlockPool::new().await;

        let block_collection =
            BlockStorageCollection::<NullStorage, TestMetadata>::new(NullLayout::new(num_blocks))
                .unwrap();

        let blocks = block_collection.into_blocks().unwrap();

        for block in blocks {
            pool.insert(block).await.unwrap();
        }

        pool
    }

    pub async fn acquire_blocks(
        tokens: Tokens,
        block_size: usize,
        pool: &InactiveBlockPool<NullStorage, TestMetadata>,
    ) -> (Vec<UniqueBlock<NullStorage, TestMetadata>>, usize) {
        let (mut token_blocks, _partial_token_block) =
            tokens.into_sequence(block_size).into_parts();

        let total_complete_blocks = token_blocks.len();

        // this will match the token_blocks to any matching blocks in the inactive pool
        // these blocks have the same sequence hash as the token_blocks, thus no updates are needed
        let matched_blocks = pool.match_token_blocks(&token_blocks).await.unwrap();
        let matched_block_count = matched_blocks.len();
        println!("matched_blocks: {:?}", matched_block_count);

        // all matched blocks should be in the complete or registered state
        for block in &matched_blocks {
            assert!(block.is_complete());
        }

        // drain the matched blocks from the token_blocks
        token_blocks.drain(0..matched_block_count);

        assert_eq!(
            token_blocks.len() + matched_blocks.len(),
            total_complete_blocks
        );

        // try to acquire the remaining blocks
        let mut unmatched_blocks = pool.take_blocks(token_blocks.len() as u32).await.unwrap();

        assert_eq!(unmatched_blocks.len(), token_blocks.len());

        for unmatched in &unmatched_blocks {
            assert!(unmatched.is_empty());
        }

        for (unmatched, token_block) in unmatched_blocks.iter_mut().zip(token_blocks.into_iter()) {
            assert!(unmatched.is_empty());
            *unmatched.state_mut() = BlockState::Complete(CompleteState { token_block });
        }

        let mut blocks = matched_blocks;
        blocks.extend(unmatched_blocks);
        (blocks, matched_block_count)
    }

    #[tokio::test]
    async fn test_block_pool_lifecycle() {
        dynamo_runtime::logging::init();

        const PAGE_SIZE: usize = 2;

        let pool = create_block_pool(10).await;
        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let blocks = pool.take_blocks(10).await.unwrap();
        assert_eq!(blocks.len(), 10);
        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 0);

        drop(blocks);
        pool.fence().await.unwrap();

        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let tokens = create_token_sequence(&[1, 2, 3, 4]);

        let (blocks, matched_block_count) = acquire_blocks(tokens.clone(), PAGE_SIZE, &pool).await;
        assert_eq!(blocks.len(), 2);
        assert_eq!(matched_block_count, 0);
        assert_eq!(pool.available_blocks(), 8);

        drop(blocks);
        pool.fence().await.unwrap();

        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let (blocks, matched_block_count) = acquire_blocks(tokens.clone(), PAGE_SIZE, &pool).await;
        assert_eq!(blocks.len(), 2);
        assert_eq!(matched_block_count, 2);
        assert_eq!(pool.available_blocks(), 8);

        drop(blocks);
        pool.fence().await.unwrap();

        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let blocks = pool.take_blocks(10).await.unwrap();
        for block in &blocks {
            assert!(block.is_empty());
        }
    }

    // #[tokio::test]
    // async fn test_basic_sequence_matching() {
    //     let pool = InactiveBlockPool::new().await;

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
    //     let matched = pool.match_sequence_hashes(hashes.clone()).await.unwrap();
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
    //     let pool = InactiveBlockPool::new().await;

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
    //     let pool = InactiveBlockPool::new().await;

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
    //     let pool = InactiveBlockPool::new().await;

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
    //     let pool = InactiveBlockPool::new().await;

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
    //     let matched = pool.match_sequence_hashes(block_hashes).await.unwrap();
    //     assert_eq!(matched.len(), 0);
    // }

    // #[tokio::test]
    // async fn test_reset_block2() {
    //     let pool = InactiveBlockPool::new().await;

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
    //     let matched = pool.match_sequence_hashes(block2_hashes).await.unwrap();
    //     assert_eq!(matched.len(), 0);

    //     let matched = pool.match_sequence_hashes(block1_hashes).await.unwrap();
    //     assert_eq!(matched.len(), 2);
    // }
}
