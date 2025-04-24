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

//! # KV Cache Block Pool Management
//!
//! This module provides the primary [`BlockPool`] structure for managing KV cache blocks.
//! It orchestrates the allocation, registration, and reuse of blocks by coordinating
//! between an [`ActiveBlockPool`] and an [`InactiveBlockPool`].
//!
//! ## Core Components:
//!
//! - **[`BlockPool`]**: The main entry point for interacting with the block management system.
//!   It holds the shared state containing both active and inactive pools.
//! - **[`ActiveBlockPool`]**: Manages blocks that are currently associated with active sequences.
//!   It primarily uses weak references to track these blocks, allowing them to be potentially
//!   reclaimed by the inactive pool if no strong references remain.
//! - **[`InactiveBlockPool`]**: Manages blocks that are not currently in active use. It supports
//!   block reuse by matching sequence hashes and employs a priority-based eviction strategy
//!   for acquiring free blocks.
//! - **[`BlockRegistry`]**: Manages the registration of blocks that have transitioned from the
//!   Complete to Registered state.
//! - **[`MutableBlock`]**: Represents a uniquely owned block, typically obtained from allocation.
//!   It allows modification and is returned to the inactive pool upon being dropped.
//! - **[`ImmutableBlock`]**: Represents a shared, immutable reference to a block, usually after
//!   it has been registered or matched. Ensures that multiple sequences can reference the
//!   same underlying block data.
//!
//! ## Workflow:
//!
//! 1.  Blocks are initially added to the [`BlockPool`] via [`BlockPool::add_blocks`], populating the
//!     [`InactiveBlockPool`].
//! 2.  Sequences request blocks via [`BlockPool::allocate_blocks`], which attempts to acquire them
//!     from the [`InactiveBlockPool`]. This returns [`MutableBlock`]s.
//! 3.  Once a [`MutableBlock`] is filled and ready, it's registered using [`BlockPool::register_block`].
//!     This process checks the both the [`ActiveBlockPool`] and the [`InactiveBlockPool`] for existing blocks
//!     with the same content hash. It returns an [`ImmutableBlock`] representing the canonical block
//!     (either the one provided or an existing one).
//! 4.  Sequences can also try to reuse blocks directly using [`BlockPool::match_sequence_hash`], which
//!     checks both the active and inactive pools.
//! 5.  When an [`ImmutableBlock`] is no longer needed by any sequence (its `Arc` count drops to zero),
//!     the underlying [`MutableBlock`] (if it still exists via the weak reference in the active pool)
//!     can eventually be returned to the [`InactiveBlockPool`] when its final strong reference (the `Arc`
//!     within `ImmutableBlock`) is dropped.
//! 6.  Dropped [`MutableBlock`]s are automatically returned to the [`InactiveBlockPool`].

mod active;
mod inactive;
mod priority_key;
mod state;

use active::ActiveBlockPool;
use derive_builder::Builder;
use derive_getters::Dissolve;
use inactive::InactiveBlockPool;
use priority_key::PriorityKey;
use tokio::sync::oneshot;

use super::block::BlockError;
use super::block::{registry::BlockRegistry, Block, BlockMetadata};
use super::events::{EventManager, NullEventManager};
use super::layout::BlockLayout;

use crate::block_manager::block::BlockState;
use crate::tokens::{SequenceHash, TokenBlock};

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    ops::{Deref, DerefMut},
    sync::{Arc, Weak},
};
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    utils::pool::{PoolItem, SharedPoolItem},
    Result,
};

pub type BlockType<S, M> = Block<S, M>;
pub type UniqueBlock<S, M> = PoolItem<Block<S, M>>;
pub type SharedBlock<S, M> = SharedPoolItem<Block<S, M>>;

#[derive(Debug, thiserror::Error)]
pub enum BlockPoolError {
    #[error("Block is not complete")]
    BlockNotComplete,

    #[error("Not enough blocks available, requested: {0}, available: {1}")]
    NotEnoughBlocksAvailable(usize, usize),

    #[error("Invalid MutableBlock: {0}")]
    InvalidMutableBlock(String),

    #[error("Failed to register block: {0}")]
    FailedToRegisterBlock(String),

    #[error("Progress engine shutdown")]
    ProgressEngineShutdown,

    #[error(transparent)]
    BlockError(#[from] BlockError),
}

#[derive(Builder, Dissolve)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct BlockPoolArgs<S: BlockLayout, M: BlockMetadata> {
    #[builder(default = "Runtime::default()")]
    runtime: Runtime,

    #[builder(default = "NullEventManager::new()")]
    event_manager: Arc<dyn EventManager>,

    #[builder(default = "CancellationToken::new()")]
    cancel_token: CancellationToken,

    #[builder(default)]
    blocks: Vec<Block<S, M>>,
}

impl<S: BlockLayout, M: BlockMetadata> BlockPoolArgsBuilder<S, M> {
    pub fn build(self) -> anyhow::Result<BlockPool<S, M>> {
        let args = self.build_internal()?;
        let (runtime, event_manager, cancel_token, blocks) = args.dissolve();

        let pool = BlockPool::new(event_manager, runtime, cancel_token);
        pool.add_blocks_blocking(blocks)?;
        Ok(pool)
    }

    pub fn build_with_progress_engine(
        self,
    ) -> anyhow::Result<(BlockPool<S, M>, ProgressEngine<S, M>)> {
        let args = self.build_internal()?;
        let (runtime, event_manager, cancel_token, blocks) = args.dissolve();
        let (pool, mut progress_engine) =
            BlockPool::with_progress_engine(event_manager, runtime, cancel_token);

        progress_engine.state.inactive.add_blocks(blocks);

        Ok((pool, progress_engine))
    }
}
/// Manages the blocks in a specific storage backenda
pub struct BlockPool<S: BlockLayout, M: BlockMetadata> {
    runtime: Runtime,
    priority_tx: tokio::sync::mpsc::UnboundedSender<PriorityRequest<S, M>>,
    ctrl_tx: tokio::sync::mpsc::UnboundedSender<ControlRequest<S, M>>,
}

impl<S: BlockLayout, M: BlockMetadata> Clone for BlockPool<S, M> {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            priority_tx: self.priority_tx.clone(),
            ctrl_tx: self.ctrl_tx.clone(),
        }
    }
}

#[derive(Dissolve)]
struct Unary<Req, Resp> {
    request: Req,
    response_tx: oneshot::Sender<Resp>,
}

impl<Req, Resp> Unary<Req, Resp> {
    fn make_request(request: Req) -> (Self, oneshot::Receiver<Resp>) {
        let (response_tx, response_rx) = oneshot::channel();
        (
            Self {
                request,
                response_tx,
            },
            response_rx,
        )
    }
}

enum PriorityRequest<S: BlockLayout, M: BlockMetadata> {
    AllocateBlocks(Unary<usize, Result<Vec<MutableBlock<S, M>>, BlockPoolError>>),
    RegisterBlocks(
        Unary<Vec<MutableBlock<S, M>>, Result<Vec<ImmutableBlock<S, M>>, BlockPoolError>>,
    ),
    MatchSequenceHashes(Unary<Vec<SequenceHash>, Vec<ImmutableBlock<S, M>>>),
}

enum ControlRequest<S: BlockLayout, M: BlockMetadata> {
    AddBlocks(Unary<Vec<Block<S, M>>, ()>),
}

#[derive(Debug, Clone)]
pub enum Runtime {
    Handle(tokio::runtime::Handle),
    Runtime(Arc<tokio::runtime::Runtime>),
}

impl Default for Runtime {
    fn default() -> Self {
        // detect if we are running in a tokio runtime
        if let Ok(_) = tokio::runtime::Handle::try_current() {
            Self::current_thread()
        } else {
            Self::single_threaded()
        }
    }
}

impl Runtime {
    fn current_thread() -> Self {
        Self::Handle(tokio::runtime::Handle::current())
    }

    fn single_threaded() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .max_blocking_threads(1)
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("failed to create runtime");

        Self::Runtime(Arc::new(runtime))
    }

    fn handle(&self) -> tokio::runtime::Handle {
        match self {
            Runtime::Handle(handle) => handle.clone(),
            Runtime::Runtime(runtime) => runtime.handle().clone(),
        }
    }
}

pub struct MutableBlock<S: BlockLayout, M: BlockMetadata> {
    block: Option<Block<S, M>>,
    return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, M>>,
}

impl<S: BlockLayout, M: BlockMetadata> MutableBlock<S, M> {
    fn new(block: Block<S, M>, return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, M>>) -> Self {
        Self {
            block: Some(block),
            return_tx,
        }
    }
}

impl<S: BlockLayout, M: BlockMetadata> std::fmt::Debug for MutableBlock<S, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MutableBlock {{ block: {:?} }}", self.block)
    }
}

impl<S: BlockLayout, M: BlockMetadata> Drop for MutableBlock<S, M> {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            if self.return_tx.send(block).is_err() {
                tracing::warn!("block pool shutdown before block was returned");
            }
        }
    }
}

#[derive(Debug)]
pub struct ImmutableBlock<S: BlockLayout, M: BlockMetadata> {
    block: Arc<MutableBlock<S, M>>,
}

impl<S: BlockLayout, M: BlockMetadata> BlockPool<S, M> {
    pub fn builder() -> BlockPoolArgsBuilder<S, M> {
        BlockPoolArgsBuilder::default()
    }

    /// Creates a new [`BlockPool`] with the given [`EventManager`].
    ///
    /// The pool starts empty and requires blocks to be added via [`add_blocks`].
    ///
    /// # Arguments
    ///
    /// * `event_manager` - An [`Arc<dyn EventManager>`] used for publishing block registration/removal events.
    ///
    /// # Returns
    ///
    /// A new [`BlockPool`] instance.
    fn new(
        event_manager: Arc<dyn EventManager>,
        runtime: Runtime,
        cancel_token: CancellationToken,
    ) -> Self {
        let (pool, progress_engine) =
            Self::with_progress_engine(event_manager, runtime, cancel_token);

        pool.runtime.handle().spawn(async move {
            let mut progress_engine = progress_engine;
            while progress_engine.step().await {
                tracing::trace!("progress engine step");
            }
        });

        pool
    }

    fn with_progress_engine(
        event_manager: Arc<dyn EventManager>,
        runtime: Runtime,
        cancel_token: CancellationToken,
    ) -> (Self, ProgressEngine<S, M>) {
        let (priority_tx, priority_rx) = tokio::sync::mpsc::unbounded_channel();
        let (ctrl_tx, ctrl_rx) = tokio::sync::mpsc::unbounded_channel();

        let progress_engine =
            ProgressEngine::<S, M>::new(event_manager, priority_rx, ctrl_rx, cancel_token);

        (
            Self {
                runtime,
                priority_tx,
                ctrl_tx,
            },
            progress_engine,
        )
    }

    /// Adds a vector of [`Block`]s to the [`InactiveBlockPool`].
    ///
    /// These blocks are typically created from a [`super::block::Blocks`]
    /// and represent the initial set of available cache blocks.
    /// Blocks added this way are initially reset.
    ///
    /// # Arguments
    ///
    /// * `blocks` - A [`Vec<Block<S, M>>`] to add to the inactive pool.
    pub async fn add_blocks(&self, blocks: Vec<Block<S, M>>) -> Result<(), BlockPoolError> {
        // Create the request
        let (req, resp_rx) = Unary::<_, ()>::make_request(blocks);

        // Issue the request
        self.ctrl_tx
            .send(ControlRequest::AddBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        // Await a response
        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;
        Ok(())
    }

    /// Blocking version of [`BlockPool::add_blocks`].
    pub fn add_blocks_blocking(&self, blocks: Vec<Block<S, M>>) -> Result<(), BlockPoolError> {
        self.runtime
            .handle()
            .block_on(async move { self.add_blocks(blocks).await })
    }

    /// Attempts to allocate a specified number of free blocks from the [`InactiveBlockPool`].
    ///
    /// Blocks acquired this way are returned as [`MutableBlock`]s, granting unique ownership
    /// and allowing modification. Dropping a [`MutableBlock`] automatically returns it
    /// to the [`InactiveBlockPool`].
    ///
    /// # Arguments
    ///
    /// * `count` - The number of blocks to allocate.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing:
    /// - `Ok(Vec<MutableBlock<S, M>>)`: If successful, a vector of allocated mutable blocks.
    /// - `Err(BlockPoolError)`: If not enough blocks are available in the inactive pool.
    pub async fn allocate_blocks(
        &self,
        count: usize,
    ) -> Result<Vec<MutableBlock<S, M>>, BlockPoolError> {
        // Create the request
        let (req, resp_rx) =
            Unary::<_, Result<Vec<MutableBlock<S, M>>, BlockPoolError>>::make_request(count);

        // Issue the request
        self.priority_tx
            .send(PriorityRequest::AllocateBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        // Await a response
        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    // /// Registers a [`MutableBlock`] (presumably after filling it) with the pool,
    // /// making it potentially available for sharing via the [`ActiveBlockPool`].
    // ///
    // /// This function checks if a block with the same sequence hash already exists
    // /// in the active pool. If so, it returns an [`ImmutableBlock`] pointing to the
    // /// existing block, and the provided `block` is implicitly dropped (returned to
    // /// the inactive pool). If no matching block exists, the provided `block` is
    // /// added to the active pool (via a weak reference) and an [`ImmutableBlock`]
    // /// pointing to it is returned.
    // ///
    // /// # Arguments
    // ///
    // /// * `block` - The [`MutableBlock<S, M>`] to register.
    // ///
    // /// # Returns
    // ///
    // /// A [`Result`] containing:
    // /// - `Ok(ImmutableBlock<S, M>)`: An immutable, shareable reference to the registered block.
    // /// - `Err(BlockPoolError)`: If the provided block is in an invalid state (e.g., has no sequence hash).
    // pub async fn register_block(
    //     &mut self,
    //     block: MutableBlock<S, M>,
    // ) -> Result<ImmutableBlock<S, M>, BlockPoolError> {
    //     self.register_blocks(vec![block])
    //         .await?
    //         .first()
    //         .ok_or(BlockPoolError::FailedToRegisterBlock("".to_string()))
    // }

    /// Registers a vector of [`MutableBlock`]s (presumably after filling them) with the pool,
    /// making them available for sharing via the [`ActiveBlockPool`].
    ///
    /// This function checks if any of the blocks have the same sequence hash as an existing block
    /// in the active pool. If so, it returns an [`ImmutableBlock`] pointing to the existing block,
    /// and the provided `block` is implicitly dropped (returned to the [`InactiveBlockPool`]).
    pub async fn register_blocks(
        &mut self,
        blocks: Vec<MutableBlock<S, M>>,
    ) -> Result<Vec<ImmutableBlock<S, M>>, BlockPoolError> {
        // Make the request
        let (req, resp_rx) =
            Unary::<_, Result<Vec<ImmutableBlock<S, M>>, BlockPoolError>>::make_request(blocks);

        // Issue the request
        self.priority_tx
            .send(PriorityRequest::RegisterBlocks(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        // Await a response
        resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    /// Attempts to match the given [`SequenceHash`] to an existing block, checking
    /// both the active and inactive pools.
    ///
    /// Checks the [`ActiveBlockPool`] first. If a valid strong reference exists, it returns
    /// an [`ImmutableBlock`] cloned from it. If the weak reference exists but is stale,
    /// it's removed.
    ///
    /// If not found in the active pool, it checks the [`InactiveBlockPool`]. If found there,
    /// the block is moved to the active pool (tracked by a weak reference) and returned
    /// as a new [`ImmutableBlock`].
    ///
    /// # Arguments
    ///
    /// * `sequence_hash` - The [`SequenceHash`] to look for.
    ///
    /// # Returns
    ///
    /// An [`Option<ImmutableBlock<S, M>>`] containing the shared block if found, otherwise `None`.
    pub async fn match_sequence_hashes(
        &mut self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<ImmutableBlock<S, M>>, BlockPoolError> {
        // Create the request
        let (req, resp_rx) =
            Unary::<_, Vec<ImmutableBlock<S, M>>>::make_request(sequence_hashes.into());

        // Issue the request
        self.priority_tx
            .send(PriorityRequest::MatchSequenceHashes(req))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        // Await a response
        Ok(resp_rx
            .await
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?)
    }
}

impl<S: BlockLayout, M: BlockMetadata> Deref for MutableBlock<S, M> {
    type Target = Block<S, M>;

    fn deref(&self) -> &Self::Target {
        self.block.as_ref().expect("block was dropped")
    }
}

impl<S: BlockLayout, M: BlockMetadata> DerefMut for MutableBlock<S, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.block.as_mut().expect("block was dropped")
    }
}

impl<S: BlockLayout, M: BlockMetadata> Deref for ImmutableBlock<S, M> {
    type Target = Block<S, M>;
    fn deref(&self) -> &Self::Target {
        self.block
            .as_ref()
            .block
            .as_ref()
            .expect("block was dropped")
    }
}

struct State<S: BlockLayout, M: BlockMetadata> {
    active: ActiveBlockPool<S, M>,
    inactive: InactiveBlockPool<S, M>,
    registry: BlockRegistry,
    return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, M>>,
    event_manager: Arc<dyn EventManager>,
}

struct ProgressEngine<S: BlockLayout, M: BlockMetadata> {
    event_manager: Arc<dyn EventManager>,
    priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, M>>,
    ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, M>>,
    cancel_token: CancellationToken,
    state: State<S, M>,
    return_rx: tokio::sync::mpsc::UnboundedReceiver<Block<S, M>>,
}

#[cfg(test)]
mod tests {
    use crate::block_manager::block::BlockExt;

    use super::super::block::{BasicMetadata, Blocks};
    use super::super::layout::tests::setup_layout;
    use super::*;

    #[tokio::test]
    async fn test_default_runtime_async() {
        let runtime = Runtime::default();
        assert!(matches!(runtime, Runtime::Handle(_)));
    }

    #[test]
    fn test_default_runtime_blocking() {
        let runtime = Runtime::default();
        assert!(matches!(runtime, Runtime::Runtime(_)));
    }

    #[tokio::test]
    async fn test_block_pool_state() {
        let layout = setup_layout(None).unwrap();
        let blocks = Blocks::<_, BasicMetadata>::new(layout)
            .unwrap()
            .into_blocks()
            .unwrap();

        let (_pool, mut progress) = BlockPool::builder()
            .blocks(blocks)
            .build_with_progress_engine()
            .unwrap();

        assert_eq!(progress.state.inactive.available_blocks(), 7);

        let blocks = progress.state.allocate_blocks(1).unwrap();
        assert_eq!(progress.state.inactive.available_blocks(), 6);
        assert_eq!(blocks.len(), 1);

        drop(blocks);
        progress.step().await;
        assert_eq!(progress.state.inactive.available_blocks(), 7);

        let mut blocks = progress.state.allocate_blocks(1).unwrap();
        assert_eq!(progress.state.inactive.available_blocks(), 6);
        assert_eq!(blocks.len(), 1);

        let mut block = blocks.pop().unwrap();

        block.initialize_sequence(1337).unwrap();
        block.add_token(1).unwrap();
        block.add_token(2).unwrap();
        block.add_token(3).unwrap();
        block.add_token(4).unwrap();

        assert!(block.add_token(5).is_err());
    }

    #[tokio::test]
    async fn test_block_pool() {
        let layout = setup_layout(None).unwrap();
        let blocks = Blocks::<_, BasicMetadata>::new(layout)
            .unwrap()
            .into_blocks()
            .unwrap();

        let (pool, mut progress) = BlockPool::builder()
            .blocks(blocks)
            .build_with_progress_engine()
            .unwrap();

        assert_eq!(progress.state.inactive.available_blocks(), 7);

        let pool_clone = pool.clone();
        let allocate_1_block =
            tokio::spawn(async move { pool_clone.allocate_blocks(1).await.unwrap() });
        progress.step().await;

        let blocks = allocate_1_block.await.unwrap();
        assert_eq!(progress.state.inactive.available_blocks(), 6);
        assert_eq!(blocks.len(), 1);

        // drop the single block
        drop(blocks);

        // check before and after the progress engine step
        assert_eq!(progress.state.inactive.available_blocks(), 6);
        progress.step().await;
        assert_eq!(progress.state.inactive.available_blocks(), 7);
    }
}
