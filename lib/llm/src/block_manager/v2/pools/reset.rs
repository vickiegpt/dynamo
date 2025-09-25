// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reset Pool

use super::{Block, BlockAllocator, BlockMetadata, MutableBlock, Reset};
use parking_lot::Mutex;
use std::{collections::VecDeque, sync::Arc};

#[derive(Clone)]
pub struct ResetPool<T> {
    block_allocator: Arc<Mutex<dyn BlockAllocator<T> + Send + Sync>>,
    return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
}

impl<T: BlockMetadata> ResetPool<T> {
    pub fn new(blocks: Vec<Block<T, Reset>>) -> Self {
        let allocator = DequeBlockAllocator::new();
        Self::from_block_allocator(allocator, blocks)
    }

    pub fn from_block_allocator(
        allocator: impl BlockAllocator<T> + Send + Sync + 'static,
        blocks: Vec<Block<T, Reset>>,
    ) -> Self {
        assert_eq!(blocks.len(), 0, "BlockAllocator must be empty");

        // validate that the blocks ids are monotonically increasing starting at 0
        for (i, block) in blocks.iter().enumerate() {
            if block.block_id() != i as u64 {
                panic!("Block ids must be monotonically increasing starting at 0");
            }
        }

        let block_allocator = Arc::new(Mutex::new(allocator));

        let allocator = block_allocator.clone();
        let return_fn = Arc::new(move |block: Block<T, Reset>| {
            allocator.lock().insert(block);
        });

        Self {
            block_allocator,
            return_fn,
        }
    }

    pub fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        let mut allocator = self.block_allocator.lock();
        if allocator.len() < count {
            return None;
        }

        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            blocks.push(MutableBlock::new(
                allocator.pop().unwrap(),
                self.return_fn.clone(),
            ));
        }

        Some(blocks)
    }

    pub fn try_allocate_blocks(&self, count: usize) -> Vec<MutableBlock<T>> {
        let mut blocks = Vec::with_capacity(count);
        let mut allocator = self.block_allocator.lock();
        let available_count = std::cmp::min(count, allocator.len());

        for _ in 0..available_count {
            blocks.push(MutableBlock::new(
                allocator.pop().unwrap(),
                self.return_fn.clone(),
            ));
        }

        blocks
    }

    /// Get the number of available blocks
    pub fn available_blocks(&self) -> usize {
        self.block_allocator.lock().len()
    }

    pub fn len(&self) -> usize {
        self.block_allocator.lock().len()
    }

    /// Create a return function for blocks to return to this pool
    /// This allows other pools to create MutableBlocks that return here
    pub(crate) fn return_fn(&self) -> Arc<dyn Fn(Block<T, Reset>) + Send + Sync> {
        self.return_fn.clone()
    }
}

#[derive(Debug)]
pub struct DequeBlockAllocator<T: BlockMetadata> {
    blocks: VecDeque<Block<T, Reset>>,
}

impl<T: BlockMetadata> DequeBlockAllocator<T> {
    pub fn new() -> Self {
        Self {
            blocks: VecDeque::new(),
        }
    }
}

impl<T: BlockMetadata> BlockAllocator<T> for DequeBlockAllocator<T> {
    fn insert(&mut self, block: Block<T, Reset>) {
        self.blocks.push_back(block);
    }

    fn pop(&mut self) -> Option<Block<T, Reset>> {
        self.blocks.pop_front()
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }
}
