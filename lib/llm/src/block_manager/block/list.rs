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

//! Module for managing a list of blocks.

use crate::tokens::SequenceHash;

use super::{BlockIdentifier, BlockMetadata, ImmutableBlock, MutableBlock, Storage};

pub type BlockID = usize;

#[derive(Debug)]
pub enum BlockListContainer<S: Storage, M: BlockMetadata> {
    Immutable(Vec<ImmutableBlock<S, M>>),
    Mutable(Vec<MutableBlock<S, M>>),
}

#[derive(Debug, thiserror::Error)]
pub enum BlockListError {
    #[error("Invalid block type")]
    InvalidBlockType,
}

/// A list of blocks.
///
/// This is a wrapper around a vector of blocks that provides a more convenient interface for
/// managing a list of blocks.
///
/// The blocks are dropped in reverse order, so that the last block in the list is the first one to be
/// dropped. The reverse drop is important so the block pools see the tail blocks dropped first, then
/// backwards to the head/root blocks.
#[derive(Debug)]
pub struct BlockList<S: Storage, M: BlockMetadata> {
    block_ids: Vec<BlockID>,
    blocks: Option<BlockListContainer<S, M>>,
    block_hashes: Option<Vec<SequenceHash>>,
}

impl<S, M> BlockList<S, M>
where
    S: Storage,
    M: BlockMetadata,
{
    /// Creates a new, empty `BlockList`.
    pub fn new() -> Self {
        Self {
            block_ids: Vec::new(),
            blocks: None,
            block_hashes: None,
        }
    }

    /// Creates a new `BlockList` from an existing vector of blocks.
    pub fn from_matched_blocks(blocks: Vec<ImmutableBlock<S, M>>) -> Self {
        let block_ids = blocks.iter().map(|b| b.block_id()).collect();
        let block_hashes = blocks
            .iter()
            .map(|b| {
                b.sequence_hash()
                    .expect("immutable block should return sequence hash")
            })
            .collect();
        Self {
            block_ids,
            blocks: Some(BlockListContainer::Immutable(blocks)),
            block_hashes: Some(block_hashes),
        }
    }

    pub fn from_mutable_blocks(blocks: Vec<MutableBlock<S, M>>) -> Self {
        let block_ids = blocks.iter().map(|b| b.block_id()).collect();
        Self {
            block_ids,
            blocks: Some(BlockListContainer::Mutable(blocks)),
            block_hashes: None,
        }
    }

    pub fn clear(&mut self) {
        if let Some(BlockListContainer::Immutable(blocks)) = &mut self.blocks {
            blocks.reverse();
            blocks.clear();
        }
    }

    pub fn add_immutable_block(
        &mut self,
        block: ImmutableBlock<S, M>,
    ) -> Result<(), BlockListError> {
        match &mut self.blocks {
            Some(BlockListContainer::Immutable(blocks)) => {
                self.block_ids.push(block.block_id());

                self.block_hashes
                    .as_mut()
                    .expect("block_hashes should be initialized")
                    .push(
                        block
                            .sequence_hash()
                            .expect("sequence hash should be valid"),
                    );
                blocks.push(block);
            }
            Some(BlockListContainer::Mutable(_blocks)) => {
                return Err(BlockListError::InvalidBlockType);
            }
            None => {
                self.block_ids.push(block.block_id());
                self.block_hashes = Some(vec![block
                    .sequence_hash()
                    .expect("sequence hash should be valid")]);
                self.blocks = Some(BlockListContainer::Immutable(vec![block]));
            }
        }

        Ok(())
    }

    pub fn add_mutable_block(&mut self, block: MutableBlock<S, M>) -> Result<(), BlockListError> {
        match &mut self.blocks {
            Some(BlockListContainer::Immutable(_blocks)) => {
                return Err(BlockListError::InvalidBlockType);
            }
            Some(BlockListContainer::Mutable(blocks)) => {
                self.block_ids.push(block.block_id());
                blocks.push(block);
            }
            None => {
                self.block_ids.push(block.block_id());
                self.blocks = Some(BlockListContainer::Mutable(vec![block]));
            }
        }

        Ok(())
    }

    pub fn block_ids(&self) -> &[BlockID] {
        &self.block_ids
    }

    pub fn blocks(&self) -> Option<&BlockListContainer<S, M>> {
        self.blocks.as_ref()
    }

    pub fn blocks_mut(&mut self) -> Option<&mut BlockListContainer<S, M>> {
        self.blocks.as_mut()
    }

    pub fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.block_ids.len()
    }

    pub fn has_blocks(&self) -> bool {
        self.blocks.is_some()
    }

    pub fn has_mutable_blocks(&self) -> bool {
        matches!(self.blocks, Some(BlockListContainer::Mutable(_)))
    }

    pub fn has_immutable_blocks(&self) -> bool {
        matches!(self.blocks, Some(BlockListContainer::Immutable(_)))
    }
}

impl<S, M> Default for BlockList<S, M>
where
    S: Storage,
    M: BlockMetadata,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S, M> From<Vec<ImmutableBlock<S, M>>> for BlockList<S, M>
where
    S: Storage,
    M: BlockMetadata,
{
    fn from(blocks: Vec<ImmutableBlock<S, M>>) -> Self {
        Self::from_matched_blocks(blocks)
    }
}

impl<S, M> From<Vec<MutableBlock<S, M>>> for BlockList<S, M>
where
    S: Storage,
    M: BlockMetadata,
{
    fn from(blocks: Vec<MutableBlock<S, M>>) -> Self {
        Self::from_mutable_blocks(blocks)
    }
}

impl<S, M> Drop for BlockList<S, M>
where
    S: Storage,
    M: BlockMetadata,
{
    fn drop(&mut self) {
        self.clear();
    }
}
