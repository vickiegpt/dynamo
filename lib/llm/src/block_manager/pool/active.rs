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

use super::*;

/// Manages active blocks being used by sequences
pub struct ActiveBlockPool<S: Storage, M: BlockMetadata> {
    pub(super) map: HashMap<SequenceHash, Weak<MutableBlock<S, M>>>,
}

impl<S: Storage, M: BlockMetadata> ActiveBlockPool<S, M> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Inserts a weak reference to a block into the active pool map.
    ///
    /// This is typically used when a block is matched from the inactive pool
    /// and needs to be tracked as potentially active.
    ///
    /// # Arguments
    ///
    /// * `sequence_hash` - The sequence hash of the block.
    /// * `weak_block_ref` - A weak reference ([`Weak<MutableBlock<S, M>>`]) to the block.
    pub fn insert_weak_block_ref(
        &mut self,
        sequence_hash: SequenceHash,
        weak_block_ref: Weak<MutableBlock<S, M>>,
    ) {
        self.map.insert(sequence_hash, weak_block_ref);
    }

    pub fn register(
        &mut self,
        block: MutableBlock<S, M>,
    ) -> Result<ImmutableBlock<S, M>, BlockPoolError> {
        let sequence_hash = block.sequence_hash().map_err(|_| {
            BlockPoolError::InvalidMutableBlock("block has no sequence hash".to_string())
        })?;

        let shared = Arc::new(block);

        match self.map.entry(sequence_hash) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let weak = entry.get();
                if let Some(arc) = weak.upgrade() {
                    Ok(ImmutableBlock { block: arc })
                } else {
                    // Weak reference is no longer alive, update it in the map
                    entry.insert(Arc::downgrade(&shared));
                    Ok(ImmutableBlock { block: shared })
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Arc::downgrade(&shared));
                Ok(ImmutableBlock { block: shared })
            }
        }
    }

    pub fn remove(&mut self, block: &mut Block<S, M>) {
        if let Ok(sequence_hash) = block.sequence_hash() {
            if let Some(weak) = self.map.get(&sequence_hash) {
                if let Some(_arc) = weak.upgrade() {
                    block.reset();
                    return;
                }
            }
            self.map.remove(&sequence_hash);
        }
    }

    pub fn match_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Option<ImmutableBlock<S, M>> {
        if let Some(weak) = self.map.get(&sequence_hash) {
            if let Some(arc) = weak.upgrade() {
                Some(ImmutableBlock { block: arc })
            } else {
                // Weak reference is no longer alive, remove it from the map
                self.map.remove(&sequence_hash);
                None
            }
        } else {
            None
        }
    }
}
