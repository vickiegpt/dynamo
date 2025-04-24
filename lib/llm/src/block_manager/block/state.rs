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

use std::sync::Arc;

use derive_getters::Getters;

use super::registry::RegistrationHandle;
use super::Result;
use crate::tokens::{PartialTokenBlock, SaltHash, Token, TokenBlock};

#[derive(Debug, thiserror::Error)]
#[error("Block state is invalid: {0}")]
pub struct BlockStateInvalid(String);

#[derive(Debug)]
pub enum BlockState {
    Reset,
    Partial(PartialState),
    Complete(CompleteState),
    Registered(Arc<RegistrationHandle>),
}

impl BlockState {
    pub fn initialize_sequence(
        &mut self,
        page_size: usize,
        salt_hash: SaltHash,
    ) -> Result<(), BlockStateInvalid> {
        if !matches!(self, BlockState::Reset) {
            return Err(BlockStateInvalid("Block is not reset".to_string()));
        }

        let block = PartialTokenBlock::create_sequence_root(page_size, salt_hash);
        *self = BlockState::Partial(PartialState::new(block));
        Ok(())
    }

    pub fn add_token(&mut self, token: &Token) -> Result<()> {
        match self {
            BlockState::Partial(state) => {
                return Ok(state.add_token(token)?);
            }
            _ => {
                return Err(BlockStateInvalid("Block is not partial".to_string()))?;
            }
        }
    }
}

#[derive(Debug)]
pub struct PartialState {
    block: PartialTokenBlock,
}

impl PartialState {
    pub fn new(block: PartialTokenBlock) -> Self {
        Self { block }
    }

    pub fn add_token(&mut self, token: &Token) -> Result<()> {
        Ok(self.block.push_token(*token)?)
    }
}

#[derive(Debug, Getters)]
pub struct CompleteState {
    token_block: TokenBlock,
}

impl CompleteState {
    pub fn new(token_block: TokenBlock) -> Self {
        Self { token_block }
    }
}
