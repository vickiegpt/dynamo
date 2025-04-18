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

use derive_getters::Getters;

use crate::{
    block_manager::events::RegistrationHandle,
    tokens::{BlockHash, SequenceHash, TokenBlock, TokenBlockSequence},
};

#[derive(Debug)]
pub enum BlockState {
    Reset,
    Partial(PartialState),
    Complete(CompleteState),
    Registered(RegisteredState),
}

#[derive(Debug)]
pub struct PartialState {
    pub sequence: TokenBlockSequence,
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

#[derive(Debug, Getters)]
pub struct RegisteredState {
    #[getter(copy)]
    block_hash: BlockHash,

    #[getter(copy)]
    sequence_hash: SequenceHash,

    #[getter(skip)]
    registration_handle: RegistrationHandle,
}

impl RegisteredState {
    pub fn new(complete_state: &CompleteState, registration_handle: RegistrationHandle) -> Self {
        Self {
            block_hash: complete_state.token_block().block_hash(),
            sequence_hash: complete_state.token_block().sequence_hash(),
            registration_handle,
        }
    }

    pub fn is_armed(&self) -> bool {
        self.registration_handle.is_armed()
    }
}
