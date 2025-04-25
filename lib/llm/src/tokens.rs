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

use crate::kv_router::indexer::compute_hash_v2;
use bytemuck::cast_slice;
use derive_getters::{Dissolve, Getters};
use rayon::prelude::*;

/// A token is a 32-bit unsigned integer.
pub type Token = u32;

/// A salt is a vector of bytes.
pub type Salt = Vec<u8>;

/// A hash of the salt computed from [compute_hash_v2] with a seed of 0.
pub type SaltHash = u64;

/// A hash of the only the tokens within a block computed from [compute_hash_v2] using the salt hash as the seed.
pub type BlockHash = u64;

/// A sequence aware hash that combines the previous block's sequence hash with the current block's hash.
pub type SequenceHash = u64;

/// A collection of tokens.
#[derive(Debug, Clone, Dissolve, Default)]
pub struct Tokens(Vec<Token>);

impl AsRef<[Token]> for Tokens {
    fn as_ref(&self) -> &[Token] {
        &self.0
    }
}

impl std::ops::Deref for Tokens {
    type Target = [Token];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::borrow::Borrow<[Token]> for Tokens {
    fn borrow(&self) -> &[Token] {
        &self.0
    }
}

impl From<Vec<Token>> for Tokens {
    fn from(tokens: Vec<Token>) -> Self {
        Tokens(tokens)
    }
}

impl From<&[Token]> for Tokens {
    fn from(tokens: &[Token]) -> Self {
        Tokens(tokens.to_vec())
    }
}

impl From<Vec<i32>> for Tokens {
    fn from(tokens: Vec<i32>) -> Self {
        Tokens(tokens.into_iter().map(|t| t as u32).collect())
    }
}

impl From<&[i32]> for Tokens {
    fn from(tokens: &[i32]) -> Self {
        Tokens(tokens.iter().map(|&t| t as u32).collect())
    }
}

impl From<Tokens> for Vec<Token> {
    fn from(tokens: Tokens) -> Self {
        tokens.0
    }
}

impl Tokens {
    pub fn into_sequence(
        self,
        block_size: usize,
        salt_hash: Option<SaltHash>,
    ) -> TokenBlockSequence {
        TokenBlockSequence::new(self, block_size, salt_hash)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TokenBlockError {
    #[error("TokenBlock is full")]
    Full,

    #[error("TokenBlock is incomplete")]
    Incomplete,

    #[error("TokenBlock is empty")]
    Empty,

    #[error("TokenBlock has insufficient tokens")]
    InsufficientTokens,
}

#[derive(Debug)]
pub struct PartialTokenBlock {
    tokens: Tokens,
    block_size: usize,
    salt_hash: SaltHash,
    parent_sequence_hash: Option<SequenceHash>,
}

impl PartialTokenBlock {
    pub fn create_sequence_root(block_size: usize, salt_hash: SaltHash) -> Self {
        Self {
            tokens: Tokens::default(),
            block_size,
            salt_hash,
            parent_sequence_hash: None,
        }
    }

    /// Attempt to push a token onto the block, if the block is full, return an error
    pub fn push_token(&mut self, token: Token) -> Result<(), TokenBlockError> {
        if self.tokens.0.len() >= self.block_size {
            return Err(TokenBlockError::Full);
        }

        self.tokens.0.push(token);
        Ok(())
    }

    /// Attempt to push a slice of tokens onto the block, when the block is full,
    /// a vec of remaining tokens is returned
    pub fn push_tokens(&mut self, tokens: Tokens) -> Tokens {
        let remaining_space = self.block_size - self.tokens.0.len();

        if remaining_space == 0 {
            return tokens;
        }

        if tokens.0.len() <= remaining_space {
            // All tokens fit in the block
            self.tokens.0.extend(tokens.0);
            return Tokens::default();
        } else {
            // Only some tokens fit, return the rest
            let (to_add, remaining) = tokens.0.split_at(remaining_space);
            self.tokens.0.extend_from_slice(to_add);
            return Tokens(remaining.to_vec());
        }
    }

    /// Attempt to pop a token from the end of the block, if the block is empty, return an error
    pub fn pop_token(&mut self) -> Result<(), TokenBlockError> {
        if self.tokens.0.is_empty() {
            return Err(TokenBlockError::Empty);
        }

        self.tokens.0.pop();
        Ok(())
    }

    /// Attempt to pop a specified number of tokens from the end of the block,
    /// if the block has insufficient tokens, return an error
    pub fn pop_tokens(&mut self, count: usize) -> Result<(), TokenBlockError> {
        if self.tokens.0.len() < count {
            return Err(TokenBlockError::InsufficientTokens);
        }

        self.tokens.0.truncate(self.tokens.0.len() - count);
        Ok(())
    }
    /// Commits the current [PartialTokenBlock] to generate a new [TokenBlock], then
    /// resets the current [PartialTokenBlock] to be a downstream block of the new [TokenBlock].
    pub fn commit(&mut self) -> Result<TokenBlock, TokenBlockError> {
        let len = self.tokens.0.len();
        if len < self.block_size {
            return Err(TokenBlockError::Incomplete);
        }

        assert_eq!(self.tokens.0.len(), self.block_size);
        let tokens = std::mem::take(&mut self.tokens);

        let chunk = TokenBlockChunk::new(tokens, self.salt_hash);
        let block = TokenBlock::from_chunk(chunk, self.parent_sequence_hash);
        *self = block.next_block();

        Ok(block)
    }

    /// Get the number of tokens remaining in the block
    pub fn remaining(&self) -> usize {
        self.block_size - self.tokens.0.len()
    }

    /// Get the number of tokens in the block
    pub fn len(&self) -> usize {
        self.tokens.0.len()
    }

    /// Get the tokens in the block
    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }
}

impl std::ops::Deref for PartialTokenBlock {
    type Target = Tokens;

    fn deref(&self) -> &Self::Target {
        &self.tokens
    }
}

/// This is an intermediate structure used to compute the hash of a block.
/// It is used to compute the chunks independently and possibly in parallel; however, does not
/// provide the sequence hash.
struct TokenBlockChunk {
    tokens: Tokens,
    salt_hash: SaltHash,
    block_hash: BlockHash,
}

impl TokenBlockChunk {
    fn new(tokens: Tokens, salt_hash: SaltHash) -> Self {
        let block_hash = compute_hash_v2(cast_slice(&tokens), salt_hash);
        Self {
            tokens,
            salt_hash,
            block_hash,
        }
    }

    fn from_tokens(tokens: &[Token], salt_hash: SaltHash) -> Self {
        let block_hash = compute_hash_v2(cast_slice(tokens), salt_hash);
        Self {
            tokens: tokens.into(),
            salt_hash,
            block_hash,
        }
    }
}

#[derive(Debug, Clone, Getters, Default)]
pub struct TokenBlock {
    tokens: Tokens,

    #[getter(copy)]
    salt_hash: u64,

    #[getter(copy)]
    block_hash: BlockHash,

    #[getter(copy)]
    sequence_hash: SequenceHash,

    #[getter(copy)]
    parent_sequence_hash: Option<SequenceHash>,
}

impl TokenBlock {
    pub fn next_block(&self) -> PartialTokenBlock {
        PartialTokenBlock {
            tokens: Tokens::default(),
            block_size: self.tokens.len(),
            salt_hash: self.salt_hash,
            parent_sequence_hash: Some(self.sequence_hash),
        }
    }

    fn from_chunk(chunk: TokenBlockChunk, parent_sequence_hash: Option<SequenceHash>) -> Self {
        match parent_sequence_hash {
            Some(parent) => {
                let sequence_hash = compute_hash_v2(
                    bytemuck::cast_slice(&[parent, chunk.block_hash]),
                    chunk.salt_hash,
                );
                Self {
                    tokens: chunk.tokens,
                    salt_hash: chunk.salt_hash,
                    block_hash: chunk.block_hash,
                    sequence_hash,
                    parent_sequence_hash: Some(parent),
                }
            }
            None => Self {
                tokens: chunk.tokens,
                salt_hash: chunk.salt_hash,
                block_hash: chunk.block_hash,
                sequence_hash: chunk.block_hash,
                parent_sequence_hash: None,
            },
        }
    }
}

/// Structure that holds a sequence of tokens broken into blocks where the blocks are hashed.
///
/// The block hashes computed are designed to be used externally from the LLM backend to provide uniqueness which must also
/// account for the differences in the model architecture, model weights, associated PEFT used to generate the sequence, etc.
///
/// To account for these differences, the salt hash is used as the seed for the hash function. One might choose to serialize some
/// metadata about the model, PEFT, etc, convert it to a byte slice using `serde_json::to_vec` then compute a u64 hash from that object
/// which can be used as the `salt_hash` for the [TokenBlockSequence].
///
/// There are two critical hashes:
/// - `block_hash`: a hash computed from only the local tokens within the block seeding the hashing function with the `salt_hash`
/// - `sequence_hash`: a hash computed from the previous block's `sequence_hash` and the current block's `block_hash` using the `salt_hash` as the seed
#[derive(Debug)]
pub struct TokenBlockSequence {
    blocks: Vec<TokenBlock>,
    current_block: PartialTokenBlock,
    salt_hash: SaltHash,
}

impl TokenBlockSequence {
    pub fn new(tokens: Tokens, block_size: usize, salt_hash: Option<SaltHash>) -> Self {
        let salt_hash = salt_hash.unwrap_or(0);
        let (blocks, current_block) = Self::split_tokens(tokens, block_size, salt_hash);

        Self {
            blocks,
            current_block,
            salt_hash,
        }
    }

    pub fn push_token(&mut self, token: Token) -> Option<&TokenBlock> {
        let remaining = self.current_block.remaining();
        assert_ne!(self.current_block.remaining(), 0);
        self.current_block
            .push_token(token)
            .expect("TokenBlock is full");

        // 1 indicates that the block is now full after the push
        if remaining == 1 {
            let token_block = self
                .current_block
                .commit()
                .expect("TokenBlock is not complete");
            self.blocks.push(token_block);
            self.blocks.last()
        } else {
            None
        }
    }

    pub fn blocks(&self) -> &[TokenBlock] {
        &self.blocks
    }

    pub fn last_complete_block(&self) -> Option<&TokenBlock> {
        self.blocks.last()
    }
    pub fn current_block(&self) -> &PartialTokenBlock {
        &self.current_block
    }

    pub fn into_parts(self) -> (Vec<TokenBlock>, PartialTokenBlock) {
        (self.blocks, self.current_block)
    }

    /// Get the salt for the sequence
    pub fn salt_hash(&self) -> SaltHash {
        self.salt_hash
    }

    pub fn split_tokens(
        tokens: Tokens,
        block_size: usize,
        salt_hash: u64,
    ) -> (Vec<TokenBlock>, PartialTokenBlock) {
        let chunks: Vec<TokenBlockChunk> = tokens
            .as_ref()
            .par_chunks_exact(block_size)
            .map(|chunk| TokenBlockChunk::from_tokens(chunk, salt_hash))
            .collect();

        let mut result_blocks = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            // Get the sequence hash of the previous block, if it exists
            let last_sequence_hash = result_blocks.last().map(|b: &TokenBlock| b.sequence_hash());

            // Use the constructor which encapsulates the sequence hash logic
            let new_block = TokenBlock::from_chunk(chunk, last_sequence_hash);

            // Push the new block to the result
            result_blocks.push(new_block);
        }

        let remainder = tokens.chunks_exact(block_size).remainder();

        let current_block = PartialTokenBlock {
            tokens: remainder.into(),
            block_size,
            salt_hash,
            // The parent sequence hash for the next partial block is the hash of the last full block
            parent_sequence_hash: result_blocks.last().map(|b| b.sequence_hash()),
        };

        (result_blocks, current_block)
    }
}

impl PartialEq<Vec<Token>> for Tokens {
    fn eq(&self, other: &Vec<Token>) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Tokens> for Vec<Token> {
    fn eq(&self, other: &Tokens) -> bool {
        *self == other.0
    }
}

impl PartialEq<[Token]> for Tokens {
    fn eq(&self, other: &[Token]) -> bool {
        self.0.as_slice() == other
    }
}

impl PartialEq<Tokens> for &[Token] {
    fn eq(&self, other: &Tokens) -> bool {
        *self == other.0.as_slice()
    }
}

impl PartialEq<Vec<Token>> for &Tokens {
    fn eq(&self, other: &Vec<Token>) -> bool {
        self.0 == *other
    }
}

impl<'a> PartialEq<&'a Tokens> for Vec<Token> {
    fn eq(&self, other: &&'a Tokens) -> bool {
        *self == other.0
    }
}

impl PartialEq<[Token]> for &Tokens {
    fn eq(&self, other: &[Token]) -> bool {
        self.0.as_slice() == other
    }
}

impl<'a> PartialEq<&'a [Token]> for Tokens {
    fn eq(&self, other: &&'a [Token]) -> bool {
        self.0.as_slice() == *other
    }
}

impl PartialEq for Tokens {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Tokens {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_slice_operations() {
        let tokens = Tokens(vec![1, 2, 3, 4, 5]);

        // Test AsRef<[Token]>
        let slice: &[Token] = tokens.as_ref();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        // Test Deref
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], 1);
        assert_eq!(tokens[4], 5);

        // Test iteration
        let sum: u32 = tokens.iter().sum();
        assert_eq!(sum, 15);

        // Test slicing
        let slice = &tokens[1..4];
        assert_eq!(slice, &[2, 3, 4]);

        // Test Borrow
        let borrowed: &[Token] = std::borrow::Borrow::borrow(&tokens);
        assert_eq!(borrowed, &[1, 2, 3, 4, 5]);

        // Test with functions that accept &[Token]
        fn takes_slice(slice: &[Token]) -> usize {
            slice.len()
        }

        assert_eq!(takes_slice(&tokens), 5);
    }

    #[test]
    fn test_tokens_conversions() {
        // Test From<Vec<Token>> for Tokens
        let vec = vec![1, 2, 3, 4, 5];
        let tokens: Tokens = vec.clone().into();
        assert_eq!(tokens.0, vec);

        // Test Into<Vec<Token>> for Tokens
        let tokens = Tokens(vec![6, 7, 8, 9, 10]);
        let vec: Vec<Token> = tokens.into();
        assert_eq!(vec, vec![6, 7, 8, 9, 10]);

        // Test From<&[Token]> for Tokens
        let slice: &[Token] = &[11, 12, 13];
        let tokens: Tokens = slice.into();
        assert_eq!(tokens.0, vec![11, 12, 13]);

        // Test From<Vec<i32>> for Tokens
        let i32_values = vec![100_i32, 200_i32, 300_i32];
        let tokens: Tokens = i32_values.into();
        assert_eq!(tokens.0, vec![100, 200, 300]);

        // Test From<&[i32]> for Tokens
        let i32_slice: &[i32] = &[400_i32, 500_i32, 600_i32];
        let tokens: Tokens = i32_slice.into();
        assert_eq!(tokens.0, vec![400, 500, 600]);
    }

    #[test]
    fn test_tokens_blocks() {
        let tokens = Tokens(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // NOTE: 1337 was the original seed, so we are temporarily using that here to prove the logic has not changed
        let sequence = TokenBlockSequence::new(tokens, 4, Some(1337_u64));

        assert_eq!(sequence.blocks().len(), 2);
        assert_eq!(sequence.current_block().len(), 2);

        assert_eq!(sequence.blocks()[0].tokens(), vec![1, 2, 3, 4]);
        assert_eq!(sequence.blocks()[0].block_hash(), 14643705804678351452);
        assert_eq!(sequence.blocks()[0].sequence_hash(), 14643705804678351452);
        println!("blocks[0]: {:?}", sequence.blocks()[0]);

        assert_eq!(sequence.blocks()[1].tokens(), vec![5, 6, 7, 8]);
        assert_eq!(sequence.blocks()[1].block_hash(), 16777012769546811212);
        assert_eq!(sequence.blocks()[1].sequence_hash(), 4945711292740353085);
        println!("blocks[1]: {:?}", sequence.blocks()[1]);

        assert_eq!(sequence.current_block().tokens(), vec![9, 10]);

        let mut sequence = sequence;

        let new_block = sequence.push_token(11);
        assert!(new_block.is_none());
        assert_eq!(sequence.blocks().len(), 2);

        let new_block = sequence.push_token(12);
        assert!(new_block.is_some());
        assert_eq!(sequence.blocks().len(), 3);
        assert_eq!(sequence.current_block().tokens().len(), 0);
        println!("blocks[2]: {:?}", sequence.blocks()[2]);

        let (_blocks, mut current_block) = sequence.into_parts();

        let new_block = current_block.push_token(13);
        assert!(new_block.is_ok());
        assert_eq!(current_block.tokens().len(), 1);

        let new_block = current_block.push_token(14);
        assert!(new_block.is_ok());
        assert_eq!(current_block.tokens().len(), 2);

        let new_block = current_block.push_token(15);
        assert!(new_block.is_ok());
        assert_eq!(current_block.tokens().len(), 3);

        let new_block = current_block.push_token(16);
        assert!(new_block.is_ok());
        assert_eq!(current_block.tokens().len(), 4);

        let new_block = current_block.commit().expect("TokenBlock is not complete");
        assert_eq!(new_block.tokens().len(), 4);
        assert_eq!(
            Some(new_block.sequence_hash()),
            current_block.parent_sequence_hash
        );
        assert_eq!(current_block.tokens().len(), 0);
        assert_eq!(current_block.block_size, 4);
        assert_eq!(current_block.salt_hash, 1337);
    }

    #[test]
    fn test_build_sequence() {
        let mut sequence = TokenBlockSequence::new(Tokens::default(), 4, Some(1337_u64));

        assert_eq!(sequence.blocks().len(), 0);
        assert_eq!(sequence.current_block().tokens().len(), 0);

        sequence.push_token(1);
        assert_eq!(sequence.blocks().len(), 0);
        assert_eq!(sequence.current_block().tokens().len(), 1);

        sequence.push_token(2);
        assert_eq!(sequence.blocks().len(), 0);
        assert_eq!(sequence.current_block().tokens().len(), 2);

        sequence.push_token(3);
        assert_eq!(sequence.blocks().len(), 0);
        assert_eq!(sequence.current_block().tokens().len(), 3);

        sequence.push_token(4);
        assert_eq!(sequence.blocks().len(), 1);
        assert_eq!(sequence.current_block().tokens().len(), 0);
        assert_eq!(sequence.blocks()[0].sequence_hash(), 14643705804678351452);

        sequence.push_token(5);
        assert_eq!(sequence.blocks().len(), 1);
        assert_eq!(sequence.current_block().tokens().len(), 1);

        sequence.push_token(6);
        assert_eq!(sequence.blocks().len(), 1);
        assert_eq!(sequence.current_block().tokens().len(), 2);

        sequence.push_token(7);
        assert_eq!(sequence.blocks().len(), 1);
        assert_eq!(sequence.current_block().tokens().len(), 3);

        sequence.push_token(8);
        assert_eq!(sequence.blocks().len(), 2);
        assert_eq!(sequence.current_block().tokens().len(), 0);
        assert_eq!(sequence.blocks()[1].sequence_hash(), 4945711292740353085);
    }

    #[test]
    fn test_push_tokens() {
        let mut partial = PartialTokenBlock::create_sequence_root(4, 1337);

        let tokens = Tokens(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let remaining = partial.push_tokens(tokens);
        assert_eq!(partial.tokens().len(), 4);
        assert_eq!(remaining.len(), 6);
    }
}
