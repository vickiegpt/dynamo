use super::*;
use crate::block_manager::pool::{BlockPoolError, BlockPoolResult, ResetBlocksResponse};
use std::sync::{Arc, Mutex};

/// Direct access to the block pool state, bypassing the progress engine.
/// This provides synchronous access for performance-critical paths.
///
/// Note: This is a simplified initial implementation that provides basic
/// direct access without complex retry logic.
pub struct DirectAccess<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    state: Arc<Mutex<State<S, L, M>>>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Clone for DirectAccess<S, L, M> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> DirectAccess<S, L, M> {
    pub fn new(state: Arc<Mutex<State<S, L, M>>>) -> Self {
        Self { state }
    }

    /// Get a reference to the state - used for testing
    #[allow(dead_code)]
    pub(crate) fn state(&self) -> Arc<Mutex<State<S, L, M>>> {
        self.state.clone()
    }

    /// Allocate a set of blocks from the pool.
    pub fn allocate_blocks(&self, count: usize) -> BlockPoolResult<Vec<MutableBlock<S, L, M>>> {
        let mut state = self.state.lock().unwrap();
        state.allocate_blocks(count)
    }

    /// Add blocks to the inactive pool.
    pub fn add_blocks(&self, blocks: Vec<Block<S, L, M>>) {
        let mut state = self.state.lock().unwrap();
        state.inactive.add_blocks(blocks);
    }

    /// Try to return a block to the pool.
    pub fn try_return_block(&self, block: Vec<Block<S, L, M>>) -> BlockPoolResult<()> {
        if block.is_empty() {
            return Ok(());
        }

        let mut state = self.state.lock().unwrap();
        for b in block {
            state.return_block(b);
        }

        Ok(())
    }

    /// Get the current status of the block pool.
    pub fn status(&self) -> Result<BlockPoolStatus, BlockPoolError> {
        let state = self.state.lock().unwrap();
        Ok(state.status())
    }

    /// Reset the pool, returning all blocks to the inactive state.
    pub fn reset(&self) -> Result<(), BlockPoolError> {
        let mut state = self.state.lock().unwrap();
        state.inactive.reset()
    }

    /// Reset specific blocks by sequence hash.
    pub fn reset_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<ResetBlocksResponse, BlockPoolError> {
        let mut state = self.state.lock().unwrap();
        Ok(state.try_reset_blocks(sequence_hashes))
    }
}
