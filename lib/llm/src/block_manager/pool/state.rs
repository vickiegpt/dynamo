use super::*;

impl<S: Storage, M: BlockMetadata> State<S, M> {
    pub fn new(events: Arc<dyn EventManager>) -> Self {
        Self {
            active: ActiveBlockPool::new(),
            inactive: InactiveBlockPool::new(),
            events,
        }
    }

    pub fn allocate_blocks(
        &mut self,
        count: usize,
        state: Arc<Mutex<State<S, M>>>,
    ) -> Result<Vec<MutableBlock<S, M>>, BlockPoolError> {
        let available_blocks = self.inactive.available_blocks() as usize;

        if available_blocks < count {
            tracing::debug!(
                "not enough blocks available, requested: {}, available: {}",
                count,
                available_blocks
            );
            return Err(BlockPoolError::NotEnoughBlocksAvailable(
                count,
                available_blocks,
            ));
        }

        let mut blocks = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(block) = self.inactive.acquire_free_block() {
                blocks.push(MutableBlock {
                    block: Some(block),
                    state: state.clone(),
                });
            }
        }

        Ok(blocks)
    }

    pub fn register_block(
        &mut self,
        block: MutableBlock<S, M>,
    ) -> Result<ImmutableBlock<S, M>, BlockPoolError> {
        let (mut block, state) = block.into_parts();

        let mut block = block.take().ok_or(BlockPoolError::InvalidMutableBlock(
            "inner block was dropped".to_string(),
        ))?;

        // need to validate that the block is in a complete with a valid sequence hash
        // next, we need to ensure there were no mid-air collisions with the sequence hash, meaning:
        // - the sequence hash is not already in the active map
        // - the sequence hash is not already in the inactive pool

        let sequence_hash = block.sequence_hash().map_err(|e| {
            BlockPoolError::InvalidMutableBlock(format!(
                "block has no sequence hash: {}",
                e
            ))
        })?;

        if let Some(immutable) = self.active.match_sequence_hash(sequence_hash) {
            self.return_block(block);
            return Ok(immutable);
        }

        if let Some(mutable) = self.inactive.match_sequence_hash(sequence_hash) {
            self.return_block(block);
            let block = MutableBlock {
                block: Some(mutable),
                state,
            };
            return self.active.register(block);
        }

        // the block is not in the active or inactive pool; now we can register it with the event manager
        // and add it to the active pool

        block
            .register(self.events.as_ref())
            .map_err(|e| BlockPoolError::FailedToRegisterBlock(e.to_string()))?;

        assert!(block.is_registered(), "block is not registered");

        let mutable = MutableBlock {
            block: Some(block),
            state,
        };

        self.active.register(mutable)
    }

    /// Returns a block to the inactive pool
    pub fn return_block(&mut self, mut block: Block<S, M>) {
        self.active.remove(&mut block);
        self.inactive.return_block(block);
    }
}
