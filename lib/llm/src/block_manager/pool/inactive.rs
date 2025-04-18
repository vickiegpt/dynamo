use super::*;

impl<T: Storage, M: BlockMetadata> InactiveBlockPool<T, M> {
    pub(crate) fn new() -> Self {
        Self {
            lookup_map: HashMap::new(),
            priority_set: BTreeSet::new(),
            uninitialized_set: VecDeque::new(),
            return_tick: 0,
            total_blocks: 0,
        }
    }

    pub fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    pub fn available_blocks(&self) -> u64 {
        self.uninitialized_set.len() as u64 + self.lookup_map.len() as u64
    }

    fn insert_with_sequence_hash(&mut self, block: BlockType<T, M>, sequence_hash: SequenceHash) {
        let priority_key = PriorityKey::new(block.metadata().clone(), sequence_hash);
        if self.priority_set.contains(&priority_key) {
            tracing::debug!("multiple entries with the same priority key, resetting block and inserting into uninitialized set");
            let mut block = block;
            block.reset();
            self.uninitialized_set.push_back(block);
        } else if let std::collections::hash_map::Entry::Vacant(e) =
            self.lookup_map.entry(sequence_hash)
        {
            tracing::debug!("inserting block to map and priority set");
            self.priority_set.insert(priority_key);
            e.insert(block);
        } else {
            tracing::debug!("multiple entries in lookup map with the same sequence hash, inserting into uninitialized set");
            let mut block = block;
            block.reset();
            self.uninitialized_set.push_back(block);
        }
    }

    // Insert an item with a given key and sequence_hash
    fn insert(&mut self, block: BlockType<T, M>) {
        tracing::debug!("inserting block into available blocks");

        // If we already have an entry for this sequence hash or the block is reset,
        // we need to move it to the uninitialized set
        match block.state() {
            BlockState::Reset => {
                tracing::debug!("inserted block to uninitialized set");
                self.uninitialized_set.push_back(block);
            }
            BlockState::Partial(_) => {
                tracing::debug!("inserted block to uninitialized set");
                let mut block = block;
                block.reset();
                self.uninitialized_set.push_back(block);
            }
            BlockState::Complete(_) => {
                tracing::debug!("inserting completed/unregistered block to uninitialized set");
                let mut block = block;
                block.reset();
                self.uninitialized_set.push_back(block);
            }
            BlockState::Registered(state) => {
                tracing::debug!("inserting registered block to map and priority set");
                let sequence_hash = state.sequence_hash();
                self.insert_with_sequence_hash(block, sequence_hash);
            }
        }
    }

    pub fn add_blocks(&mut self, blocks: Vec<BlockType<T, M>>) {
        let count = blocks.len() as u64;

        for mut block in blocks {
            block.reset();
            self.insert(block);
        }

        self.total_blocks += count;
    }

    pub fn return_block(&mut self, mut block: BlockType<T, M>) {
        // increment the return tick
        self.return_tick += 1;

        // update the metadata
        block.metadata_mut().on_returned(self.return_tick);

        // insert the block into the pool
        self.insert(block);
    }

    pub fn return_blocks(&mut self, blocks: Vec<BlockType<T, M>>) {
        // return the block to the pool from tail to head
        for block in blocks.into_iter().rev() {
            self.return_block(block);
        }
    }

    fn take_with_sequence_hash(&mut self, sequence_hash: SequenceHash) -> Option<BlockType<T, M>> {
        match self.lookup_map.remove(&sequence_hash) {
            Some(block) => {
                // Remove from priority set
                let priority_key = PriorityKey::new(block.metadata().clone(), sequence_hash);
                self.priority_set.remove(&priority_key);
                Some(block)
            }
            None => None,
        }
    }

    pub fn match_sequence_hash(&mut self, sequence_hash: SequenceHash) -> Option<BlockType<T, M>> {
        self.take_with_sequence_hash(sequence_hash)
    }

    pub fn match_sequence_hashes(
        &mut self,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Vec<BlockType<T, M>> {
        let mut matched_blocks = Vec::with_capacity(sequence_hashes.len());

        for hash in sequence_hashes {
            if let Some(block) = self.take_with_sequence_hash(hash) {
                matched_blocks.push(block);
            } else {
                break;
            }
        }

        matched_blocks
    }

    pub fn match_token_blocks(&mut self, token_blocks: &[TokenBlock]) -> Vec<BlockType<T, M>> {
        let mut matched_blocks = Vec::with_capacity(token_blocks.len());

        for token_block in token_blocks {
            if let Some(block) = self.take_with_sequence_hash(token_block.sequence_hash()) {
                matched_blocks.push(block);
            } else {
                break;
            }
        }

        matched_blocks
    }

    pub fn acquire_free_block(&mut self) -> Option<BlockType<T, M>> {
        // First try uninitialized blocks - these are often part of sequences
        // that have been arranged in the correct order
        if let Some(mut block) = self.uninitialized_set.pop_front() {
            tracing::trace!("acquired uninitialized block");
            block.metadata_mut().on_acquired();
            return Some(block);
        }

        // if we have blocks in the priority set, pop the first (it's sorted by priority)
        // a fatal error will occur if the block is not found in the lookup map
        if let Some(key) = self.priority_set.pop_first() {
            tracing::trace!("acquired priority/registered block map; resetting block");
            match self.lookup_map.remove(&key.sequence_hash()) {
                Some(mut block) => {
                    block.reset();
                    block.metadata_mut().on_acquired();
                    Some(block)
                }
                None => {
                    // This case should ideally not happen if the sets are consistent.
                    // Log an error or handle appropriately.
                    tracing::error!(
                        sequence_hash = ?key.sequence_hash(),
                        "Block from priority set not found in lookup map! Inconsistency detected."
                    );
                    // Attempt to continue by trying again or returning None
                    // Depending on desired robustness, might panic in debug builds.
                    None
                }
            }
        } else {
            // No blocks available in either set
            None
        }
    }

    pub fn acquire_free_blocks(
        &mut self,
        count: usize,
    ) -> Result<Vec<BlockType<T, M>>, BlockPoolError> {
        let mut blocks = Vec::with_capacity(count);

        let available_now = self.uninitialized_set.len() + self.lookup_map.len();

        if count > available_now {
            return Err(BlockPoolError::InsufficientBlocksAvailable(
                count,
                available_now,
            ));
        }

        for _ in 0..count {
            // Directly call the logic in acquire_free_block
            if let Some(block) = self.acquire_free_block() {
                blocks.push(block);
            } else {
                // This should not happen if the initial check passed and there are no concurrent modifications.
                // If it does, it indicates an inconsistency or a logic error.
                tracing::error!(
                    requested = count,
                    acquired = blocks.len(),
                    available_at_start = available_now,
                    current_available = self.uninitialized_set.len() + self.lookup_map.len(),
                    "Insufficient blocks during acquisition loop despite initial check."
                );
                // Return the blocks acquired so far, or handle as an error.
                // For now, we break and return what we have, but decrementing 'available_blocks'
                // needs to account for the actual number acquired.
                // Consider returning an error or panicking in debug.
                break;
            }
        }

        // Check if we got the requested number of blocks
        if blocks.len() != count {
            // This path is taken if the loop broke early due to unexpected `None` from acquire_free_block
            // Return an error indicating partial success or failure
            // Depending on the desired behavior, you might return the partial list
            // or a more specific error.
            // For consistency with the original check, let's return an error if count wasn't met.
            return Err(BlockPoolError::InsufficientBlocksAvailable(
                count,
                blocks.len(),
            ));
        }

        Ok(blocks)
    }

    // fn handle_take(&mut self, take: Take<T, M>) {
    //     let (count, return_handle, tx) = take.dissolve();

    //     let mut taken_blocks = Vec::with_capacity(count as usize);

    //     for _ in 0..count {
    //         if let Some(block) = self.take() {
    //             taken_blocks.push(self.create_pool_item(block, return_handle.clone()));
    //         } else {
    //             break;
    //         }
    //     }

    //     let count = taken_blocks.len() as u64;
    //     self.available_blocks_tx
    //         .send_modify(|n| *n = n.saturating_sub(count));

    //     // Send the result back through the channel
    //     if tx.send(taken_blocks).is_err() {
    //         tracing::trace!("Failed to send matched blocks to requester");
    //     }
    // }

    // fn handle_match_request(&mut self, match_request: MatchRequest<T, M>) {
    //     match match_request {
    //         MatchRequest::MatchSingle(match_single) => self.handle_match_single(match_single),
    //         MatchRequest::MatchMultiple(match_multiple) => {
    //             self.handle_match_multiple(match_multiple)
    //         }
    //         MatchRequest::Take(take) => self.handle_take(take),
    //     }
    // }

    // fn handle_control_request(&mut self, control_request: ControlRequest<T, M>) {
    //     match control_request {
    //         ControlRequest::Insert(insert) => {
    //             let (block, tx) = insert.dissolve();
    //             self.handle_insert(block);
    //             if tx.send(()).is_err() {
    //                 tracing::trace!("Failed to send insert ack; receiver dropped");
    //             }
    //         }
    //         ControlRequest::UpdateSingle(update_single) => {
    //             let (update, tx) = update_single.dissolve();
    //             self.handle_update_single(update);
    //             if tx.send(()).is_err() {
    //                 tracing::trace!("Failed to send update single ack; receiver dropped");
    //             }
    //         }
    //         ControlRequest::UpdateMultiple(update_multiple) => {
    //             let (updates, tx) = update_multiple.dissolve();
    //             self.handle_update_multiple(updates);
    //             if tx.send(()).is_err() {
    //                 tracing::trace!("Failed to send update multiple ack; receiver dropped");
    //             }
    //         }
    //         ControlRequest::Reset(reset) => {
    //             let (sequence_hashes, tx, _) = reset.dissolve();
    //             self.handle_reset(sequence_hashes);
    //             if tx.send(()).is_err() {
    //                 tracing::trace!("Failed to send reset ack; receiver dropped");
    //             }
    //         }
    //         ControlRequest::ResetAll(reset_all) => {
    //             let (tx, _) = reset_all.dissolve();
    //             self.handle_reset_all();
    //             if tx.send(()).is_err() {
    //                 tracing::trace!("Failed to send reset all ack; receiver dropped");
    //             }
    //         }
    //     }
    // }

    // fn handle_insert(&mut self, block: Block<T, M>) {
    //     self.available_blocks_tx.send_modify(|n| *n += 1);
    //     self.total_blocks_tx.send_modify(|n| *n += 1);
    //     self.return_tick += 1;

    //     self.insert(PoolValue::Direct(block));
    // }

    // fn handle_return(&mut self, block: BlockType<T, M>) {
    //     self.available_blocks_tx.send_modify(|n| *n += 1);
    //     self.return_tick += 1;

    //     self.insert(block);
    // }

    // fn handle_update_single(&mut self, update: UpdateBlock<M>) {
    //     self.update_block(vec![update]);
    // }

    // fn handle_update_multiple(&mut self, updates: Vec<UpdateBlock<M>>) {
    //     for update in updates {
    //         if let Some(mut block) = self.take_with_sequence_hash(update.hash) {
    //             *block.metadata_mut() = update.metadata;
    //             self.insert(block);
    //         }
    //     }
    // }

    // fn update_block(&mut self, updates: Vec<UpdateBlock<M>>) {
    //     for update in updates {
    //         if let Some(mut block) = self.take_with_sequence_hash(update.hash) {
    //             *block.metadata_mut() = update.metadata;
    //             self.insert(block);
    //         }
    //     }
    // }

    // fn handle_reset(&mut self, sequence_hashes: Vec<SequenceHash>) {
    //     for hash in sequence_hashes {
    //         if let Some(mut block) = self.take_with_sequence_hash(hash) {
    //             // Reset metadata through deref
    //             block.metadata_mut().reset_metadata();
    //             self.insert(block);
    //         }
    //     }
    // }

    // fn handle_reset_all(&mut self) {
    //     while let Some(priority_key) = self.priority_set.pop_first() {
    //         if let Some(mut block) = self.lookup_map.remove(&priority_key.sequence_hash()) {
    //             // reset block -- both state and metadata
    //             block.reset();
    //             self.insert(block);
    //         } else {
    //             panic!("block from priority set not found in lookup map");
    //         }
    //     }
    // }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{
        block_manager::{
            block::{state::CompleteState, BlockStorageCollection},
            events::NullEventManager,
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

    pub fn create_block_pool(num_blocks: usize) -> InactiveBlockPool<NullStorage, TestMetadata> {
        let mut pool = InactiveBlockPool::new();

        let block_collection =
            BlockStorageCollection::<NullStorage, TestMetadata>::new(NullLayout::new(num_blocks))
                .unwrap();

        let blocks = block_collection.into_blocks().unwrap();
        pool.add_blocks(blocks);

        pool
    }

    pub fn acquire_blocks(
        tokens: Tokens,
        block_size: usize,
        pool: &mut InactiveBlockPool<NullStorage, TestMetadata>,
    ) -> (Vec<Block<NullStorage, TestMetadata>>, usize) {
        let (mut token_blocks, _partial_token_block) =
            tokens.into_sequence(block_size, None).into_parts();

        let total_complete_blocks = token_blocks.len();

        // this will match the token_blocks to any matching blocks in the inactive pool
        // these blocks have the same sequence hash as the token_blocks, thus no updates are needed
        let mut matched_blocks = pool.match_token_blocks(&token_blocks);
        let matched_block_count = matched_blocks.len();
        println!("matched_blocks: {:?}", matched_block_count);

        let event_manager = NullEventManager {};

        // all matched blocks should be in the complete or registered state
        for block in &mut matched_blocks {
            assert!(block.is_registered());
        }

        // drain the matched blocks from the token_blocks
        token_blocks.drain(0..matched_block_count);

        assert_eq!(
            token_blocks.len() + matched_blocks.len(),
            total_complete_blocks
        );

        // try to acquire the remaining blocks
        let mut unmatched_blocks = pool.acquire_free_blocks(token_blocks.len()).unwrap();

        assert_eq!(unmatched_blocks.len(), token_blocks.len());

        for unmatched in &unmatched_blocks {
            assert!(unmatched.is_empty());
        }

        for (unmatched, token_block) in unmatched_blocks.iter_mut().zip(token_blocks.into_iter()) {
            assert!(unmatched.is_empty());
            *unmatched.state_mut() = BlockState::Complete(CompleteState::new(token_block));
            unmatched.register(&event_manager).unwrap();
            assert!(unmatched.is_registered());
        }

        let mut blocks = matched_blocks;
        blocks.extend(unmatched_blocks);
        (blocks, matched_block_count)
    }

    #[test]
    fn test_block_pool_lifecycle() {
        dynamo_runtime::logging::init();

        const PAGE_SIZE: usize = 2;

        let mut pool = create_block_pool(10);
        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let blocks = pool.acquire_free_blocks(10).unwrap();
        assert_eq!(blocks.len(), 10);
        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 0);

        pool.return_blocks(blocks);

        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let tokens = create_token_sequence(&[1, 2, 3, 4]);

        let (blocks, matched_block_count) = acquire_blocks(tokens.clone(), PAGE_SIZE, &mut pool);
        assert_eq!(blocks.len(), 2);
        assert_eq!(matched_block_count, 0);
        assert_eq!(pool.available_blocks(), 8);

        pool.return_blocks(blocks);

        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let (blocks, matched_block_count) = acquire_blocks(tokens.clone(), PAGE_SIZE, &mut pool);
        assert_eq!(blocks.len(), 2);
        assert_eq!(matched_block_count, 2);
        assert_eq!(pool.available_blocks(), 8);

        pool.return_blocks(blocks);

        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let blocks = pool.acquire_free_blocks(10).unwrap();
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

    //

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

    //

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

    //

    //     let blocks = pool.acquire_free_blocks(4).await.unwrap();
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

    //

    //     let blocks = pool.acquire_free_blocks(4).await.unwrap();
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

    //

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

    //

    //     let blocks = pool.acquire_free_blocks(4).await.unwrap();
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
    //

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
    //

    //     // Try to match from block 2 hashes, expect no matches
    //     let matched = pool.match_sequence_hashes(block2_hashes).await.unwrap();
    //     assert_eq!(matched.len(), 0);

    //     let matched = pool.match_sequence_hashes(block1_hashes).await.unwrap();
    //     assert_eq!(matched.len(), 2);
    // }
}
