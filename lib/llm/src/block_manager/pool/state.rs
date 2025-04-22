use super::*;

impl<S: BlockLayout, M: BlockMetadata> State<S, M> {
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

    pub fn register_blocks(
        &mut self,
        blocks: Vec<MutableBlock<S, M>>,
        state: Arc<Mutex<State<S, M>>>,
    ) -> Result<Vec<ImmutableBlock<S, M>>, RegisterResult<S, M>> {
        let mut blocks = RegisterResult::new(blocks).map_err(|e| e)?;

        while let Some((idx, sequence_hash, block)) = blocks.mutable.pop_front() {
            if let Some(immutable) = self.active.match_sequence_hash(sequence_hash) {
                blocks.immutable.push(immutable);
                blocks.ok_to_drop.push(block);
            } else if let Some(raw_block) = self.inactive.match_sequence_hash(sequence_hash) {
                let registration = self
                    .active
                    .register_v2(MutableBlock::new(raw_block, state.clone()));

                match registration.into_result() {
                    Ok(immutable) => {
                        blocks.immutable.push(immutable);
                        blocks.ok_to_drop.push(block);
                    }
                    Err(raw) => {
                        tracing::error!("failed to register block {idx}; inactive match found, but failed to promote");
                        blocks.mutable.push_front((idx, sequence_hash, block));
                        blocks.ok_to_drop.push(raw);
                        return Err(blocks);
                    }
                }
            } else {
                // we need to re-insert the block into the deque to ensure it is processed
                blocks.mutable.push_front((idx, sequence_hash, block));
                break;
            }
        }

        blocks.into_result()
    }

    /// Returns a block to the inactive pool
    pub fn return_block(&mut self, mut block: Block<S, M>) {
        self.active.remove(&mut block);
        self.inactive.return_block(block);
    }
}

#[derive(Debug, Default)]
pub(crate) struct RegisterResult<S: BlockLayout, M: BlockMetadata> {
    immutable: Vec<ImmutableBlock<S, M>>,
    mutable: VecDeque<(usize, SequenceHash, MutableBlock<S, M>)>,
    ok_to_drop: Vec<MutableBlock<S, M>>,
}

impl<S: BlockLayout, M: BlockMetadata> RegisterResult<S, M> {
    fn new(blocks: Vec<MutableBlock<S, M>>) -> Result<Self, Self> {
        let immutable = Vec::new();
        let ok_to_drop = Vec::new();
        let mut mutable = VecDeque::new();
        let mut errors = Vec::new();

        if blocks.is_empty() {
            return Ok(Self {
                immutable,
                mutable,
                ok_to_drop,
            });
        }

        for (idx, block) in blocks.into_iter().enumerate() {
            if !block.is_complete() {
                errors.push(BlockPoolError::InvalidMutableBlock(format!(
                    "block {} is not complete",
                    idx
                )));
            }

            let sequence_hash = block
                .sequence_hash()
                .map_err(|e| {
                    errors.push(BlockPoolError::InvalidMutableBlock(format!(
                        "block {} has no sequence hash: {}",
                        idx, e
                    )));
                })
                .ok()
                .unwrap_or(0);

            mutable.push_back((idx, sequence_hash, block));
        }

        if errors.is_empty() {
            Ok(Self {
                immutable,
                mutable,
                ok_to_drop,
            })
        } else {
            Err(Self {
                immutable,
                mutable,
                ok_to_drop,
            })
        }
    }

    pub fn into_result(self) -> Result<Vec<ImmutableBlock<S, M>>, Self> {
        if self.mutable.is_empty() {
            Ok(self.immutable)
        } else {
            Err(self)
        }
    }

    pub fn immutable_count(&self) -> usize {
        self.immutable.len()
    }

    pub fn mutable_count(&self) -> usize {
        self.mutable.len()
    }
}
