use super::*;
use crate::block_manager::block::{BlockState, registry::{RegistrationHandle, BlockHandle}};

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

use std::marker::PhantomData;


#[derive(Builder, Dissolve)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct ExternalBlockPoolArgs<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    #[builder(default = "CancellationToken::new()")]
    cancel_token: CancellationToken,

    #[builder(default = "Handle::current()")]
    async_runtime: Handle,

    marker: PhantomData<(S, L, M)>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ExternalBlockPoolArgsBuilder<S, L, M> {
    pub fn build(self) -> anyhow::Result<ExternalBlockPool<S, L, M>> {
        let args = self.build_internal()?;

        let (cancel_token, async_runtime, _) = args.dissolve();

        ExternalBlockPool::new(cancel_token, async_runtime)
    }
}

type ImmutableBlockRequest = Vec<(usize, TokenBlock)>;

enum ExternalBlockPoolRequest<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    AddBlocks(RequestResponse<Vec<Block<S, L, M>>, BlockPoolResult<()>>),
    GetMutableBlocksById(RequestResponse<Vec<usize>, BlockPoolResult<MutableBlocks<S, L, M>>>),
    GetImmutableBlocksById(RequestResponse<ImmutableBlockRequest, BlockPoolResult<ImmutableBlocks<S, L, M>>>),
    Reset(RequestResponse<(), BlockPoolResult<()>>),
}

struct ExternalBlockManager<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    unclaimed_blocks: HashMap<usize, Block<S, L, M>>,
    immutable_blocks: HashMap<usize, Weak<MutableBlock<S, L, M>>>,

    available_blocks: Arc<AtomicU64>,
    total_blocks: Arc<AtomicU64>,

    return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, L, M>>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ExternalBlockManager<S, L, M> {
    pub fn new(return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, L, M>>, available_blocks: Arc<AtomicU64>, total_blocks: Arc<AtomicU64>) -> Self {
        Self {
            unclaimed_blocks: HashMap::new(),
            immutable_blocks: HashMap::new(),
            available_blocks,
            total_blocks,
            return_tx,
        }
    }

    pub fn return_block(&mut self, mut block: Block<S, L, M>) -> BlockPoolResult<()> {
        block.reset();

        let id = block.block_id();
        
        if let std::collections::hash_map::Entry::Occupied(entry) = self.immutable_blocks.entry(id) {
            assert!(entry.get().upgrade().is_none());
            entry.remove();
        }

        if self.unclaimed_blocks.contains_key(&id) {
            return Err(BlockPoolError::NotReturnable);
        }

        self.unclaimed_blocks.insert(id, block);

        self.available_blocks.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    pub fn add_blocks(&mut self, blocks: Vec<Block<S, L, M>>) -> BlockPoolResult<()> {
        for block in blocks {
            self.return_block(block)?;
            self.total_blocks.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    fn consume_return_channel(&mut self, return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>) -> BlockPoolResult<()> {
        while let Ok(block) = return_rx.try_recv() {
            self.return_block(block)?;
        }

        Ok(())
    }

    pub fn get_mutable_blocks_by_id(&mut self, ids: Vec<usize>, return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>) -> BlockPoolResult<Vec<MutableBlock<S, L, M>>> {
        
        let mut blocks = Vec::new();

        for id in ids {
            if let Some(block) = self.unclaimed_blocks.remove(&id) {
                blocks.push(MutableBlock::new(block, self.return_tx.clone()));
            } else {
                self.consume_return_channel(return_rx)?;

                if let Some(block) = self.unclaimed_blocks.remove(&id) {
                    blocks.push(MutableBlock::new(block, self.return_tx.clone()));
                } else {
                    // This id doesn't exist, or is being used by another block.
                    return Err(BlockPoolError::InvalidMutableBlock("Block already in use".to_string()));
                }
            }
        }

        Ok(blocks)
    }

    pub fn get_immutable_blocks_by_id(&mut self, ids: Vec<(usize, TokenBlock)>, return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>) -> BlockPoolResult<Vec<ImmutableBlock<S, L, M>>> {
        let mut blocks = Vec::new();

        for (id, token_block) in ids {
            let handle = RegistrationHandle::from_token_block(&token_block, NullEventManager::new());
            let (tx, _) = tokio::sync::mpsc::unbounded_channel();
            let block_handle = BlockHandle::new(token_block.sequence_hash(), tx);

            let token_block_clone = token_block.clone();
            let build_mutable = |mut mutable: MutableBlock<S, L, M>| -> BlockPoolResult<Arc<MutableBlock<S, L, M>>> {
                mutable.apply_token_block(token_block_clone).map_err(|e| BlockPoolError::InvalidMutableBlock(e.to_string()))?;
                mutable.update_state(BlockState::Registered(Arc::new(handle), Arc::new(block_handle)));
                Ok(Arc::new(mutable))
            };

            let mutable_block = if let Some(block) = self.unclaimed_blocks.remove(&id) {
                let mutable = MutableBlock::new(block, self.return_tx.clone());
                build_mutable(mutable)?
            } else if let Some(block) = self.immutable_blocks.get(&id) {
                if let Some(mutable) = block.upgrade() {
                    if mutable.sequence_hash().unwrap() != token_block.sequence_hash() {
                        return Err(BlockPoolError::InvalidMutableBlock("Block sequence hash mismatch".to_string()));
                    }

                    mutable
                } else {
                    self.consume_return_channel(return_rx)?;

                    if let Some(block) = self.unclaimed_blocks.remove(&id) {
                        let mutable = MutableBlock::new(block, self.return_tx.clone());
                        build_mutable(mutable)?
                    } else {
                        // Something went terribly wrong. This id doesn't exist, or is being used by another block.
                        return Err(BlockPoolError::InvalidMutableBlock("Block already in use".to_string()));
                    }
                }
            } else {
                return Err(BlockPoolError::InvalidMutableBlock("An immutable block with this id already exists".to_string()));
            };

            blocks.push(ImmutableBlock::new(mutable_block));
        }

        Ok(blocks)
    }

    pub fn reset(&mut self, return_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>) -> BlockPoolResult<()> {
        self.consume_return_channel(return_rx)?;

        if self.immutable_blocks.is_empty() && self.available_blocks.load(Ordering::Relaxed) == self.total_blocks.load(Ordering::Relaxed) {
            Ok(())
        } else {
            Err(BlockPoolError::ResetError("Reset failed".to_string()))
        }
    }
}

pub struct ExternalBlockPool<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    req_tx: tokio::sync::mpsc::UnboundedSender<ExternalBlockPoolRequest<S, L, M>>,

    available_blocks: Arc<AtomicU64>,

    total_blocks: Arc<AtomicU64>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ExternalBlockPool<S, L, M> {
    pub fn builder() -> ExternalBlockPoolArgsBuilder<S, L, M> {
        ExternalBlockPoolArgsBuilder::default()
    }

    pub fn new(cancel_token: CancellationToken, async_runtime: Handle) -> anyhow::Result<Self> {
        let (req_tx, mut req_rx) = tokio::sync::mpsc::unbounded_channel();

        let available_blocks = Arc::new(AtomicU64::new(0));
        let total_blocks = Arc::new(AtomicU64::new(0));

        let available_blocks_clone = available_blocks.clone();
        let total_blocks_clone = total_blocks.clone();

        CriticalTaskExecutionHandle::new_with_runtime(
            |cancel_token| async move {
                let (return_tx, mut return_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut external_block_manager = ExternalBlockManager::new(return_tx, available_blocks_clone, total_blocks_clone);

                loop {
                    tokio::select! {

                        biased;

                        _ = cancel_token.cancelled() => {
                            break;
                        }

                        Some(block) = return_rx.recv() => {
                            external_block_manager.return_block(block)?;
                        }

                        Some(request) = req_rx.recv() => {
                            match request {
                                ExternalBlockPoolRequest::AddBlocks(request) => {
                                    let (blocks, response_tx) = request.dissolve();

                                    response_tx.send(external_block_manager.add_blocks(blocks)).map_err(|_| BlockPoolError::ProgressEngineShutdown)?;
                                },

                                ExternalBlockPoolRequest::GetMutableBlocksById(request) => {
                                    let (ids, response_tx) = request.dissolve();

                                    response_tx.send(external_block_manager.get_mutable_blocks_by_id(ids, &mut return_rx)).map_err(|_| BlockPoolError::ProgressEngineShutdown)?;
                                },
                                
                                ExternalBlockPoolRequest::GetImmutableBlocksById(request) => {
                                    let (request, response_tx) = request.dissolve();

                                    response_tx.send(external_block_manager.get_immutable_blocks_by_id(request, &mut return_rx)).map_err(|_| BlockPoolError::ProgressEngineShutdown)?;
                                }

                                ExternalBlockPoolRequest::Reset(request) => {
                                    let (_, response_tx) = request.dissolve();
                                    
                                    response_tx.send(external_block_manager.reset(&mut return_rx)).map_err(|_| BlockPoolError::ProgressEngineShutdown)?;
                                }
                            }
                        }
                    };
                }
                Ok(())
            },
            cancel_token,
            "External Block Pool Worker",
            &async_runtime
        )?.detach();

        Ok(Self { req_tx, available_blocks, total_blocks })
    }

    pub fn get_mutables_by_id(&self, ids: Vec<usize>) -> BlockPoolResult<MutableBlocks<S, L, M>> {
        let (req, resp_rx) = RequestResponse::new(ids);

        self.req_tx.send(ExternalBlockPoolRequest::GetMutableBlocksById(req)).map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx.blocking_recv().map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    pub fn get_immutables_by_id(&self, ids: Vec<(usize, TokenBlock)>) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        let (req, resp_rx) = RequestResponse::new(ids);

        self.req_tx.send(ExternalBlockPoolRequest::GetImmutableBlocksById(req)).map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        resp_rx.blocking_recv().map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn _add_blocks(&self, blocks: Vec<Block<S, L, M>>) -> AsyncResponse<BlockPoolResult<()>> {
        let (req, resp_rx) = RequestResponse::new(blocks);

        self.req_tx.send(ExternalBlockPoolRequest::AddBlocks(req)).map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }

    fn _reset(&self) -> AsyncResponse<BlockPoolResult<()>> {
        let (req, resp_rx) = RequestResponse::new(());

        self.req_tx.send(ExternalBlockPoolRequest::Reset(req)).map_err(|_| BlockPoolError::ProgressEngineShutdown)?;

        Ok(resp_rx)
    }
}

#[async_trait]
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> BlockPool<S, L, M> for ExternalBlockPool<S, L, M> {
    async fn add_blocks(&self, blocks: Vec<Block<S, L, M>>) -> Result<(), BlockPoolError> {
        self._add_blocks(blocks)?.await.map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn add_blocks_blocking(&self, blocks: Vec<Block<S, L, M>>) -> Result<(), BlockPoolError> {
        self._add_blocks(blocks)?.blocking_recv().map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    async fn allocate_blocks(&self, _count: usize) -> Result<Vec<MutableBlock<S, L, M>>, BlockPoolError> {
        unimplemented!()
    }

    fn allocate_blocks_blocking(&self, _count: usize) -> Result<Vec<MutableBlock<S, L, M>>, BlockPoolError> {
        unimplemented!()
    }

    async fn register_blocks(&self, _blocks: Vec<MutableBlock<S, L, M>>) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        unimplemented!()
    }
    
    fn register_blocks_blocking(&self, _blocks: Vec<MutableBlock<S, L, M>>) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        unimplemented!()
    }

    async fn match_sequence_hashes(&self, _sequence_hashes: &[SequenceHash]) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        unimplemented!()
    }
    
    fn match_sequence_hashes_blocking(&self, _sequence_hashes: &[SequenceHash]) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        unimplemented!()
    }

    async fn touch_blocks(&self, _sequence_hashes: &[SequenceHash]) -> Result<(), BlockPoolError> {
        unimplemented!()
    }

    fn touch_blocks_blocking(&self, _sequence_hashes: &[SequenceHash]) -> Result<(), BlockPoolError> {
        unimplemented!()
    }

    async fn reset(&self) -> BlockPoolResult<()> {
        self._reset()?.await.map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    fn reset_blocking(&self) -> BlockPoolResult<()> {
        self._reset()?.blocking_recv().map_err(|_| BlockPoolError::ProgressEngineShutdown)?
    }

    async fn try_return_block(&self, _block: OwnedBlock<S, L, M>) -> BlockPoolResult<()> {
        unimplemented!()
    }

    fn try_return_block_blocking(&self, _block: OwnedBlock<S, L, M>) -> BlockPoolResult<()> {
        unimplemented!()
    }

    fn total_blocks(&self) -> u64 {
        self.total_blocks.load(Ordering::Relaxed)
    }

    fn available_blocks(&self) -> u64 {
        self.available_blocks.load(Ordering::Relaxed)
    }
}