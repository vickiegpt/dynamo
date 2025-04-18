use super::*;

use std::sync::Weak;
use tokio::sync::oneshot;

type ActiveRequestSender<S, M> = mpsc::UnboundedSender<RequestType<S, M>>;
type ActiveRequestReceiver<S, M> = mpsc::UnboundedReceiver<RequestType<S, M>>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum ActiveBlockError {
    #[error("Block is not in a completed state")]
    BlockNotCompleted,

    #[error("ActiveBlockPool has been shutdown")]
    PoolShutdown,
}

pub struct ActiveBlockPool<S: Storage, M: BlockMetadata> {
    req_tx: ActiveRequestSender<S, M>,
}

impl<S: Storage, M: BlockMetadata> ActiveBlockPool<S, M> {
    pub async fn new() -> Self {
        let (req_tx, req_rx) = mpsc::unbounded_channel();

        // let _ = tokio::spawn(progress_engine(req_rx));

        Self { req_tx }
    }

    pub async fn register_block(
        &self,
        block: UniqueBlock<S, M>,
    ) -> Result<ActiveBlock<S, M>, ActiveBlockError> {
        match block.state() {
            BlockState::Complete(_) => {}
            _ => {
                tracing::error!("Block is not in a completed state");
                return Err(ActiveBlockError::BlockNotCompleted);
            }
        }

        let (tx, rx) = oneshot::channel();
        let req = RequestType::Register(RequestRegister {
            block,
            ret_tx: self.req_tx.clone(),
            resp_tx: tx,
        });

        if let Err(e) = self.req_tx.send(req) {
            tracing::error!("Failed to return block to pool: {}", e);
            return Err(ActiveBlockError::PoolShutdown);
        }

        rx.await.map_err(|_| ActiveBlockError::PoolShutdown)
    }
}

#[derive(Clone, Debug)]
pub struct ActiveBlock<S: Storage, M: BlockMetadata> {
    inner: Arc<ActiveBlockInner<S, M>>,
}

impl<S: Storage, M: BlockMetadata> Deref for ActiveBlock<S, M> {
    type Target = Block<S, M>;
    fn deref(&self) -> &Self::Target {
        &self.inner.block
    }
}

impl<S: Storage, M: BlockMetadata> ActiveBlock<S, M> {
    // pub fn new(block: SharedBlock<S, M>, ret_tx: ActiveRequestSender<S, M>) -> Self {
    //     Self {
    //         inner: Arc::new(ActiveBlockInner { block, ret_tx }),
    //     }
    // }
}

#[derive(Clone)]
struct ActiveBlockInner<S: Storage, M: BlockMetadata> {
    block: SharedBlock<S, M>,
    ret_tx: ActiveRequestSender<S, M>,
}

impl<S: Storage, M: BlockMetadata> std::fmt::Debug for ActiveBlockInner<S, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ActiveBlock: {:?}", self.block)
    }
}

impl<S: Storage, M: BlockMetadata> Drop for ActiveBlockInner<S, M> {
    // At this moment, the Arc<ActiveBlockInner> has been dropped, and we are dropping the inner ActiveBlockInner.
    // This means the Weak<ActiveBlockInner> no longer has a strong reference to the ActiveBlockInner.
    fn drop(&mut self) {
        let sequence_hash = self.block.sequence_hash().expect("Block is not complete");
        if let Err(e) = self.ret_tx.send(RequestType::Drop(sequence_hash)) {
            tracing::error!("Failed to return block to pool: {}", e);
        }
    }
}

/// Operation allowed on the RequestChannel
enum RequestType<S: Storage, M: BlockMetadata> {
    Match(RequestMatch<S, M>),
    Register(RequestRegister<S, M>),
    Drop(SequenceHash),
}

/// Match request
///
/// Will return a list of active blocks that match the given hashes.
/// Blocks are matched by sequence hash in order such that the first block in the request
/// which is not found in the pool breaks the search.
#[derive(Dissolve)]
struct RequestMatch<S: Storage, M: BlockMetadata> {
    hashes: Vec<SequenceHash>,
    resp_tx: oneshot::Sender<Vec<ActiveBlock<S, M>>>,
}

/// Register request
///
/// The UniqueBlock must be in a Completed state.
/// This should be validated in the driver, not on the progress engine, thus we don't
/// have to return a Result/Option.
#[derive(Dissolve)]
struct RequestRegister<S: Storage, M: BlockMetadata> {
    block: UniqueBlock<S, M>,
    ret_tx: ActiveRequestSender<S, M>,
    resp_tx: oneshot::Sender<ActiveBlock<S, M>>,
}

// async fn progress_engine<S: Storage, M: BlockMetadata>(
//     req_rx: ActiveRequestReceiver<S, M>,
//     event_manager: Arc<dyn EventManager>,
// ) {
//     let mut req_rx = req_rx;
//     let mut active_blocks = HashMap::<SequenceHash, Weak<ActiveBlockInner<S, M>>>::new();
//     while let Some(req) = req_rx.recv().await {
//         match req {
//             RequestType::Register(block) => {
//                 let (block, ret_tx, resp_tx) = block.dissolve();
//                 let sequence_hash = block.sequence_hash().expect("Block is not complete");

//                 if let Some(weak) = active_blocks.get(&sequence_hash) {
//                     // If the block is already registered AND it is still active, then return the active block
//                     // which is different from the one we are registering. The one we are registered is dropped
//                     // and returned to the pool.
//                     if let Some(block) = weak.upgrade() {
//                         // Create a new ActiveBlock from the inner block
//                         let block = ActiveBlock { inner: block };
//                         if let Err(e) = resp_tx.send(block) {
//                             tracing::warn!("Failed to complete registration; the requesting task dropped the response channel");
//                         }
//                         continue;
//                     }
//                 }

//                 // Otherwise, the block is not active, so we insert it into the active blocks map
//                 // Note: it might be the case that there is an entry in the map; however, the weak
//                 // reference is not alive, so we will replace the entry.
//                 let registration_handle

//                 let block = block.into_shared();
//                 let inner = Arc::new(ActiveBlockInner { block, ret_tx });
//                 active_blocks.insert(sequence_hash, Arc::downgrade(&inner));
//                 let active_block = ActiveBlock { inner };
//                 if let Err(e) = resp_tx.send(active_block) {
//                     tracing::warn!("Failed to complete registration; the requesting task dropped the response channel");
//                 }
//             }
//             RequestType::Match(hashes) => {
//                 let (hashes, resp_tx) = hashes.dissolve();
//                 let mut inner_blocks = Vec::new();
//                 for hash in hashes {
//                     if let Some(weak) = active_blocks.get(&hash) {
//                         if let Some(block) = weak.upgrade() {
//                             inner_blocks.push(block);
//                             continue;
//                         }
//                     }
//                     break;
//                 }
//                 let active_blocks = inner_blocks
//                     .into_iter()
//                     .map(|block| ActiveBlock { inner: block })
//                     .collect();
//                 if resp_tx.send(active_blocks).is_err() {
//                     tracing::warn!("Failed to complete match; the requesting task dropped the response channel");
//                 }
//             }
//             RequestType::Drop(block) => {
//                 // A block entry is expected, but if the weak reference is alive, then a registration event
//                 // has occurred before the drop event was processed — if this is the case, we simply continue.
//                 if let Some(weak) = active_blocks.get(&block) {
//                     if let Some(_block) = weak.upgrade() {
//                         continue;
//                     }
//                 }
//                 // Otherwise, the block is not active, so we remove it from the active blocks map
//                 active_blocks.remove(&block);
//             }
//         }
//     }
// }
