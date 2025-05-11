use std::sync::Arc;
use std::thread::spawn;
use tokio::sync::mpsc;

use crate::block_manager::block::{BlockMetadata, MutableBlock};
use crate::block_manager::storage::Storage;

use anyhow::Result;
use cudarc::driver::CudaEvent;

pub struct PendingOffload<S: Storage, M: BlockMetadata> {
    _block: Arc<MutableBlock<S, M>>,
    event: CudaEvent,
}

impl<S: Storage, M: BlockMetadata> PendingOffload<S, M> {
    pub fn new(block: Arc<MutableBlock<S, M>>, event: CudaEvent) -> Self {
        Self {
            _block: block,
            event,
        }
    }
}

const MAX_OFFLOAD_STREAM_DEPTH: usize = 4;

pub struct PendingOffloadManager<S: Storage, M: BlockMetadata> {
    pub pending_offload_q: mpsc::Sender<PendingOffload<S, M>>,
}

impl<S: Storage, M: BlockMetadata> PendingOffloadManager<S, M> {
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::channel::<PendingOffload<S, M>>(MAX_OFFLOAD_STREAM_DEPTH);

        spawn(move || {
            while let Some(pending_offload) = rx.blocking_recv() {
                pending_offload.event.synchronize()?;
                drop(pending_offload);
            }
            Ok::<(), anyhow::Error>(())
        });

        Self {
            pending_offload_q: tx,
        }
    }

    pub async fn handle_pending_offload(
        &self,
        pending_offload: PendingOffload<S, M>,
    ) -> Result<()> {
        self.pending_offload_q.send(pending_offload).await?;

        Ok(())
    }
}
