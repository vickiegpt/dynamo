use super::block::{Block, BlockMetadata};
use super::storage::Storage;

pub struct OffloadSender<SendType: Storage, Metadata: BlockMetadata> {
    block_tx: tokio::sync::mpsc::UnboundedSender<(Block<SendType, Metadata>, oneshot::Sender<()>)>,
}

pub struct OffloadReceiver<ReceiveType: Storage, Metadata: BlockMetadata> {
    block_rx:
        tokio::sync::mpsc::UnboundedReceiver<(Block<ReceiveType, Metadata>, oneshot::Sender<()>)>,
}

pub fn build_offload_manager<BlockType: Storage, Metadata: BlockMetadata>() -> (
    OffloadSender<BlockType, Metadata>,
    OffloadReceiver<BlockType, Metadata>,
) {
    let (block_tx, block_rx) = tokio::sync::mpsc::unbounded_channel();

    (OffloadSender { block_tx }, OffloadReceiver { block_rx })
}
