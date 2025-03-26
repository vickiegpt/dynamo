use super::{
    block::{DeviceBlockStorage, PinnedBlockStorage},
    layer::CopyStream,
    *,
};

use dynamo_runtime::utils::pool::{Pool, PoolExt};

pub trait LayerCopyEngine<
    S: BlockStorage + Send + Sync + 'static,
    D: BlockStorage + Send + Sync + 'static,
>
{
    fn copy_for_layer(&self, layer_id: usize) -> Result<()>;
}

pub struct TransferManager {
    copy_stream_pool: Pool<CopyStream>,
}

impl TransferManager {
    pub fn new(stream_count: usize, num_layers: usize, num_blocks: usize) -> Result<Self> {
        let mut copy_streams = Vec::new();
        for _ in 0..stream_count {
            copy_streams.push(PoolValue::Direct(CopyStream::new(num_layers, num_blocks)?));
        }
        Ok(Self {
            copy_stream_pool: Pool::new(copy_streams),
        })
    }

    pub async fn create_context<
        S: BlockStorage + Send + Sync + 'static,
        D: BlockStorage + Send + Sync + 'static,
    >(
        &self,
        src: Vec<KvBlock<S>>,
        dst: Vec<KvBlock<D>>,
    ) -> Result<TransferContext<S, D>> {
        let copy_stream = self.copy_stream_pool.acquire().await;
        let context = TransferContext { src, dst };
        Ok(context)
    }
}

pub struct TransferContext<
    S: BlockStorage + Send + Sync + 'static,
    D: BlockStorage + Send + Sync + 'static,
> {
    src: Vec<KvBlock<S>>,
    dst: Vec<KvBlock<D>>,
}

impl<S: BlockStorage + Send + Sync + 'static, D: BlockStorage + Send + Sync + 'static>
    TransferContext<S, D>
{
    pub fn new(src: Vec<KvBlock<S>>, dst: Vec<KvBlock<D>>) -> Self {
        Self { src, dst }
    }
}

impl LayerCopyEngine<PinnedBlockStorage, DeviceBlockStorage>
    for TransferContext<PinnedBlockStorage, DeviceBlockStorage>
{
    fn copy_for_layer(&self, layer_id: usize) -> Result<()> {
        // setup the copy stream
        // run the copy stream
        unimplemented!()
    }
}

impl LayerCopyEngine<DeviceBlockStorage, PinnedBlockStorage>
    for TransferContext<DeviceBlockStorage, PinnedBlockStorage>
{
    fn copy_for_layer(&self, layer_id: usize) -> Result<()> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_agent() {}
}
