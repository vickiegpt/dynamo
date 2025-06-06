use dynamo_llm::block_manager::storage::{DeviceAllocator, DiskAllocator};
use dynamo_llm::block_manager::{
    block::transfer::{write_blocks_to, NixlTransfer, TransferContext},
    block::MutableBlock,
    layout::{FullyContiguous, LayoutConfig},
    BasicMetadata, Blocks, DType, NixlLayout,
};

use nixl_sys::Agent;

use cudarc::driver::CudaContext;
use std::sync::Arc;
use tokio::sync::mpsc;

fn make_agent() -> anyhow::Result<Agent> {
    let agent = Agent::new("test-agent")?;

    let (_, gds_params) = agent.get_plugin_params("GDS").unwrap();
    agent.create_backend("GDS", &gds_params)?;

    let (_, ucx_params) = agent.get_plugin_params("UCX").unwrap();
    agent.create_backend("UCX", &ucx_params)?;

    Ok(agent)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = make_agent()?;

    let config = LayoutConfig {
        num_blocks: 1,
        num_layers: 64,
        outer_dim: 1,
        page_size: 1024,
        inner_dim: 1024,
        alignment: 1,
        dtype: DType::FP16,
    };

    let mut device = FullyContiguous::allocate(config.clone(), &DeviceAllocator::default())?;
    let mut disk = FullyContiguous::allocate(config.clone(), &DiskAllocator::default())?;

    device.nixl_register(&agent, None)?;
    disk.nixl_register(&agent, None)?;

    let (tx_device, _) = mpsc::unbounded_channel();
    let (tx_disk, _) = mpsc::unbounded_channel();

    let device_block = MutableBlock::new(
        Blocks::<_, BasicMetadata>::new(device, 42, 0)?
            .into_blocks()?
            .into_iter()
            .next()
            .unwrap(),
        tx_device,
    );
    let disk_block = MutableBlock::new(
        Blocks::<_, BasicMetadata>::new(disk, 42, 0)?
            .into_blocks()?
            .into_iter()
            .next()
            .unwrap(),
        tx_disk,
    );

    let cuda_ctx = CudaContext::new(0)?;
    let stream = cuda_ctx.new_stream()?;

    let handle = tokio::runtime::Handle::current();
    let ctx = Arc::new(TransferContext::new(Arc::new(Some(agent)), stream, handle));

    let sources = vec![Arc::new(device_block)];
    let mut destinations = vec![disk_block];

    for _ in 0..10 {
        let start = std::time::Instant::now();
        let fut = write_blocks_to(
            sources.as_slice(),
            destinations.as_mut_slice(),
            &ctx,
            NixlTransfer::Write,
        )?;

        fut.await;
        println!("Transfer complete! Time: {:?}", start.elapsed());
    }

    Ok(())
}
