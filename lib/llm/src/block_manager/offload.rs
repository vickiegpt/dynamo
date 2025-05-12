// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, Notify};

use super::storage::Storage;
use crate::block_manager::block::{
    transfer::WriteTo, BlockExt, BlockMetadata, BlockState, ImmutableBlock, MutableBlock,
};
use crate::block_manager::state::TransferContext;
use crate::block_manager::{BlockPool, CacheLevel, DeviceStorage, PinnedStorage};

use anyhow::Result;
use cudarc::driver::{sys::CUevent_flags, CudaContext};
use std::any::Any;

use std::collections::BTreeSet;

mod pending;
mod request;

use pending::{PendingOffload, PendingOffloadManager};
use request::{OffloadRequest, OffloadRequestKey};

/// The offload manager receives and enqueues offload requests from the device.
pub struct OffloadManager<Metadata: BlockMetadata> {
    dtoh_offload_tx: mpsc::UnboundedSender<OffloadRequest<DeviceStorage, Metadata>>,

    tick: Arc<Mutex<u64>>,
}

impl<Metadata: BlockMetadata> OffloadManager<Metadata> {
    pub fn new(
        device: Arc<Option<BlockPool<DeviceStorage, Metadata>>>,
        host: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
    ) -> Arc<Self> {
        let (dtoh_offload_tx, dtoh_offload_rx) = mpsc::unbounded_channel();

        let dtoh_offload_queue = Arc::new(Mutex::new(BTreeSet::new()));
        let dtoh_offload_notify = Arc::new(Notify::new());

        // The task responsible for receiving and enqueuing offload requests.
        let dtoh_offload_queue_clone = dtoh_offload_queue.clone();
        let dtoh_offload_notify_clone = dtoh_offload_notify.clone();
        tokio::spawn(async move {
            OffloadManager::ingress_worker(
                dtoh_offload_queue_clone,
                dtoh_offload_notify_clone,
                dtoh_offload_rx,
            )
            .await
        });

        // The task responsible for processing offload requests.
        tokio::spawn(async move {
            OffloadManager::process_worker(dtoh_offload_queue, dtoh_offload_notify, device, host)
                .await
        });

        Arc::new(Self {
            dtoh_offload_tx,
            tick: Arc::new(Mutex::new(0)),
        })
    }

    fn build_request<S: Storage>(
        block: &ImmutableBlock<S, Metadata>,
        key: OffloadRequestKey,
    ) -> Result<OffloadRequest<S, Metadata>> {
        Ok(OffloadRequest {
            block: Arc::downgrade(block.mutable_block()),
            sequence_hash: block.sequence_hash()?,
            key,
        })
    }

    async fn handle_offload<Source: Storage, Target: Storage>(
        source: &Arc<MutableBlock<Source, Metadata>>,
        mut target: MutableBlock<Target, Metadata>,
        target_pool: &BlockPool<Target, Metadata>,
    ) -> Result<()> {
        let target_mut = &mut target;

        if let BlockState::Registered(reg_handle) = source.state() {
            target_mut.reset();
            target_mut.update_metadata(source.metadata().clone());
            target_mut.apply_token_block(reg_handle.token_block().clone())?;

            target_pool.register_blocks(vec![target]).await?;
        } else {
            panic!("Invalid block state for offload! This should never happen.")
        }

        Ok(())
    }

    async fn ingress_worker(
        dtoh_offload_queue: Arc<Mutex<BTreeSet<OffloadRequest<DeviceStorage, Metadata>>>>,
        dtoh_offload_notify: Arc<Notify>,
        mut dtoh_offload_rx: mpsc::UnboundedReceiver<OffloadRequest<DeviceStorage, Metadata>>,
    ) -> Result<()> {
        while let Some(request) = dtoh_offload_rx.recv().await {
            dtoh_offload_queue.lock().await.insert(request);
            dtoh_offload_notify.notify_one();
        }
        Ok(())
    }

    async fn process_worker(
        dtoh_offload_queue: Arc<Mutex<BTreeSet<OffloadRequest<DeviceStorage, Metadata>>>>,
        dtoh_offload_notify: Arc<Notify>,
        device: Arc<Option<BlockPool<DeviceStorage, Metadata>>>,
        host: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
    ) -> Result<()> {
        // Since cuda memcpys in streams are async, this gets a bit tricky.
        // We can't just consume the queue normally, otherwise the stream would become very backlogged.
        // We also need to ensure that we hold a strong reference to blocks currently being offloaded until the transfer corresponding to the block is complete.
        // We can't just release our strong reference to the block until the transfer is complete, because the block may be deallocated before the transfer is complete.
        // To do this, we use a queue to track blocks currently being offloaded. Once the offload is complete, the reference to the block is dropped.

        if device.is_none() || host.is_none() {
            return Ok(());
        }

        let device = device.as_ref().as_ref().unwrap();
        let host = host.as_ref().as_ref().unwrap();

        let cuda_ctx = CudaContext::new(0)?;
        let stream = cuda_ctx.new_stream()?;

        let transfer_ctx = TransferContext::new(None, stream.clone());

        let dtoh_pending_offload_manager = PendingOffloadManager::new();

        loop {
            // Try to check the offload queue.
            let request = dtoh_offload_queue.lock().await.pop_first();

            // If there is a request, process it.
            if let Some(request) = request {
                // Try to upgrade the block to a strong reference.
                let block = match request.block.upgrade() {
                    Some(block) => Some(block),
                    // If unable to, try to find the block in the pool.
                    None => device
                        .match_sequence_hashes(vec![request.sequence_hash].as_slice())
                        .await?
                        .pop()
                        .map(|block| block.mutable_block().clone()),
                };

                // If we've found the block, offload it to the host.
                if let Some(block) = block {
                    // Allocate a block from the host pool.
                    // TODO: The most likely error here is that the host pool is full.
                    // It's probably not a good idea to keep consuming queue elements in the meantime.
                    let host_blocks = match host.allocate_blocks(1).await {
                        Ok(blocks) => blocks,
                        Err(_) => {
                            continue;
                        }
                    };

                    if let Some(mut host_block) = host_blocks.into_iter().next() {
                        // Enqueue the offload into the stream.
                        block.write_to(&mut host_block, None, &transfer_ctx)?;

                        // Record an event after the transfer is complete. Use the BLOCKING_SYNC flag to ensure the event is recorded synchronously on the host.
                        let event = transfer_ctx
                            .stream()
                            .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;

                        // Update block metadata and register with host pool.
                        OffloadManager::handle_offload(&block, host_block, host).await?;

                        // Record the pending offload. This may block if too many offloads are already pending.
                        dtoh_pending_offload_manager
                            .handle_pending_offload(PendingOffload::new(block, event))
                            .await?;
                    } // TODO: How should we handle an allocation failure in the host pool?
                }
            } else {
                // If the queue is empty, wait to be notified.
                dtoh_offload_notify.notified().await;
            }
        }
    }

    pub async fn offload<S: Storage>(
        &self,
        block: &ImmutableBlock<S, Metadata>,
        location: CacheLevel,
        priority: u64,
    ) -> Result<()> {
        match block.state() {
            BlockState::Registered(_) => {}
            _ => {
                return Err(anyhow::anyhow!("Only registered blocks may be offloaded."));
            }
        }

        let any_block = block as &dyn Any;

        // For now, only consider offloads from G1 to G2.
        if let Some(device_block) =
            any_block.downcast_ref::<ImmutableBlock<DeviceStorage, Metadata>>()
        {
            if location != CacheLevel::G2 {
                return Err(anyhow::anyhow!("Only offloads to G2 are supported."));
            }

            let mut tick = self.tick.lock().await;
            let key = OffloadRequestKey {
                priority,
                timestamp: *tick,
            };
            // Increment a counter for each block. Within the same priority, blocks with lower counter values are processed first.
            *tick += 1;
            drop(tick);

            self.dtoh_offload_tx
                .send(OffloadManager::build_request(device_block, key)?)?;
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;
    use crate::block_manager::block::test_utils::get_private_token;

    use crate::block_manager::{
        block::{BasicMetadata, BlockDataExt, BlockDataProvider, Blocks},
        layout::FullyContiguous,
        pool::BlockPool,
        storage::{DeviceAllocator, DeviceStorage, PinnedAllocator, PinnedStorage},
        DType, LayoutConfig,
    };

    use cudarc::runtime::sys::{cudaMemcpy, cudaMemcpyKind, cudaMemset};

    const BLOCK_SIZE: usize = 4;

    type DevicePool = Arc<Option<BlockPool<DeviceStorage, BasicMetadata>>>;
    type HostPool = Arc<Option<BlockPool<PinnedStorage, BasicMetadata>>>;

    fn build_pools(
        device_blocks: usize,
        host_blocks: usize,
    ) -> Result<(Arc<OffloadManager<BasicMetadata>>, DevicePool, HostPool)> {
        let mut config = LayoutConfig {
            num_blocks: device_blocks,
            num_layers: 8,
            page_size: BLOCK_SIZE,
            inner_dim: 1024,
            alignment: 1,
            dtype: DType::FP16,
        };

        let device = FullyContiguous::allocate(config.clone(), &DeviceAllocator::default())?;

        config.num_blocks = host_blocks;

        let host = FullyContiguous::allocate(config, &PinnedAllocator::default())?;

        let device_blocks = Blocks::<_, BasicMetadata>::new(device, 42, 0)?.into_blocks()?;
        let host_blocks = Blocks::<_, BasicMetadata>::new(host, 42, 0)?.into_blocks()?;

        let device_pool = Arc::new(Some(BlockPool::builder().blocks(device_blocks).build()?));

        let host_pool = Arc::new(Some(BlockPool::builder().blocks(host_blocks).build()?));

        let manager = OffloadManager::new(device_pool.clone(), host_pool.clone());

        Ok((manager, device_pool, host_pool))
    }

    async fn reset_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
    ) -> Result<MutableBlock<S, Metadata>> {
        pool.allocate_blocks(1)
            .await?
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to allocate block"))
    }

    async fn partial_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
        token: u32,
    ) -> Result<MutableBlock<S, Metadata>> {
        let mut block = reset_block(pool).await?;
        block.init_sequence(42)?;
        block.add_token(token)?;
        Ok(block)
    }

    async fn completed_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
        tokens: [u32; BLOCK_SIZE],
    ) -> Result<MutableBlock<S, Metadata>> {
        let mut block = reset_block(pool).await?;
        block.init_sequence(42)?;
        for token in tokens {
            block.add_token(token)?;
        }
        block.commit()?;
        Ok(block)
    }

    fn populate_block(
        block: &impl BlockDataProvider<StorageType = DeviceStorage>,
        value: i32,
    ) -> Result<()> {
        let block_data = block.block_data(get_private_token());

        for layer in 0..block_data.num_layers() {
            let layer_data = block_data.layer_view(layer)?;
            let layer_size = layer_data.size();

            unsafe {
                cudaMemset(
                    layer_data.as_ptr() as *mut std::ffi::c_void,
                    value,
                    layer_size,
                )
                .result()?;
            }
        }

        Ok(())
    }

    async fn compare_block_contents<Metadata: BlockMetadata>(
        device_block: &ImmutableBlock<DeviceStorage, Metadata>,
        host_block: &ImmutableBlock<PinnedStorage, Metadata>,
    ) -> Result<()> {
        let host_data = host_block.block_data(get_private_token());
        let device_data = device_block.block_data(get_private_token());

        for layer in 0..host_data.num_layers() {
            let host_layer = host_data.layer_view(layer)?;
            let device_layer = device_data.layer_view(layer)?;

            let layer_size = host_layer.size();

            assert_eq!(layer_size, device_layer.size());

            let mut host_buffer = vec![0u8; layer_size];

            let host_slice;

            unsafe {
                cudaMemcpy(
                    host_buffer.as_mut_ptr() as *mut std::ffi::c_void,
                    device_layer.as_ptr() as *const std::ffi::c_void,
                    layer_size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
                .result()?;
                host_slice = std::slice::from_raw_parts(host_buffer.as_ptr(), layer_size);
            }

            assert_eq!(host_buffer, host_slice);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_invalid_blocks() -> Result<()> {
        let (offload_manager, device_pool, _) = build_pools(4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();

        // Check blocks in the 'RESET' state.
        let immutable_block = ImmutableBlock::new(Arc::new(reset_block(device_pool).await?));
        assert!(offload_manager
            .offload(&immutable_block, CacheLevel::G2, 0)
            .await
            .is_err());

        // Check blocks in the 'PARTIAL' state.
        let immutable_block = ImmutableBlock::new(Arc::new(partial_block(device_pool, 0).await?));
        assert!(offload_manager
            .offload(&immutable_block, CacheLevel::G2, 0)
            .await
            .is_err());

        // Check blocks in the 'COMPLETED' state.
        let immutable_block = ImmutableBlock::new(Arc::new(
            completed_block(device_pool, [0; BLOCK_SIZE]).await?,
        ));
        assert!(offload_manager
            .offload(&immutable_block, CacheLevel::G2, 0)
            .await
            .is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_registered_blocks() -> Result<()> {
        let (offload_manager, device_pool, host_pool) = build_pools(4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        // Create a block and register it with the offload manager
        let block = completed_block(device_pool, [0, 1, 2, 3]).await?;

        let immutable_device_block = device_pool
            .register_blocks(vec![block])
            .await?
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to register block"))?;

        assert!(offload_manager
            .offload(&immutable_device_block, CacheLevel::G1, 0)
            .await
            .is_err());
        assert!(offload_manager
            .offload(&immutable_device_block, CacheLevel::G3, 0)
            .await
            .is_err());
        assert!(offload_manager
            .offload(&immutable_device_block, CacheLevel::G4, 0)
            .await
            .is_err());

        populate_block(&immutable_device_block, 42)?;

        // Offloads should only go to G2 (for now)
        offload_manager
            .offload(&immutable_device_block, CacheLevel::G2, 0)
            .await?;

        // Wait for it to be processed.
        // TODO: This is a bit of a hack, and may lead to non-deterministic behavior.
        // In theory, the offload + memcpy should take much less time than this.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the block exists in the host pool
        let host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?;

        assert_eq!(host_blocks.len(), 1);
        assert_eq!(
            host_blocks[0].sequence_hash()?,
            immutable_device_block.sequence_hash()?
        );

        compare_block_contents(&immutable_device_block, &host_blocks[0]).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_no_host_blocks_available() -> Result<()> {
        let (offload_manager, device_pool, host_pool) = build_pools(4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        let host_blocks = host_pool.allocate_blocks(4).await?;
        assert_eq!(host_blocks.len(), 4);

        let device_block = completed_block(device_pool, [0, 1, 2, 3]).await?;
        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        offload_manager
            .offload(&immutable_device_block, CacheLevel::G2, 0)
            .await?;

        // Wait for offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // The offload should fail due to a lack of host blocks.
        let matched_host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(matched_host_blocks.len(), 0);

        // Wait for blocks to be returned to the pool.
        drop(host_blocks);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Try the offload again.
        offload_manager
            .offload(&immutable_device_block, CacheLevel::G2, 0)
            .await?;

        // Wait for offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // This time, the offload should succeed.
        let matched_host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(matched_host_blocks.len(), 1);

        Ok(())
    }
}
