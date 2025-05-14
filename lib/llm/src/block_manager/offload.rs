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

use super::block::{
    transfer::WriteToStrategy, BlockError, BlockMetadata, BlockState, ImmutableBlock, MutableBlock,
    ReadableBlock, WritableBlock,
};
use super::pool::BlockPoolError;
use super::state::TransferContext;
use super::storage::{Cuda, Local, Storage};
use super::{BlockPool, DeviceStorage, DiskStorage, PinnedStorage};

use anyhow::Result;
use std::any::Any;

use std::collections::BTreeSet;

mod pending;
mod request;

use pending::{
    PendingCudaTransferManager, PendingDiskTransferManager, PendingTransfer, PendingTransferManager, PendingDiskOnboardManager
};
use request::{OffloadRequest, OffloadRequestKey, OnboardRequest};

const MAX_OFFLOAD_STREAM_DEPTH: usize = 4;
const DISK_TRANSFER_NUM_WORKERS: usize = 4;

/// The offload manager handles all block transfers between different cache levels.
pub struct OffloadManager<Metadata: BlockMetadata> {
    device: Arc<Option<BlockPool<DeviceStorage, Metadata>>>,
    host: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
    disk: Arc<Option<BlockPool<DiskStorage, Metadata>>>,

    /// Priority queue of pending offloads
    dtoh_offload_queue: Arc<Mutex<BTreeSet<OffloadRequest<DeviceStorage, Metadata>>>>,
    /// Used to notify the offload worker that an item has been added to the priority queue
    dtoh_offload_notify: Arc<Notify>,

    htodisk_offload_queue: Arc<Mutex<BTreeSet<OffloadRequest<PinnedStorage, Metadata>>>>,
    htodisk_offload_notify: Arc<Notify>,

    /// An incrementing counter for offloaded blocks. Within the same priority, blocks with lower tick values are processed first.
    tick: Arc<Mutex<u64>>,

    /// Queue of pending onboarding requests.
    htod_onboard_tx: mpsc::UnboundedSender<OnboardRequest<PinnedStorage, DeviceStorage, Metadata>>,
    disktod_onboard_tx: mpsc::UnboundedSender<OnboardRequest<DiskStorage, DeviceStorage, Metadata>>,
}

impl<Metadata: BlockMetadata> OffloadManager<Metadata> {
    pub fn new(
        device: Arc<Option<BlockPool<DeviceStorage, Metadata>>>,
        host: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
        disk: Arc<Option<BlockPool<DiskStorage, Metadata>>>,
    ) -> Result<Arc<Self>> {
        let dtoh_offload_queue = Arc::new(Mutex::new(BTreeSet::new()));
        let dtoh_offload_notify = Arc::new(Notify::new());

        let htodisk_offload_queue = Arc::new(Mutex::new(BTreeSet::new()));
        let htodisk_offload_notify = Arc::new(Notify::new());

        let (htod_onboard_tx, htod_onboard_rx) = mpsc::unbounded_channel();
        let (disktod_onboard_tx, disktod_onboard_rx) = mpsc::unbounded_channel();

        let this = Arc::new(Self {
            device,
            host,
            disk,
            dtoh_offload_queue,
            dtoh_offload_notify,
            htodisk_offload_queue,
            htodisk_offload_notify,
            tick: Arc::new(Mutex::new(0)),
            htod_onboard_tx,
            disktod_onboard_tx,
        });

        let cuda_ctx = Cuda::device_or_create(0)?;
        let offload_transfer_ctx = Arc::new(TransferContext::new(None, cuda_ctx.new_stream()?));

        // Device -> Host offload
        let device_clone = this.device.clone();
        let host_clone = this.host.clone();
        let dtoh_offload_queue = this.dtoh_offload_queue.clone();
        let dtoh_offload_notify = this.dtoh_offload_notify.clone();
        let offload_transfer_ctx_clone = offload_transfer_ctx.clone();
        tokio::spawn(async move {
            OffloadManager::offload_worker(
                device_clone,
                host_clone,
                dtoh_offload_queue,
                dtoh_offload_notify,
                Arc::new(PendingCudaTransferManager::new(
                    MAX_OFFLOAD_STREAM_DEPTH,
                    offload_transfer_ctx_clone,
                )),
            )
            .await
        });

        // Host -> Disk offload
        let host_clone = this.host.clone();
        let disk_clone = this.disk.clone();
        let htodisk_offload_queue = this.htodisk_offload_queue.clone();
        let htodisk_offload_notify = this.htodisk_offload_notify.clone();
        let offload_transfer_ctx_clone = offload_transfer_ctx.clone();
        tokio::spawn(async move {
            OffloadManager::offload_worker(
                host_clone,
                disk_clone,
                htodisk_offload_queue,
                htodisk_offload_notify,
                Arc::new(PendingDiskTransferManager::new(
                    MAX_OFFLOAD_STREAM_DEPTH,
                    DISK_TRANSFER_NUM_WORKERS,
                    offload_transfer_ctx_clone,
                )),
            )
            .await
        });

        let onboard_transfer_ctx = Arc::new(TransferContext::new(None, cuda_ctx.new_stream()?));

        // Host -> Device onboarding
        let host_clone = this.host.clone();
        let device_clone = this.device.clone();
        let onboard_transfer_ctx_clone = onboard_transfer_ctx.clone();
        tokio::spawn(async move {
            OffloadManager::onboard_worker(
                host_clone,
                device_clone,
                htod_onboard_rx,
                Arc::new(PendingCudaTransferManager::new(
                    16384,
                    onboard_transfer_ctx_clone,
                )),
            )
            .await
        });

        // Disk -> Device onboarding
        let disk_clone = this.disk.clone();
        let host_clone = this.host.clone();
        let device_clone = this.device.clone();
        let onboard_transfer_ctx_clone = onboard_transfer_ctx.clone();

        tokio::spawn(async move {
            OffloadManager::onboard_worker(
                disk_clone,
                device_clone,
                disktod_onboard_rx,
                Arc::new(PendingDiskOnboardManager::new(
                    DISK_TRANSFER_NUM_WORKERS,
                    onboard_transfer_ctx_clone,
                    host_clone
                )),
            )
        });
        Ok(this)
    }

    async fn offload_worker<Source, Target>(
        source_pool_arc: Arc<Option<BlockPool<Source, Metadata>>>,
        target_pool_arc: Arc<Option<BlockPool<Target, Metadata>>>,
        offload_queue: Arc<Mutex<BTreeSet<OffloadRequest<Source, Metadata>>>>,
        offload_notify: Arc<Notify>,
        transfer_manager: Arc<dyn PendingTransferManager<Source, Target, Metadata>>,
    ) -> Result<()>
    where
        Source: Storage,
        Target: Storage,
        Metadata: BlockMetadata,
        // Check that the source block is readable, local, and writable to the target block.
        MutableBlock<Source, Metadata>: ReadableBlock<StorageType = Source>
            + Local
            + WriteToStrategy<MutableBlock<Target, Metadata>>,
        // Check that the target block is writable.
        MutableBlock<Target, Metadata>: WritableBlock<StorageType = Target>,
    {
        if source_pool_arc.is_none() || target_pool_arc.is_none() {
            return Ok(());
        }

        let source_pool = source_pool_arc.as_ref().as_ref().unwrap();
        let target_pool = target_pool_arc.as_ref().as_ref().unwrap();

        loop {
            // Try to check the offload queue.
            let request = offload_queue.lock().await.pop_first();

            // If there is a request, process it.
            if let Some(request) = request {
                // Try to upgrade the block to a strong reference.
                let block = match request.block.upgrade() {
                    Some(block) => Some(block),
                    // If unable to upgrade, the block may have been moved to the inactive pool.
                    None => source_pool
                        .match_sequence_hashes(vec![request.sequence_hash].as_slice())
                        .await?
                        .pop()
                        .map(|block| block.mutable_block().clone()),
                };

                // If we've found the block, offload it.
                if let Some(block) = block {
                    // Allocate a block from the host pool.
                    // TODO: The most likely error here is that the host pool is full.
                    // It's probably not a good idea to keep consuming queue elements in the meantime.
                    let target_blocks = match target_pool.allocate_blocks(1).await {
                        Ok(blocks) => blocks,
                        Err(_) => {
                            continue;
                        }
                    };

                    if let Some(target_block) = target_blocks.into_iter().next() {
                        transfer_manager
                            .begin_transfer(PendingTransfer::new(
                                vec![block],
                                vec![target_block],
                                None,
                                target_pool_arc.clone(),
                            ))
                            .await?;
                    }
                }
            } else {
                // If the queue is empty, wait to be notified.
                offload_notify.notified().await;
            }
        }
    }

    async fn onboard_worker<Source, Target>(
        source_pool_arc: Arc<Option<BlockPool<Source, Metadata>>>,
        target_pool_arc: Arc<Option<BlockPool<Target, Metadata>>>,
        mut onboard_rx: mpsc::UnboundedReceiver<OnboardRequest<Source, Target, Metadata>>,
        transfer_manager: Arc<dyn PendingTransferManager<Source, Target, Metadata>>,
    ) -> Result<()>
    where
        Source: Storage,
        Target: Storage,
        Metadata: BlockMetadata,
        // Check that the source block is readable, local, and writable to the target block.
        MutableBlock<Source, Metadata>: ReadableBlock<StorageType = Source>
            + Local
            + WriteToStrategy<MutableBlock<Target, Metadata>>,
        // Check that the target block is writable.
        MutableBlock<Target, Metadata>: WritableBlock<StorageType = Target>,
    {
        if source_pool_arc.is_none() || target_pool_arc.is_none() {
            return Ok(());
        }

        let target_pool = target_pool_arc.as_ref().as_ref().unwrap();

        while let Some(request) = onboard_rx.recv().await {
            let target_blocks = match target_pool.allocate_blocks(request.blocks.len()).await {
                Ok(blocks) => blocks,
                Err(err) => {
                    request.response_tx.send(Err(err))?;
                    continue;
                }
            };

            let sources = request
                .blocks
                .iter()
                .map(|b| b.mutable_block().clone())
                .collect();

            transfer_manager
                .begin_transfer(PendingTransfer::new(
                    sources,
                    target_blocks,
                    Some(request.response_tx),
                    target_pool_arc.clone(),
                ))
                .await?;
        }
        Ok(())
    }

    pub async fn offload<S: Storage>(
        &self,
        block: &ImmutableBlock<S, Metadata>,
        priority: u64,
    ) -> core::result::Result<(), BlockPoolError> {
        match block.state() {
            BlockState::Registered(_) => {}
            _ => {
                return Err(BlockPoolError::BlockError(BlockError::InvalidState(
                    "Block is not registered.".to_string(),
                )));
            }
        }
        // This can get called by all pools, regardless of whether or not they have a place to offload to.
        // Because of this, we need to check the block type here.
        let any_block = block as &dyn Any;

        let mut tick = self.tick.lock().await;
        let key = OffloadRequestKey {
            priority,
            timestamp: *tick,
        };
        // Increment a counter for each block. Within the same priority, blocks with lower counter values are processed first.
        *tick += 1;
        drop(tick);

        // TODO: What's the performance penalty of this runtime type-checking?
        if let Some(device_block) =
            any_block.downcast_ref::<ImmutableBlock<DeviceStorage, Metadata>>()
        {
            if self.device.is_none() || self.host.is_none() {
                return Ok(());
            }

            let request = OffloadRequest {
                block: Arc::downgrade(device_block.mutable_block()),
                sequence_hash: device_block.sequence_hash()?,
                key,
            };

            self.dtoh_offload_queue.lock().await.insert(request);
            self.dtoh_offload_notify.notify_one();
        } else if let Some(pinned_block) =
            any_block.downcast_ref::<ImmutableBlock<PinnedStorage, Metadata>>()
        {
            if self.host.is_none() || self.disk.is_none() {
                return Ok(());
            }

            let request = OffloadRequest {
                block: Arc::downgrade(pinned_block.mutable_block()),
                sequence_hash: pinned_block.sequence_hash()?,
                key,
            };

            self.htodisk_offload_queue.lock().await.insert(request);
            self.htodisk_offload_notify.notify_one();
        }

        Ok(())
    }

    pub async fn onboard<S: Storage>(
        &self,
        blocks: Vec<ImmutableBlock<S, Metadata>>,
    ) -> core::result::Result<Vec<ImmutableBlock<DeviceStorage, Metadata>>, BlockPoolError> {
        for block in &blocks {
            match block.state() {
                BlockState::Registered(_) => {}
                _ => {
                    return Err(BlockPoolError::BlockError(BlockError::InvalidState(
                        "Block is not registered.".to_string(),
                    )));
                }
            }
        }

        if blocks.is_empty() {
            return Ok(vec![]);
        }

        let (tx, rx) = oneshot::channel();

        let any_block = blocks.first().unwrap() as &dyn Any;

        if any_block.is::<ImmutableBlock<PinnedStorage, Metadata>>() {
            let host_blocks = blocks
                .iter()
                .map(|b| {
                    (b as &dyn Any)
                        .downcast_ref::<ImmutableBlock<PinnedStorage, Metadata>>()
                        .unwrap()
                        .clone()
                })
                .collect();

            self.htod_onboard_tx
                .send(OnboardRequest::new(host_blocks, tx))
                .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;
        } else if any_block.is::<ImmutableBlock<DiskStorage, Metadata>>() {
            let disk_blocks = blocks
                .iter()
                .map(|b| {
                    (b as &dyn Any)
                        .downcast_ref::<ImmutableBlock<DiskStorage, Metadata>>()
                        .unwrap()
                        .clone()
                })
                .collect();

            self.disktod_onboard_tx
                .send(OnboardRequest::new(disk_blocks, tx))
                .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;
        } else {
            // Block must be pinned or disk.
            return Err(BlockPoolError::BlockError(BlockError::InvalidState(
                "Block is not a pinned or disk block.".to_string(),
            )));
        }

        match rx.await {
            Ok(res) => res,
            Err(_) => Err(BlockPoolError::ProgressEngineShutdown),
        }
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;
    use crate::block_manager::block::test_utils::get_private_token;

    use crate::block_manager::storage::SystemAccessible;
    use crate::block_manager::{
        block::{
            BasicMetadata, BlockDataExt, BlockDataProvider, BlockExt, Blocks, ImmutableBlock,
            MutableBlock,
        },
        layout::FullyContiguous,
        pool::BlockPool,
        storage::{
            cuda::CudaAccessible, DeviceAllocator, DeviceStorage, DiskAllocator, DiskStorage,
            PinnedAllocator, PinnedStorage,
        },
        DType, LayoutConfig,
    };
    use nixl_sys::NixlDescriptor;

    use cudarc::runtime::sys::{cudaMemcpy, cudaMemcpyKind, cudaMemset};

    const BLOCK_SIZE: usize = 4;

    type DevicePool = Arc<Option<BlockPool<DeviceStorage, BasicMetadata>>>;
    type HostPool = Arc<Option<BlockPool<PinnedStorage, BasicMetadata>>>;
    type DiskPool = Arc<Option<BlockPool<DiskStorage, BasicMetadata>>>;

    fn build_pools(
        device_blocks: usize,
        host_blocks: usize,
        disk_blocks: usize,
    ) -> Result<(
        Arc<OffloadManager<BasicMetadata>>,
        DevicePool,
        HostPool,
        DiskPool,
    )> {
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

        let host = FullyContiguous::allocate(config.clone(), &PinnedAllocator::default())?;

        config.num_blocks = disk_blocks;

        let disk = FullyContiguous::allocate(config, &DiskAllocator)?;

        let device_blocks = Blocks::<_, BasicMetadata>::new(device, 42, 0)?.into_blocks()?;
        let host_blocks = Blocks::<_, BasicMetadata>::new(host, 42, 0)?.into_blocks()?;
        let disk_blocks = Blocks::<_, BasicMetadata>::new(disk, 42, 0)?.into_blocks()?;

        let device_pool = Arc::new(Some(BlockPool::builder().blocks(device_blocks).build()?));

        let host_pool = Arc::new(Some(BlockPool::builder().blocks(host_blocks).build()?));

        let disk_pool = Arc::new(Some(BlockPool::builder().blocks(disk_blocks).build()?));

        let manager =
            OffloadManager::new(device_pool.clone(), host_pool.clone(), disk_pool.clone())?;

        Ok((manager, device_pool, host_pool, disk_pool))
    }

    /// Create a block in the 'RESET' state.
    async fn get_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
    ) -> Result<MutableBlock<S, Metadata>> {
        pool.allocate_blocks(1)
            .await?
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to allocate block"))
    }

    /// Create a block in the 'PARTIAL' state.
    async fn partial_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
        token: u32,
    ) -> Result<MutableBlock<S, Metadata>> {
        let mut block = get_block(pool).await?;
        block.init_sequence(42)?;
        block.add_token(token)?;
        Ok(block)
    }

    /// Create a block in the 'COMPLETED' state.
    async fn completed_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
        tokens: [u32; BLOCK_SIZE],
    ) -> Result<MutableBlock<S, Metadata>> {
        let mut block = get_block(pool).await?;
        block.init_sequence(42)?;
        for token in tokens {
            block.add_token(token)?;
        }
        block.commit()?;
        Ok(block)
    }

    fn populate_cuda_block<S: Storage + CudaAccessible + NixlDescriptor>(
        block: &impl BlockDataProvider<StorageType = S>,
        value: i32,
    ) -> Result<()> {
        let block_data = block.block_data(get_private_token()).block_view()?;
        let block_size = block_data.size();

        unsafe {
            cudaMemset(
                block_data.as_ptr() as *mut std::ffi::c_void,
                value,
                block_size,
            )
            .result()?;
        }
        Ok(())
    }

    /// Compare the contents of a device block and a host block.
    async fn compare_block_contents<S: Storage + SystemAccessible + NixlDescriptor>(
        device_block: &impl BlockDataProvider<StorageType = DeviceStorage>,
        host_block: &impl BlockDataProvider<StorageType = S>,
    ) -> Result<()> {
        let host_data = host_block.block_data(get_private_token()).block_view()?;
        let device_data = device_block.block_data(get_private_token()).block_view()?;

        let size = host_data.size();

        assert_eq!(size, device_data.size());

        let mut host_buffer = vec![0u8; size];
        let host_slice;

        unsafe {
            cudaMemcpy(
                host_buffer.as_mut_ptr() as *mut std::ffi::c_void,
                device_data.as_ptr() as *const std::ffi::c_void,
                size,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
            .result()?;
            host_slice = std::slice::from_raw_parts(host_buffer.as_ptr(), size);
        }

        assert_eq!(host_buffer, host_slice);

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_invalid_blocks() -> Result<()> {
        let (offload_manager, device_pool, _, _) = build_pools(4, 4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();

        // Check blocks in the 'RESET' state.
        let immutable_block = ImmutableBlock::new(Arc::new(get_block(device_pool).await?));

        assert!(matches!(
            offload_manager.offload(&immutable_block, 0).await,
            Err(BlockPoolError::BlockError(BlockError::InvalidState(_)))
        ));

        // Check blocks in the 'PARTIAL' state.
        let immutable_block = ImmutableBlock::new(Arc::new(partial_block(device_pool, 0).await?));
        assert!(matches!(
            offload_manager.offload(&immutable_block, 0).await,
            Err(BlockPoolError::BlockError(BlockError::InvalidState(_)))
        ));

        // Check blocks in the 'COMPLETED' state.
        let immutable_block = ImmutableBlock::new(Arc::new(
            completed_block(device_pool, [0; BLOCK_SIZE]).await?,
        ));
        assert!(matches!(
            offload_manager.offload(&immutable_block, 0).await,
            Err(BlockPoolError::BlockError(BlockError::InvalidState(_)))
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_registered_blocks() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, 4, 4)?;

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

        populate_cuda_block(&immutable_device_block, 42)?;

        // Offloads should only go to G2 (for now)
        offload_manager.offload(&immutable_device_block, 0).await?;

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
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, 4, 4)?;

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

        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // The offload should fail gracefuly due to a lack of host blocks
        let matched_host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(matched_host_blocks.len(), 0);

        // Wait for blocks to be returned to the pool.
        drop(host_blocks);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Try the offload again.
        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // This time, the offload should succeed.
        let matched_host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(matched_host_blocks.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_onboard() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, 4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        // Allocate and fill a block on the host.
        let host_block = completed_block(host_pool, [0, 1, 2, 3]).await?;
        let immutable_host_block = host_pool
            .register_blocks(vec![host_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_cuda_block(&immutable_host_block, 42)?;

        // Onboard the block.
        let onboarded_blocks = offload_manager
            .onboard(vec![immutable_host_block.clone()])
            .await?;

        assert_eq!(onboarded_blocks.len(), 1);
        // Check that the sequence hash is the same.
        assert_eq!(
            onboarded_blocks[0].sequence_hash()?,
            immutable_host_block.sequence_hash()?
        );
        // Check that the block is registered.
        assert!(matches!(
            onboarded_blocks[0].state(),
            BlockState::Registered(_)
        ));

        compare_block_contents(&onboarded_blocks[0], &immutable_host_block).await?;

        // Wait for the new value to show up in the device pool.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let device_blocks = device_pool
            .match_sequence_hashes(vec![onboarded_blocks[0].sequence_hash()?].as_slice())
            .await?;
        assert_eq!(device_blocks.len(), 1);
        assert_eq!(
            device_blocks[0].sequence_hash()?,
            onboarded_blocks[0].sequence_hash()?
        );

        // Check that this is the same block.
        compare_block_contents(&device_blocks[0], &immutable_host_block).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_onboard() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, 4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        let device_block = completed_block(device_pool, [0, 1, 2, 3]).await?;
        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_cuda_block(&immutable_device_block, 42)?;
        // Offload the block to the host.
        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for the offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the block exists in the host pool.
        let immutable_host_block = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?
            .into_iter()
            .next()
            .unwrap();

        compare_block_contents(&immutable_device_block, &immutable_host_block).await?;

        // Remove the device block from the pool by dropping it and allocating more blocks.
        drop(immutable_device_block);

        // Wait for the block to be returned to the pool.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let device_blocks = device_pool.allocate_blocks(4).await?;
        assert_eq!(device_blocks.len(), 4);

        drop(device_blocks);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the block is not in the device pool.
        let device_blocks = device_pool
            .match_sequence_hashes(vec![immutable_host_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(device_blocks.len(), 0);

        // Onboard the block back to the device pool.
        let onboarded_blocks = offload_manager
            .onboard(vec![immutable_host_block.clone()])
            .await?;
        assert_eq!(onboarded_blocks.len(), 1);
        assert_eq!(
            onboarded_blocks[0].sequence_hash()?,
            immutable_host_block.sequence_hash()?
        );
        assert!(matches!(
            onboarded_blocks[0].state(),
            BlockState::Registered(_)
        ));

        compare_block_contents(&onboarded_blocks[0], &immutable_host_block).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_onboard_err_handling() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, 4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        let host_block = completed_block(host_pool, [0, 1, 2, 3]).await?;
        let immutable_host_block = host_pool
            .register_blocks(vec![host_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        let device_blocks = device_pool.allocate_blocks(4).await?;
        assert_eq!(device_blocks.len(), 4);

        let res = offload_manager
            .onboard(vec![immutable_host_block.clone()])
            .await;
        assert!(matches!(
            res.err().unwrap(),
            BlockPoolError::NotEnoughBlocksAvailable(_, _)
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_disk() -> Result<()> {
        let (offload_manager, device_pool, host_pool, disk_pool) = build_pools(4, 4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();
        let disk_pool = disk_pool.as_ref().as_ref().unwrap();

        let device_block = completed_block(device_pool, [0, 1, 2, 3]).await?;
        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_cuda_block(&immutable_device_block, 42)?;

        offload_manager.offload(&immutable_device_block, 0).await?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let immutable_host_block = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?
            .into_iter()
            .next()
            .unwrap();

        offload_manager.offload(&immutable_host_block, 0).await?;

        // Wait for the offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the block exists in the disk pool.
        let disk_blocks = disk_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(disk_blocks.len(), 1);
        assert_eq!(
            disk_blocks[0].sequence_hash()?,
            immutable_device_block.sequence_hash()?
        );

        compare_block_contents(&immutable_device_block, &disk_blocks[0]).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_onboard_disk() -> Result<()> {
        let (offload_manager, device_pool, host_pool, disk_pool) = build_pools(4, 4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();
        let disk_pool = disk_pool.as_ref().as_ref().unwrap();

        let disk_block = completed_block(disk_pool, [0, 1, 2, 3]).await?;

        let immutable_disk_block = disk_pool
            .register_blocks(vec![disk_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        let device_blocks = offload_manager.onboard(vec![immutable_disk_block.clone()]).await?;

        assert_eq!(device_blocks.len(), 1);
        assert_eq!(
            device_blocks[0].sequence_hash()?,
            immutable_disk_block.sequence_hash()?
        );

        compare_block_contents(&device_blocks[0], &immutable_disk_block).await?;

        Ok(())
    }
}
