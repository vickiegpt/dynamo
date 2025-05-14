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

use crate::block_manager::block::{
    transfer::WriteTo, transfer::WriteToStrategy, BlockError, BlockExt, BlockMetadata, BlockState,
    ImmutableBlock, MutableBlock, ReadableBlock, WritableBlock,
};
use crate::block_manager::pool::BlockPoolError;
use crate::block_manager::state::TransferContext;
use crate::block_manager::storage::{DeviceStorage, DiskStorage, Local, PinnedStorage, Storage};
use crate::block_manager::BlockPool;
use anyhow::Result;
use async_trait::async_trait;
use cudarc::driver::{sys::CUevent_flags, CudaEvent};
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};
use tokio::sync::mpsc;

type BlockResult<Target, Metadata> = Result<Vec<ImmutableBlock<Target, Metadata>>, BlockPoolError>;

/// Manage a set of pending transfers.
pub struct PendingTransfer<Source: Storage, Target: Storage, Metadata: BlockMetadata> {
    /// The block being copied from.
    sources: Vec<Arc<MutableBlock<Source, Metadata>>>,
    /// The block being copied to.
    targets: Vec<MutableBlock<Target, Metadata>>,
    /// The oneshot sender that optionally returns the registered blocks once the transfer is complete.
    completion_indicator: Option<oneshot::Sender<BlockResult<Target, Metadata>>>,
    /// The target pool that will receive the registered block.
    target_registration_pool: Arc<Option<BlockPool<Target, Metadata>>>,
}

impl<Source: Storage, Target: Storage, Metadata: BlockMetadata>
    PendingTransfer<Source, Target, Metadata>
{
    pub fn new(
        sources: Vec<Arc<MutableBlock<Source, Metadata>>>,
        targets: Vec<MutableBlock<Target, Metadata>>,
        completion_indicator: Option<oneshot::Sender<BlockResult<Target, Metadata>>>,
        target_registration_pool: Arc<Option<BlockPool<Target, Metadata>>>,
    ) -> Self {
        Self {
            sources,
            targets,
            completion_indicator,
            target_registration_pool,
        }
    }

    fn handle_complete(self) -> Result<()> {
        let Self {
            targets,
            target_registration_pool,
            completion_indicator,
            ..
        } = self;

        if let Some(target_registration_pool) = target_registration_pool.as_ref() {
            let blocks = target_registration_pool.register_blocks_blocking(targets)?;

            if let Some(completion_indicator) = completion_indicator {
                completion_indicator.send(Ok(blocks))?;
            }
        }

        Ok(())
    }
}

fn transfer_metadata<Source: Storage, Target: Storage, Metadata: BlockMetadata>(
    source: &Arc<MutableBlock<Source, Metadata>>,
    target: &mut MutableBlock<Target, Metadata>,
) -> Result<()> {
    // Only registered blocks can be transferred. There are upstream checks for this, so this shouldn't ever fail.
    if let BlockState::Registered(reg_handle) = source.state() {
        // Bring the block back to the 'Reset' state.
        target.reset();
        // Transfer metadata.
        target.update_metadata(source.metadata().clone());
        // Copy tokens
        target.apply_token_block(reg_handle.token_block().clone())?;
    } else {
        Err(BlockPoolError::BlockError(BlockError::InvalidState(
            "Block is not registered.".to_string(),
        )))?;
    }

    Ok(())
}

#[async_trait]
pub trait PendingTransferManager<Source: Storage, Target: Storage, Metadata: BlockMetadata>:
    Send + Sync
{
    /// Begin a transfer. Blocks if the pending queue is full.
    async fn begin_transfer(
        &self,
        pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()>;
}

pub struct PendingCudaTransferManager<Source: Storage, Target: Storage, Metadata: BlockMetadata> {
    pending_transfer_q: mpsc::Sender<(PendingTransfer<Source, Target, Metadata>, CudaEvent)>,
    transfer_ctx: Arc<TransferContext>,
}

impl<Source: Storage, Target: Storage, Metadata: BlockMetadata>
    PendingCudaTransferManager<Source, Target, Metadata>
{
    pub fn new(max_depth: usize, transfer_ctx: Arc<TransferContext>) -> Self {
        let (tx, mut rx) =
            mpsc::channel::<(PendingTransfer<Source, Target, Metadata>, CudaEvent)>(max_depth);

        spawn(move || {
            while let Some((pending_transfer, event)) = rx.blocking_recv() {
                // Wait for the event.
                event.synchronize()?;
                pending_transfer.handle_complete()?;
            }
            Ok::<(), anyhow::Error>(())
        });

        Self {
            pending_transfer_q: tx,
            transfer_ctx,
        }
    }
}

#[async_trait]
impl<Source, Target, Metadata> PendingTransferManager<Source, Target, Metadata>
    for PendingCudaTransferManager<Source, Target, Metadata>
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
    async fn begin_transfer(
        &self,
        mut pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()> {
        for (source, target) in pending_transfer
            .sources
            .iter()
            .zip(pending_transfer.targets.iter_mut())
        {
            transfer_metadata(source, target)?;
            source.write_to(target, None, self.transfer_ctx.as_ref())?;
        }

        let event = self
            .transfer_ctx
            .stream()
            .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;

        self.pending_transfer_q
            .send((pending_transfer, event))
            .await?;

        Ok(())
    }
}

pub struct PendingDiskTransferManager<Source: Storage, Target: Storage, Metadata: BlockMetadata> {
    transfer_q: Arc<mpsc::Sender<PendingTransfer<Source, Target, Metadata>>>,
}

impl<Source: Storage, Target: Storage, Metadata: BlockMetadata>
    PendingDiskTransferManager<Source, Target, Metadata>
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
    pub fn new(max_depth: usize, num_workers: usize, transfer_ctx: Arc<TransferContext>) -> Self {
        let (transfer_tx, transfer_rx) =
            mpsc::channel::<PendingTransfer<Source, Target, Metadata>>(max_depth);

        let transfer_rx = Arc::new(Mutex::new(transfer_rx));
        for _ in 0..num_workers {
            let transfer_rx = transfer_rx.clone();
            let transfer_ctx = transfer_ctx.clone();
            let _: JoinHandle<Result<()>> = spawn(move || loop {
                let mut transfer_rx = transfer_rx.lock().unwrap();

                if let Some(mut pending_transfer) = transfer_rx.blocking_recv() {
                    drop(transfer_rx);

                    for (source, target) in pending_transfer
                        .sources
                        .iter()
                        .zip(pending_transfer.targets.iter_mut())
                    {
                        transfer_metadata(source, target)?;
                        source.write_to(target, None, transfer_ctx.as_ref())?;
                    }
    
                    pending_transfer.handle_complete()?;
                } else {
                    return Ok(())
                }

            });
        }

        let transfer_q = Arc::new(transfer_tx);
        Self { transfer_q }
    }
}

#[async_trait]
impl<Source, Target, Metadata> PendingTransferManager<Source, Target, Metadata>
    for PendingDiskTransferManager<Source, Target, Metadata>
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
    async fn begin_transfer(
        &self,
        pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()> {
        self.transfer_q.send(pending_transfer).await?;

        Ok(())
    }
}

pub struct PendingDiskOnboardManager<Metadata: BlockMetadata> {
    disk_transfer_manager: Arc<PendingDiskTransferManager<DiskStorage, PinnedStorage, Metadata>>,
    cuda_transfer_manager: Arc<PendingCudaTransferManager<PinnedStorage, DeviceStorage, Metadata>>,
    host_pool: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
}

impl<Metadata: BlockMetadata> PendingDiskOnboardManager<Metadata> {
    pub fn new(
        num_workers: usize,
        transfer_ctx: Arc<TransferContext>,
        host_pool: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
    ) -> Self {
        let disk_transfer_manager = Arc::new(PendingDiskTransferManager::new(
            16384,
            num_workers,
            transfer_ctx.clone(),
        ));
        let cuda_transfer_manager =
            Arc::new(PendingCudaTransferManager::new(16384, transfer_ctx.clone()));

        Self {
            disk_transfer_manager,
            cuda_transfer_manager,
            host_pool,
        }
    }
}

#[async_trait]
impl<Metadata: BlockMetadata> PendingTransferManager<DiskStorage, DeviceStorage, Metadata>
    for PendingDiskOnboardManager<Metadata>
{
    async fn begin_transfer(
        &self,
        pending_transfer: PendingTransfer<DiskStorage, DeviceStorage, Metadata>,
    ) -> Result<()> {
        let disk_transfer_manager = self.disk_transfer_manager.clone();
        let cuda_transfer_manager = self.cuda_transfer_manager.clone();
        let host_pool = self.host_pool.clone();

        tokio::spawn(async move {
            let PendingTransfer {
                sources: disk_sources,
                targets: device_targets,
                completion_indicator: device_completion_indicator,
                target_registration_pool: device_target_registration_pool,
            } = pending_transfer;

            let host = host_pool.as_ref().as_ref().unwrap();

            let host_blocks = host.allocate_blocks(disk_sources.len()).await?;

            let (host_completion_indicator, host_completion_rx) = oneshot::channel();

            disk_transfer_manager
                .begin_transfer(PendingTransfer::new(
                    disk_sources,
                    host_blocks,
                    Some(host_completion_indicator),
                    host_pool.clone(),
                ))
                .await?;

            let host_blocks = host_completion_rx
                .await
                .unwrap()?
                .iter()
                .map(|b| b.mutable_block().clone())
                .collect();

            cuda_transfer_manager
                .begin_transfer(PendingTransfer::new(
                    host_blocks,
                    device_targets,
                    device_completion_indicator,
                    device_target_registration_pool,
                ))
                .await?;

            Ok::<(), anyhow::Error>(())
        });

        Ok(())
    }
}
