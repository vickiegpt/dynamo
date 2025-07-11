// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use nixl_sys::NixlDescriptor;
use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    block::{
        data::local::LocalBlockData,
        debug::hash_block_contents,
        locality,
        transfer::{TransferContext, WriteTo, WriteToStrategy},
        Block, BlockDataProvider, BlockDataProviderMut, ReadableBlock, WritableBlock,
    },
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
    BasicMetadata, Storage,
};

use anyhow::Result;
use async_trait::async_trait;
use std::{any::Any, sync::Arc};

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
pub struct BlockTransferHandler {
    device: Option<LocalBlockDataList<DeviceStorage>>,
    host: Option<LocalBlockDataList<PinnedStorage>>,
    disk: Option<LocalBlockDataList<DiskStorage>>,
    context: Arc<TransferContext>,
}

impl BlockTransferHandler {
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
    ) -> Result<Self> {
        Ok(Self {
            device: Self::get_local_data(device_blocks),
            host: Self::get_local_data(host_blocks),
            disk: Self::get_local_data(disk_blocks),
            context,
        })
    }

    fn get_local_data<S: Storage>(
        blocks: Option<Vec<LocalBlock<S, BasicMetadata>>>,
    ) -> Option<LocalBlockDataList<S>> {
        blocks.map(|blocks| {
            blocks
                .into_iter()
                .map(|b| {
                    let block_data = b.block_data() as &dyn Any;

                    block_data
                        .downcast_ref::<LocalBlockData<S>>()
                        .unwrap()
                        .clone()
                })
                .collect()
        })
    }

    /// Initiate a transfer between two pools.
    async fn begin_transfer<Source, Target>(
        &self,
        source_pool_list: &Option<LocalBlockDataList<Source>>,
        target_pool_list: &Option<LocalBlockDataList<Target>>,
        request: BlockTransferRequest,
    ) -> Result<tokio::sync::oneshot::Receiver<()>>
    where
        Source: Storage + NixlDescriptor,
        Target: Storage + NixlDescriptor,
        // Check that the source block is readable, local, and writable to the target block.
        LocalBlockData<Source>:
            ReadableBlock<StorageType = Source> + Local + WriteToStrategy<LocalBlockData<Target>>,
        // Check that the target block is writable.
        LocalBlockData<Target>: WritableBlock<StorageType = Target>,
        LocalBlockData<Source>: BlockDataProvider<Locality = locality::Local>,
        LocalBlockData<Target>: BlockDataProviderMut<Locality = locality::Local>,
    {
        let Some(source_pool_list) = source_pool_list else {
            return Err(anyhow::anyhow!("Source pool manager not initialized"));
        };
        let Some(target_pool_list) = target_pool_list else {
            return Err(anyhow::anyhow!("Target pool manager not initialized"));
        };

        // Extract the `from` and `to` indices from the request.
        let source_idxs = request.blocks().iter().map(|(from, _)| *from);
        let target_idxs = request.blocks().iter().map(|(_, to)| *to);

        // Get the blocks corresponding to the indices.
        let sources: Vec<LocalBlockData<Source>> = source_idxs
            .map(|idx| source_pool_list[idx].clone())
            .collect();
        let mut targets: Vec<LocalBlockData<Target>> = target_idxs
            .map(|idx| target_pool_list[idx].clone())
            .collect();

        // Perform the transfer, and return the notifying channel.
        let channel = match sources.write_to(&mut targets, true, self.context.clone()) {
            Ok(Some(channel)) => Ok(channel),
            Err(e) => {
                tracing::error!("Failed to write to blocks: {:?}", e);
                Err(e.into())
            }
            Ok(None) => {
                panic!("Failed to write blocks. No channel returned. This should never happen.")
            }
        };

        channel
    }
}

#[async_trait]
impl Handler for BlockTransferHandler {
    async fn handle(&self, mut message: MessageHandle) -> Result<()> {
        if message.data.len() != 1 {
            return Err(anyhow::anyhow!(
                "Block transfer request must have exactly one data element"
            ));
        }

        let request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

        if request.blocks().len() == 0 {
            let hashes = self.device.as_ref().unwrap().iter().map(hash_block_contents).collect::<Result<Vec<_>>>()?;
            tracing::debug!("Block Hashes: {:?}", hashes);
            message.ack().await?;
            return Ok(());
        }

        tracing::debug!(
            "Performing transfer of {} blocks from {:?} to {:?}",
            request.blocks().len(),
            request.from_pool(),
            request.to_pool()
        );

        let notify = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => {
                let device = self.device.as_ref().unwrap();
                let block_hash = hash_block_contents(&device[request.blocks()[0].0])?;
                tracing::debug!("Offloading block {} to {} with hash: {:?}", request.blocks()[0].0, request.blocks()[0].1, block_hash);
                
                let target_block = self.host.as_ref().unwrap()[request.blocks()[0].1].clone();

                let res = self.begin_transfer(&self.device, &self.host, request).await;

                let source_block_hash = hash_block_contents(&target_block)?;
                assert_eq!(source_block_hash, block_hash);

                res
            },
            (Host, Device) => {
                tracing::debug!("Onboarding blocks {:?}", request.blocks());
                self.begin_transfer(&self.host, &self.device, request).await
            },
            (Host, Disk) => {
                let host = self.host.as_ref().unwrap();

                let block_hash = hash_block_contents(&host[request.blocks()[0].0])?;

                let target_block = self.disk.as_ref().unwrap()[request.blocks()[0].1].clone();

                let initial_target_hash = hash_block_contents(&target_block);

                let res = self.begin_transfer(&self.host, &self.disk, request).await;

                res.unwrap().await?;
                message.ack().await?;

                let source_block_hash = hash_block_contents(&target_block)?;
                assert_eq!(source_block_hash, block_hash, "Host -> Disk content hash mismatch. Initial hash: {:?}", initial_target_hash);

                return Ok(());
            },
            (Disk, Device) => {
                tracing::debug!("Onboarding blocks {:?}", request.blocks());
                let disk = self.disk.as_ref().unwrap();

                let source_block = disk[request.blocks()[0].0].clone();

                let initial_source_hash = hash_block_contents(&source_block)?;

                let target_block = self.device.as_ref().unwrap()[request.blocks()[0].1].clone();
                let initial_target_hash = hash_block_contents(&target_block);

                let res = self.begin_transfer(&self.disk, &self.device, request).await;

                res.unwrap().await?;

                let final_source_hash = hash_block_contents(&source_block)?;
                let final_target_hash = hash_block_contents(&target_block)?;
                assert_eq!(initial_source_hash, final_target_hash, "Disk -> Device content hash mismatch. intial source hash: {:?}, final source hash: {:?}, initial target hash: {:?}, final target hash: {:?}", initial_source_hash, final_source_hash, initial_target_hash, final_target_hash);

                message.ack().await?;

                return Ok(());
            },
            _ => {
                return Err(anyhow::anyhow!("Invalid transfer type."));
            }
        }?;

        notify.await?;
        message.ack().await?;

        Ok(())
    }
}
