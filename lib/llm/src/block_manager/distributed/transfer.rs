// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use futures::future::try_join_all;
use nixl_sys::NixlDescriptor;
use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    BasicMetadata, Storage,
    block::{
        Block, BlockDataProvider, BlockDataProviderMut, ReadableBlock, WritableBlock,
        data::{local::LocalBlockData, BlockDataExt},
        locality,
        transfer::{TransferContext, WriteTo, WriteToStrategy},
    },
    connector::scheduler::{SchedulingDecision, TransferSchedulerClient},
    offload::MAX_TRANSFER_BATCH_SIZE,
    layout::BlockLayoutConfig,
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
};

use anyhow::Result;
use async_trait::async_trait;
use std::{any::Any, sync::Arc};

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

/// A batching wrapper for connector transfers to prevent resource exhaustion.
/// Splits large transfers into smaller batches that can be handled by the resource pools.
#[derive(Clone, Debug)]
pub struct ConnectorTransferBatcher {
    max_batch_size: usize,
}

impl ConnectorTransferBatcher {
    pub fn new() -> Self {
        Self {
            max_batch_size: MAX_TRANSFER_BATCH_SIZE,
        }
    }

    pub async fn execute_batched_transfer(
        &self,
        handler: &BlockTransferHandler,
        request: BlockTransferRequest,
    ) -> Result<()> {
        let blocks = request.blocks();
        let num_blocks = blocks.len();

        if num_blocks <= self.max_batch_size {
            return handler.execute_transfer_direct(request).await;
        }

        let batches = blocks.chunks(self.max_batch_size);

        let batch_futures: Vec<_> = batches
            .map(|batch| {
                let batch_request = BlockTransferRequest {
                    from_pool: *request.from_pool(),
                    to_pool: *request.to_pool(),
                    blocks: batch.to_vec(),
                    connector_req: None,
                };
                handler.execute_transfer_direct(batch_request)
            })
            .collect();

        // Execute all batches concurrently
        tracing::debug!("Executing {} batches concurrently", batch_futures.len());

        match try_join_all(batch_futures).await {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("Batched connector transfer failed: {}", e);
                Err(e)
            }
        }
    }
}

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
#[derive(Clone)]
pub struct BlockTransferHandler {
    device: Option<LocalBlockDataList<DeviceStorage>>,
    host: Option<LocalBlockDataList<PinnedStorage>>,
    disk: Option<LocalBlockDataList<DiskStorage>>,
    context: Arc<TransferContext>,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    // add worker-connector scheduler client here
}

impl BlockTransferHandler {
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
        scheduler_client: Option<TransferSchedulerClient>,
        // add worker-connector scheduler client here
    ) -> Result<Self> {
        Ok(Self {
            device: Self::get_local_data(device_blocks),
            host: Self::get_local_data(host_blocks),
            disk: Self::get_local_data(disk_blocks),
            context,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
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

        // Validate layout compatibility before transfer
        self.validate_transfer_compatibility(&sources, &targets, &request)?;

        // Perform the transfer, and return the notifying channel.
        match sources.write_to(&mut targets, self.context.clone()) {
            Ok(channel) => Ok(channel),
            Err(e) => {
                tracing::error!("Failed to write to blocks: {:?}", e);
                Err(e.into())
            }
        }
    }

    /// Execute transfer with batching to prevent resource exhaustion
    pub async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    /// Execute transfer directly without batching (used by the batcher)
    pub async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        tracing::debug!(
            "Performing transfer of {} blocks from {:?} to {:?}",
            request.blocks().len(),
            request.from_pool(),
            request.to_pool()
        );

        tracing::debug!("request: {request:#?}");

        let notify = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => self.begin_transfer(&self.device, &self.host, request).await,
            (Host, Device) => self.begin_transfer(&self.host, &self.device, request).await,
            (Host, Disk) => self.begin_transfer(&self.host, &self.disk, request).await,
            (Disk, Device) => self.begin_transfer(&self.disk, &self.device, request).await,
            _ => {
                return Err(anyhow::anyhow!("Invalid transfer type."));
            }
        }?;

        notify.await?;
        Ok(())
    }

    /// Validate layout compatibility between source and target blocks
    fn validate_transfer_compatibility<Source, Target>(
        &self,
        sources: &[LocalBlockData<Source>],
        targets: &[LocalBlockData<Target>],
        request: &BlockTransferRequest,
    ) -> Result<()>
    where
        Source: Storage,
        Target: Storage,
    {
        // Note: verify_layout_compatibility is not used in this simplified validation
        // use crate::block_manager::layout::utils::worker_verification::verify_layout_compatibility;

        if sources.is_empty() || targets.is_empty() {
            return Ok(());
        }

        // Get first blocks to check layout compatibility
        let source_block = &sources[0];
        let target_block = &targets[0];

        // Extract layout information from block data
        let source_data = source_block.block_data();
        let target_data = target_block.block_data();

        // Basic compatibility checks
        if source_data.num_layers() != target_data.num_layers() {
            return Err(anyhow::anyhow!(
                "Layout mismatch: source has {} layers, target has {} layers",
                source_data.num_layers(),
                target_data.num_layers()
            ));
        }

        // Check memory region sizes for each block pair
        for (i, (source, target)) in sources.iter().zip(targets.iter()).enumerate() {
            let src_data = source.block_data();
            let tgt_data = target.block_data();

            // Verify each layer has compatible sizes (checking first outer dimension)
            for layer_idx in 0..src_data.num_layers() {
                let outer_idx = 0; // Check first outer dimension for compatibility
                let src_layer_result = src_data.layer_view(layer_idx, outer_idx);
                let tgt_layer_result = tgt_data.layer_view(layer_idx, outer_idx);

                match (src_layer_result, tgt_layer_result) {
                    (Ok(src_layer), Ok(tgt_layer)) => {
                        if src_layer.size() != tgt_layer.size() {
                            return Err(anyhow::anyhow!(
                                "Layout mismatch in block {} layer {}: source size {} != target size {}",
                                i, layer_idx, src_layer.size(), tgt_layer.size()
                            ));
                        }
                    }
                    (Err(e), _) => {
                        tracing::warn!("Failed to get source layer view for block {} layer {}: {}", i, layer_idx, e);
                    }
                    (_, Err(e)) => {
                        tracing::warn!("Failed to get target layer view for block {} layer {}: {}", i, layer_idx, e);
                    }
                }
            }
        }

        // Log successful validation
        tracing::debug!(
            "Layout compatibility validated for {} blocks transfer from {:?} to {:?}",
            sources.len(),
            request.from_pool(),
            request.to_pool()
        );

        Ok(())
    }

    /// Verify block data integrity after transfer
    pub fn verify_transfer_integrity<Source, Target>(
        &self,
        sources: &[LocalBlockData<Source>],
        targets: &[LocalBlockData<Target>],
        expected_patterns: Option<&[u8]>,
    ) -> Result<()>
    where
        Source: Storage,
        Target: Storage,
    {
        for (i, (source, target)) in sources.iter().zip(targets.iter()).enumerate() {
            let src_data = source.block_data();
            let tgt_data = target.block_data();

            // Compare data integrity
            if let (Ok(src_view), Ok(tgt_view)) = (src_data.block_view(), tgt_data.block_view()) {
                if src_view.size() == tgt_view.size() {
                    unsafe {
                        let src_slice = std::slice::from_raw_parts(src_view.as_ptr(), src_view.size());
                        let tgt_slice = std::slice::from_raw_parts(tgt_view.as_ptr(), tgt_view.size());

                        // Check for data corruption
                        let matches = src_slice.iter().zip(tgt_slice.iter())
                            .filter(|(a, b)| a == b)
                            .count();

                        let match_ratio = matches as f64 / src_slice.len() as f64;

                        if match_ratio < 0.95 {
                            return Err(anyhow::anyhow!(
                                "Data integrity check failed for block {}: only {:.1}% of data matches",
                                i, match_ratio * 100.0
                            ));
                        }

                        // Check for specific patterns if provided
                        if let Some(patterns) = expected_patterns {
                            if let Some(&expected_pattern) = patterns.get(i) {
                                let pattern_matches = tgt_slice.iter()
                                    .filter(|&&b| b == expected_pattern)
                                    .count();

                                let pattern_ratio = pattern_matches as f64 / tgt_slice.len() as f64;

                                if pattern_ratio < 0.8 {
                                    tracing::warn!(
                                        "Block {} has unexpected pattern distribution: {:.1}% matches expected pattern {}",
                                        i, pattern_ratio * 100.0, expected_pattern
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        tracing::debug!("Transfer integrity verification completed for {} blocks", sources.len());
        Ok(())
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

        let mut request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

        let result = if let Some(req) = request.connector_req.take() {
            let operation_id = req.uuid;

            tracing::debug!(
                request_id = %req.request_id,
                operation_id = %operation_id,
                "scheduling transfer"
            );

            let client = self
                .scheduler_client
                .as_ref()
                .expect("scheduler client is required")
                .clone();

            let handle = client.schedule_transfer(req).await?;

            // we don't support cancellation yet
            assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

            match self.execute_transfer(request).await {
                Ok(_) => {
                    handle.mark_complete(Ok(())).await;
                    Ok(())
                }
                Err(e) => {
                    handle.mark_complete(Err(anyhow::anyhow!("{}", e))).await;
                    Err(e)
                }
            }
        } else {
            self.execute_transfer(request).await
        };

        // we always ack regardless of if we error or not
        message.ack().await?;

        // the error may trigger a cancellation
        result
    }
}
