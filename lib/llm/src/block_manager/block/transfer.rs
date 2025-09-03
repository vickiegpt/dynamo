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

mod context;
mod cuda;
mod memcpy;
mod nixl;
mod strategy;

use super::*;

use crate::block_manager::storage::{
    nixl::{NixlRegisterableStorage, NixlStorage},
    DeviceStorage, DiskStorage, PinnedStorage, SystemStorage,
};

use nixl_sys::NixlDescriptor;
use nixl_sys::XferOp::{Read, Write};
use std::ops::Range;
use tokio::sync::oneshot;
use std::time::Duration;

pub use crate::block_manager::storage::{CudaAccessible, Local, Remote};
pub use async_trait::async_trait;
pub use context::TransferContext;

/// A block that can be the target of a write
pub trait Writable {}

/// A block that can be the source of a read
pub trait Readable {}

pub trait Mutable: Readable + Writable {}

pub trait Immutable: Readable {}

#[derive(Debug)]
pub enum BlockTarget {
    Source,
    Destination,
}

#[derive(Debug, thiserror::Error)]
pub enum TransferError {
    #[error("Builder configuration error: {0}")]
    BuilderError(String),
    #[error("Transfer execution failed: {0}")]
    ExecutionError(String),
    #[error("Incompatible block types provided: {0}")]
    IncompatibleTypes(String),
    #[error("Mismatched source/destination counts: {0} sources, {1} destinations")]
    CountMismatch(usize, usize),
    #[error("Block operation failed: {0}")]
    BlockError(#[from] BlockError),
    // TODO: Add NIXL specific errors
    #[error("No blocks provided")]
    NoBlocksProvided,

    #[error("Mismatched {0:?} block set index: {1} != {2}")]
    MismatchedBlockSetIndex(BlockTarget, usize, usize),

    #[error("Mismatched {0:?} worker ID: {1} != {2}")]
    MismatchedWorkerID(BlockTarget, usize, usize),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixlTransfer {
    Read,
    Write,
}

impl NixlTransfer {
    pub fn as_xfer_op(&self) -> nixl_sys::XferOp {
        match self {
            NixlTransfer::Read => Read,
            NixlTransfer::Write => Write,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    Memcpy,
    CudaAsyncH2D,
    CudaAsyncD2H,
    CudaAsyncD2D,
    CudaBlockingH2D,
    CudaBlockingD2H,
    Nixl(NixlTransfer),
    Invalid,
}

/// Trait for determining the transfer strategy for writing from a local
/// source to a target destination which could be local or remote
pub trait WriteToStrategy<Target> {
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Invalid
    }
}

/// Trait for determining the transfer strategy for reading from a
/// `Source` which could be local or remote into `Self` which must
/// be both local and writable.
pub trait ReadFromStrategy<Source> {
    fn read_from_strategy() -> TransferStrategy {
        TransferStrategy::Invalid
    }
}

impl<RB: ReadableBlock, WB: WritableBlock> WriteToStrategy<WB> for RB
where
    <RB as StorageTypeProvider>::StorageType:
        Local + WriteToStrategy<<WB as StorageTypeProvider>::StorageType>,
{
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        <<RB as StorageTypeProvider>::StorageType as WriteToStrategy<
            <WB as StorageTypeProvider>::StorageType,
        >>::write_to_strategy()
    }
}

impl<WB: WritableBlock, RB: ReadableBlock> ReadFromStrategy<RB> for WB
where
    <RB as StorageTypeProvider>::StorageType: Remote,
    <WB as StorageTypeProvider>::StorageType: NixlRegisterableStorage,
{
    #[inline(always)]
    fn read_from_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Read)
    }
}

pub fn handle_local_transfer<RB, WB>(
    sources: &[RB],
    targets: &mut [WB],
    ctx: Arc<TransferContext>,
) -> Result<oneshot::Receiver<()>, TransferError>
where
    RB: ReadableBlock + WriteToStrategy<WB> + Local,
    WB: WritableBlock,
    <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
{

    let (tx, rx) = oneshot::channel();

    tracing::debug!("=== TRANSFER FUNCTION START ===");
    tracing::debug!("Transfer: sources.len() = {}, targets.len() = {}", sources.len(), targets.len());
    tracing::debug!("Transfer: RB::write_to_strategy() = {:?}", RB::write_to_strategy());

    // 🔍 MANUAL MEMCHECK: Validate inputs
    if sources.is_empty() {
        tracing::error!("🚨 EMPTY SOURCES: No source blocks provided");
        return Err(TransferError::NoBlocksProvided);
    }

    if targets.is_empty() {
        tracing::error!("🚨 EMPTY TARGETS: No target blocks provided");
        return Err(TransferError::NoBlocksProvided);
    }

    if sources.len() != targets.len() {
        tracing::error!("🚨 COUNT MISMATCH: {} sources vs {} targets", sources.len(), targets.len());
        return Err(TransferError::CountMismatch(sources.len(), targets.len()));
    }

    match RB::write_to_strategy() {
        TransferStrategy::Memcpy => {
            tracing::debug!("Transfer: Using MEMCPY strategy");
            for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                // TODO: Unlike all other transfer strategies, this is fully blocking.
                // We probably want some sort of thread pool to handle these.
                memcpy::copy_block(src, dst)?;
            }

            tx.send(()).unwrap();
            Ok(rx)
        }
        TransferStrategy::CudaAsyncH2D
        | TransferStrategy::CudaAsyncD2H
        | TransferStrategy::CudaAsyncD2D => {
            tracing::debug!("Transfer: Using CUDA strategy: {:?}", RB::write_to_strategy());
            if RB::write_to_strategy() == TransferStrategy::CudaAsyncH2D {
                tracing::debug!("=== H2D TRANSFER START ===");
                tracing::debug!("H2D: sources.len() = {}, targets.len() = {}", sources.len(), targets.len());
                tracing::debug!("H2D: RB::write_to_strategy() = {:?}", RB::write_to_strategy());
                tracing::debug!("H2D: Strategy match confirmed, proceeding with H2D logic");

                // Get worker_id for cleanup
                let worker_id = if !sources.is_empty() { Some(sources[0].block_data().worker_id()) } else { None };
                tracing::debug!("H2D: worker_id = {:?}", worker_id);

                // Use simplified single kernel approach - let CUDA handle large transfers
                let selected_stream = ctx.next_h2d_stream();
                tracing::debug!("H2D: {} blocks using H2D stream pool", sources.len());

                let cleanup_result = cuda::copy_blocks_with_customized_kernel(sources, targets, selected_stream.as_ref(), RB::write_to_strategy())?;

                // 🔍 MANUAL MEMCHECK: Validate cleanup result
                if let Some((pointers, size)) = cleanup_result {
                    // Validate all pointers
                    for &ptr in &pointers {
                        if ptr == 0 {
                            tracing::error!("🚨 NULL CLEANUP POINTER: ptr=0x{:x}", ptr);
                            return Err(TransferError::ExecutionError("Null cleanup pointer detected".to_string()));
                        }
                    }

                    tracing::debug!("H2D: Cleanup needed: {} pointers, {} bytes", pointers.len(), size);
                    ctx.cuda_event_with_pinned_cleanup(tx, "H2D".to_string(), worker_id, pointers)?;
                } else {
                    tracing::debug!("H2D: No cleanup needed");
                    ctx.cuda_event(tx)?;
                }

                tracing::debug!("=== H2D TRANSFER COMPLETE ===");
                return Ok(rx);

            } else if RB::write_to_strategy() == TransferStrategy::CudaAsyncD2H {
                tracing::debug!("=== D2H TRANSFER START ===");
                tracing::debug!("D2H: sources.len() = {}, targets.len() = {}", sources.len(), targets.len());
                tracing::debug!("D2H: RB::write_to_strategy() = {:?}", RB::write_to_strategy());
                tracing::debug!("D2H: Strategy match confirmed, proceeding with D2H logic");

                // Get worker_id for cleanup
                let worker_id = if !sources.is_empty() { Some(sources[0].block_data().worker_id()) } else { None };
                tracing::debug!("D2H: worker_id = {:?}", worker_id);

                // Process in chunks of 72 blocks with dedicated streams
                const CHUNK_SIZE: usize = 128 * 72;
                let total_blocks = sources.len();
                let num_chunks = (total_blocks + CHUNK_SIZE - 1) / CHUNK_SIZE;

                tracing::debug!("D2H: Processing {} blocks in {} chunks of {}", total_blocks, num_chunks, CHUNK_SIZE);

                let mut all_cleanup_pointers = Vec::new();
                let mut chunk_events = Vec::new();

                for chunk_idx in 0..num_chunks {
                    let start_idx = chunk_idx * CHUNK_SIZE;
                    let end_idx = std::cmp::min(start_idx + CHUNK_SIZE, total_blocks);
                    let chunk_size = end_idx - start_idx;

                    let source_chunk = &sources[start_idx..end_idx];
                    let target_chunk = &mut targets[start_idx..end_idx];

                    // Get dedicated stream for this chunk
                    let chunk_stream = ctx.next_d2h_stream();
                    tracing::debug!("D2H: Chunk {}/{} ({} blocks) using dedicated D2H stream",
                                   chunk_idx + 1, num_chunks, chunk_size);

                    // Launch kernel for this chunk
                    let cleanup_result = cuda::copy_blocks_with_customized_kernel(
                        source_chunk,
                        target_chunk,
                        chunk_stream.as_ref(),
                        RB::write_to_strategy()
                    )?;

                    // Collect cleanup pointers from this chunk
                    if let Some((pointers, _size)) = cleanup_result {
                        all_cleanup_pointers.extend(pointers);
                    }

                    // Record event on this chunk's stream to track completion
                    let chunk_event = chunk_stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                        .map_err(|e| TransferError::ExecutionError(e.to_string()))?;
                    chunk_events.push(chunk_event);
                }

                // Wait for all chunk events to complete, then cleanup
                ctx.async_rt_handle().spawn(async move {
                    // Wait for all chunk events to complete
                    for (idx, chunk_event) in chunk_events.iter().enumerate() {
                        if let Err(e) = chunk_event.synchronize() {
                            tracing::error!("D2H: Chunk {} event synchronization failed: {}", idx, e);
                        }
                    }

                    // All chunks completed, now cleanup all pointers
                    if !all_cleanup_pointers.is_empty() {
                        tracing::debug!("D2H: Cleaning up {} total pointers from all chunks", all_cleanup_pointers.len());
                        if let Err(e) = cuda::cleanup_pinned_pointers(all_cleanup_pointers) {
                            tracing::error!("D2H: Failed to cleanup pointers: {}", e);
                        }
                    }

                    // Signal overall completion
                    tx.send(()).unwrap();
                });

                tracing::debug!("=== D2H TRANSFER COMPLETE ===");
                return Ok(rx);
            } else {
                // Fall back to individual copy for single H2D blocks
                for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                    cuda::copy_block(src, dst, ctx.stream().as_ref(), RB::write_to_strategy())?;
                }

                ctx.cuda_event(tx)?;
                return Ok(rx);
            }
        }
        TransferStrategy::Nixl(transfer_type) => {
            let transfer_fut = nixl::write_blocks_to(sources, targets, &ctx, transfer_type)?;

            ctx.async_rt_handle().spawn(async move {
                transfer_fut.await;
                tx.send(()).unwrap();
            });
            Ok(rx)
        }
        _ => Err(TransferError::IncompatibleTypes(format!(
            "Unsupported copy strategy: {:?}",
            RB::write_to_strategy()
        ))),
    }
}

pub trait WriteTo<Target> {
    fn write_to(
        &self,
        dst: &mut Vec<Target>,
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>;
}

impl<RB, WB, L: LocalityProvider> WriteTo<WB> for Vec<RB>
where
    RB: ReadableBlock + WriteToStrategy<WB> + Local,
    <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
    RB: BlockDataProvider<Locality = L>,
    WB: WritableBlock + BlockDataProviderMut<Locality = L>,
{
    fn write_to(
        &self,
        dst: &mut Vec<WB>,
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError> {
        L::handle_transfer(self, dst, ctx)
    }
}

// Add timeout for transfer completion to detect kernel failures
pub async fn handle_local_transfer_with_timeout<RB, WB>(
    sources: Vec<RB>,
    targets: &mut Vec<WB>,
    ctx: Arc<TransferContext>,
) -> Result<(), TransferError>
where
    RB: BlockDataProvider + Send + Sync,
    WB: BlockDataProviderMut + Send + Sync,
    Vec<RB>: WriteTo<WB>,
{
    let completion_receiver = sources.write_to(targets, ctx)?;

    // Add timeout to detect kernel failures
    let timeout_duration = Duration::from_secs(30); // 30 second timeout

    match tokio::time::timeout(timeout_duration, completion_receiver).await {
        Ok(Ok(())) => {
            tracing::debug!("Transfer completed successfully");
            Ok(())
        }
        Ok(Err(_)) => {
            let error = TransferError::ExecutionError("Transfer completion channel closed".into());
            tracing::debug!("Transfer failed: channel closed");
            Err(error)
        }
        Err(_) => {
            let error = TransferError::ExecutionError("Transfer timeout - kernel may have crashed".into());
            tracing::debug!("Transfer timed out after {:?} - possible kernel crash", timeout_duration);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_to_strategy() {
        // System to ...
        assert_eq!(
            <SystemStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingH2D
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Pinned to ...
        assert_eq!(
            <PinnedStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncH2D
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Device to ...
        assert_eq!(
            <DeviceStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2D
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Nixl to ... should fail to compile
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
    }
}






