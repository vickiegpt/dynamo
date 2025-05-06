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

mod cuda;
mod memcpy;

use super::nixl::{IsMutable, NixlBlockDataImmutable, NixlBlockDataMutable, RemoteBlock};
use super::*;

use crate::block_manager::{
    storage::{
        nixl::{NixlEnabledStorage, NixlStorage},
        DeviceStorage, PinnedStorage, SystemAccessible, SystemStorage,
    },
    CacheLevel,
};

use cudarc::driver::CudaStream;

use std::ops::Range;

pub use crate::block_manager::storage::{CudaAccessible, Local, Remote};
pub use async_trait::async_trait;

/// A block that can be the target of a write
pub trait Writable {}

/// A block that can be the source of a read
pub trait Readable {}

pub trait Mutable: Readable + Writable {}

pub trait Immutable: Readable {}

#[derive(Debug)]
enum BlockTarget {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyStrategy {
    Memcpy,
    CudaAsyncH2D,
    CudaAsyncD2H,
    CudaAsyncD2D,
    CudaBlockingH2D,
    CudaBlockingD2H,
    Nixl,
    Invalid,
}

pub trait CopyToStrategy<Target> {
    fn strategy() -> CopyStrategy {
        CopyStrategy::Invalid
    }
}

pub trait CopyFromStrategy<Source> {
    fn copy_from_strategy() -> CopyStrategy {
        CopyStrategy::Invalid
    }
}

impl CopyToStrategy<SystemStorage> for SystemStorage {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::Memcpy
    }
}

impl CopyToStrategy<PinnedStorage> for SystemStorage {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::Memcpy
    }
}

impl CopyToStrategy<DeviceStorage> for SystemStorage {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::CudaBlockingH2D
    }
}

impl CopyToStrategy<SystemStorage> for PinnedStorage {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::Memcpy
    }
}

impl CopyToStrategy<PinnedStorage> for PinnedStorage {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::Memcpy
    }
}

impl CopyToStrategy<DeviceStorage> for PinnedStorage {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::CudaAsyncH2D
    }
}

impl CopyToStrategy<SystemStorage> for DeviceStorage {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::CudaBlockingD2H
    }
}

impl CopyToStrategy<PinnedStorage> for DeviceStorage {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::CudaAsyncD2H
    }
}

impl CopyToStrategy<DeviceStorage> for DeviceStorage {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::CudaAsyncD2D
    }
}

impl<S: Storage + Local> CopyToStrategy<NixlStorage> for S {
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        CopyStrategy::Nixl
    }
}

impl<S: Storage + Local, T: Storage + Local> CopyFromStrategy<S> for T
where
    T: CopyToStrategy<S>,
{
    #[inline(always)]
    fn copy_from_strategy() -> CopyStrategy {
        T::strategy()
    }
}

impl<RB: ReadableBlock, WB: WritableBlock> CopyToStrategy<WB> for RB
where
    <RB as ReadableBlock>::StorageType: Local + CopyToStrategy<<WB as WritableBlock>::StorageType>,
{
    #[inline(always)]
    fn strategy() -> CopyStrategy {
        <<RB as ReadableBlock>::StorageType as CopyToStrategy::<<WB as WritableBlock>::StorageType>>::strategy()
    }
}

pub trait CopyTo<Target> {
    fn copy_to(&self, dst: &mut Target) -> Result<(), TransferError>;
}

impl<RB: ReadableBlock, WB: WritableBlock> CopyTo<WB> for RB
where
    RB: CopyToStrategy<WB> + Local,
{
    fn copy_to(&self, dst: &mut WB) -> Result<(), TransferError> {
        let ctx = self.transfer_context();
        match Self::strategy() {
            CopyStrategy::Memcpy => memcpy::memcpy_block(self, dst),
            CopyStrategy::CudaAsyncH2D
            | CopyStrategy::CudaAsyncD2H
            | CopyStrategy::CudaAsyncD2D => {
                cuda::cuda_memcpy_block(self, dst, ctx.stream().as_ref(), RB::strategy())
            }
            _ => Err(TransferError::IncompatibleTypes(format!(
                "Unsupported copy strategy: {:?}",
                RB::strategy()
            ))),
        }
        // dispatch_copy_to(self, dst, self.transfer_context())
    }
}

#[derive(Default)]
pub struct GetXferRequestBuilder<
    'xfer,
    Source: BlockDataProvider,
    Target: BlockDataProviderMut + Local,
> {
    src: Option<&'xfer [Source]>,
    dst: Option<&'xfer [Target]>,
}

// impl<'xfer, Source: BlockDataProvider, Target: BlockDataProviderMut + Local>
//     GetXferRequestBuilder<'xfer, Source, Target>
// {
//     fn new(state: Arc<BlockTransferEngineState>) -> Self {
//         Self {
//             src: None,
//             dst: None,
//         }
//     }

//     pub fn from(&mut self, local_or_remote_blocks: &'xfer [Target]) -> &mut Self {
//         self.dst = Some(local_or_remote_blocks);
//         self
//     }

//     pub fn to(&mut self, local_mutable_blocks: &'xfer [Source]) -> &mut Self {
//         self.src = Some(local_mutable_blocks);
//         self
//     }
// }

pub struct PutXferRequestBuilder<
    'xfer,
    Source: BlockDataProvider + Local,
    Target: BlockDataProviderMut,
> {
    src: Option<&'xfer [Source]>,
    dst: Option<&'xfer [Target]>,
}

// impl<'xfer, Source: BlockDataProvider + Local, Target: BlockDataProviderMut>
//     PutXferRequestBuilder<'xfer, Source, Target>
// {
//     fn new(state: Arc<BlockTransferEngineState>) -> Self {
//         Self {
//             src: None,
//             dst: None,
//         }
//     }
//     pub fn from(&mut self, local_blocks: &'xfer [Source]) -> &mut Self {
//         self.src = Some(local_blocks);
//         self
//     }

//     pub fn to(&mut self, local_or_remote: &'xfer [Target]) -> &mut Self {
//         self.dst = Some(local_or_remote);
//         self
//     }
// }

// #[async_trait]
// impl<'xfer, Target: BlockDataProviderMut + Local>
//     AsyncBlockTransferEngine<RemoteBlock<IsImmutable>, Target>
//     for GetXferRequestBuilder<'xfer, RemoteBlock<IsImmutable>, Target>
// where
//     Target: BlockDataProviderMut + Local + Send + Sync,
// {
//     async fn execute(self) -> Result<()> {
//         unimplemented!()
//     }
// }

// #[async_trait]
// impl<'xfer, Source, Target> AsyncBlockTransferEngine<Source, Target>
//     for GetXferRequestBuilder<'xfer, Source, Target>
// where
//     Source: BlockDataProvider + Local + Send + Sync,
//     Target: BlockDataProviderMut + Local + Send + Sync,
// {
//     async fn execute(self) -> Result<()> {
//         unimplemented!()
//     }
// }

// pub trait BlockCopyTo<Target:BlockDataProviderMut + Local>: BlockDataProvider + Local {
//     fn copy_blocks

#[async_trait]
pub trait AsyncBlockTransferEngine<Source: BlockDataProvider, Target: BlockDataProviderMut + Local>
{
    async fn execute(self) -> anyhow::Result<()>;
}

pub trait BlockTransferEngineV1<Source: BlockDataProvider, Target: BlockDataProviderMut> {
    fn prepare(&mut self) -> Result<(), TransferError> {
        Ok(())
    }
    fn execute(self) -> Result<(), TransferError>;
}

// memcpy transfer engine
// - System -> System
// - Pinned -> Pinned

// cuda memcpy transfer engine
// - Pinned -> Device
// - Device -> Pinned
// - Device -> Device

// nixl memcpy transfer engine
// - NixlEnabledStorage -> Nixl
// - Nixl -> NixlEnabledStorage
// where System, Pinned, Device are NixlEnabledStorage

// Placeholder for the actual transfer plan
#[derive(Debug)]
pub struct TransferRequestPut<
    'a,
    Source: BlockDataProvider + Local,
    Destination: BlockDataProviderMut,
> {
    sources: &'a [Source],
    destinations: &'a mut [Destination],
}

// --- NIXL PUT Transfer Implementation ---

impl<Source> BlockTransferEngineV1<Source, RemoteBlock<IsMutable>>
    for TransferRequestPut<'_, Source, RemoteBlock<IsMutable>>
where
    Source: BlockDataProvider + Local, // + NixlBlockDataMutable<Source::StorageType>,
    Source::StorageType: NixlEnabledStorage,
{
    fn execute(self) -> Result<(), TransferError> {
        self.validate_counts()?;
        tracing::info!("Executing NIXL PUT transfer request");

        // TODO: Get NixlAgent handle

        for (src_block, dst_block) in self.sources.iter().zip(self.destinations.iter_mut()) {
            let src_data = src_block.block_data(private::PrivateToken);
            let src_nixl_desc = src_data.as_block_descriptor()?;

            let dst_data = dst_block.block_data_mut(private::PrivateToken);
            let dst_nixl_desc = dst_data.as_block_descriptor_mut()?;

            // TODO: Perform NIXL PUT operation
            // tracing::trace!(src = ?(src_data.worker_id, src_data.block_set_idx, src_data.block_idx), dst = ?(dst_data.worker_id, dst_data.block_set_idx, dst_data.block_idx), "NIXL PUT block");
            tracing::trace!(src_desc = ?src_nixl_desc, dst_desc = ?dst_nixl_desc, "NIXL PUT block");
        }
        Ok(())
    }
}

impl<'a, Source, Destination> TransferRequestPut<'a, Source, Destination>
where
    Source: BlockDataProvider + Local,
    Destination: BlockDataProviderMut,
{
    pub fn new(
        sources: &'a [Source],
        destinations: &'a mut [Destination],
    ) -> Result<Self, TransferError> {
        let transfer_request = Self {
            sources,
            destinations,
        };
        transfer_request.validate_counts()?;
        Ok(transfer_request)
    }

    /// Validate blocks
    ///
    /// For a put, we can have duplicate blocks on the source side, but all destinations must be unique
    /// For all transfers, the source and destination block sets must be disjoint.
    pub fn validate_blocks(&self) -> Result<(), TransferError> {
        let mut src_set = std::collections::HashSet::new();
        let mut dst_set = std::collections::HashSet::new();

        for (src_block, dst_block) in self.sources.iter().zip(self.destinations.iter()) {
            let src_data = src_block.block_data(private::PrivateToken);
            let dst_data = dst_block.block_data(private::PrivateToken);

            src_set.insert((
                src_data.block_set_idx,
                src_data.block_idx,
                src_data.worker_id,
            ));
            dst_set.insert((
                dst_data.block_set_idx,
                dst_data.block_idx,
                dst_data.worker_id,
            ));
        }

        if dst_set.len() != self.destinations.len() {
            return Err(TransferError::BuilderError(
                "Duplicate destination blocks".to_string(),
            ));
        }

        // the intersection of src_set and dst_set must be empty
        if !src_set.is_disjoint(&dst_set) {
            return Err(TransferError::BuilderError(
                "Duplicate one or more duplicate entries in source and destination list"
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Common validation for all PUT requests.
    fn validate_counts(&self) -> Result<(), TransferError> {
        if self.sources.len() != self.destinations.len() {
            Err(TransferError::CountMismatch(
                self.sources.len(),
                self.destinations.len(),
            ))
        } else if self.sources.is_empty() {
            Err(TransferError::BuilderError(
                "Sources cannot be empty".to_string(),
            ))
        } else if self.destinations.is_empty() {
            Err(TransferError::BuilderError(
                "Destinations cannot be empty".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

// // --- Local Transfer Implementations ---

// // Local Pinned -> Pinned
// impl<'a, MSource: BlockMetadata, MDest: BlockMetadata>
//     TransferRequestPut<
//         'a,
//         ImmutableBlock<PinnedStorage, MSource>,
//         MutableBlock<PinnedStorage, MDest>,
//     >
// {
//     pub fn execute(mut self) -> Result<(), TransferError> {
//         self.validate_counts()?;
//         tracing::info!("Executing local transfer: Pinned -> Pinned");
//         for (src_block, dst_block) in self.sources.iter().zip(self.destinations.iter_mut()) {
//             let src_data = src_block.block_data(private::PrivateToken);
//             let dst_data = dst_block.block_data_mut(private::PrivateToken);
//             // TODO: Implement layer-wise or block-wise CUDA memcpy H2H or std::ptr::copy
//             tracing::trace!(src = ?(src_data.worker_id, src_data.block_set_idx, src_data.block_idx), dst = ?(dst_data.worker_id, dst_data.block_set_idx, dst_data.block_idx), "Copying block");
//         }
//         Ok(())
//     }
// }

// // Local Pinned -> Device
// impl<'a, MSource: BlockMetadata, MDest: BlockMetadata>
//     TransferRequestPut<
//         'a,
//         ImmutableBlock<PinnedStorage, MSource>,
//         MutableBlock<DeviceStorage, MDest>,
//     >
// {
//     pub fn execute(mut self) -> Result<(), TransferError> {
//         self.validate_counts()?;
//         tracing::info!("Executing local transfer: Pinned -> Device");
//         for (src_block, dst_block) in self.sources.iter().zip(self.destinations.iter_mut()) {
//             let src_data = src_block.block_data(private::PrivateToken);
//             let dst_data = dst_block.block_data_mut(private::PrivateToken);
//             // TODO: Implement layer-wise or block-wise CUDA memcpy H2D
//             tracing::trace!(src = ?(src_data.worker_id, src_data.block_set_idx, src_data.block_idx), dst = ?(dst_data.worker_id, dst_data.block_set_idx, dst_data.block_idx), "Copying block");
//         }
//         Ok(())
//     }
// }

// // Local Device -> Pinned
// impl<'a, MSource: BlockMetadata, MDest: BlockMetadata>
//     TransferRequestPut<
//         'a,
//         ImmutableBlock<DeviceStorage, MSource>,
//         MutableBlock<PinnedStorage, MDest>,
//     >
// {
//     pub fn execute(mut self) -> Result<(), TransferError> {
//         self.validate_counts()?;
//         tracing::info!("Executing local transfer: Device -> Pinned");
//         for (src_block, dst_block) in self.sources.iter().zip(self.destinations.iter_mut()) {
//             let src_data = src_block.block_data(private::PrivateToken);
//             let dst_data = dst_block.block_data_mut(private::PrivateToken);
//             // TODO: Implement layer-wise or block-wise CUDA memcpy D2H
//             tracing::trace!(src = ?(src_data.worker_id, src_data.block_set_idx, src_data.block_idx), dst = ?(dst_data.worker_id, dst_data.block_set_idx, dst_data.block_idx), "Copying block");
//         }
//         Ok(())
//     }
// }

// // Local Device -> Device
// impl<'a, MSource: BlockMetadata, MDest: BlockMetadata>
//     TransferRequestPut<
//         'a,
//         ImmutableBlock<DeviceStorage, MSource>,
//         MutableBlock<DeviceStorage, MDest>,
//     >
// {
//     pub fn execute(mut self) -> Result<(), TransferError> {
//         self.validate_counts()?;
//         tracing::info!("Executing local transfer: Device -> Device");
//         for (src_block, dst_block) in self.sources.iter().zip(self.destinations.iter_mut()) {
//             let src_data = src_block.block_data(private::PrivateToken);
//             let dst_data = dst_block.block_data_mut(private::PrivateToken);
//             // TODO: Implement layer-wise or block-wise CUDA memcpy D2D
//             tracing::trace!(src = ?(src_data.worker_id, src_data.block_set_idx, src_data.block_idx), dst = ?(dst_data.worker_id, dst_data.block_set_idx, dst_data.block_idx), "Copying block");
//         }
//         Ok(())
//     }
// }

// pub fn dispatch_copy_to<RB, WB>(
//     src: &RB,
//     dst: &mut WB,
//     ctx: &TransferContext,
// ) -> Result<(), TransferError>
// where
//     RB: ReadableBlock,
//     WB: WritableBlock,
//     // Ensure the necessary capability traits are implemented for the storage types
//     // Note: These bounds aren't strictly *required* for the TypeId check,
//     // but help ensure the backend calls will compile if a match occurs.
//     // RB::Storage: SystemAccessible + CudaAccessible, // Might be too restrictive, apply within match arms
//     // WB::Storage: SystemAccessible + CudaAccessible,
// {
//     let src_type = src.storage_type_id();
//     let dst_type = dst.storage_type_id();

//     match (src_type, dst_type) {
//         // === Memcpy Cases ===
//         (s, d)
//             if (s == TypeId::of::<SystemStorage>() && d == TypeId::of::<SystemStorage>())
//                 || (s == TypeId::of::<PinnedStorage>() && d == TypeId::of::<SystemStorage>())
//                 || (s == TypeId::of::<SystemStorage>() && d == TypeId::of::<PinnedStorage>())
//                 || (s == TypeId::of::<PinnedStorage>() && d == TypeId::of::<PinnedStorage>()) =>
//         {
//             memcpy::memcpy_block(src, dst)
//         }

//         // === CUDA Cases ===
//         (s, d)
//             if (s == TypeId::of::<PinnedStorage>() && d == TypeId::of::<DeviceStorage>())
//                 || (s == TypeId::of::<DeviceStorage>() && d == TypeId::of::<PinnedStorage>())
//                 || (s == TypeId::of::<DeviceStorage>() && d == TypeId::of::<DeviceStorage>()) =>
//         {
//             cuda::cuda_memcpy_block(src, dst, ctx.stream().as_ref())
//             // let stream = stream.ok_or_else(|| {
//             //     TransferError::BuilderError("CUDA stream required for this transfer".into())
//             // })?;
//             // if is_cuda_compatible::<RB, WB>() {
//             //     tracing::debug!("Dispatching copy using CUDA");
//             //     cuda::cuda_memcpy_block(src_provider, dst_provider, stream) // Assumes cuda_memcpy_block is generic
//             // } else {
//             //     Err(TransferError::IncompatibleTypes(
//             //         "CUDA copy requires CudaAccessible storage".into(),
//             //     ))
//             // }
//         }

//         // === NIXL Cases ===
//         (s, d)
//             if d == TypeId::of::<NixlStorage>()
//                 && (s == TypeId::of::<SystemStorage>()
//                     || s == TypeId::of::<PinnedStorage>()
//                     || s == TypeId::of::<DeviceStorage>()) =>
//         {
//             unimplemented!()
//             // tracing::debug!("Dispatching copy using NIXL PUT");
//             // // TODO: Implement NIXL PUT logic
//             // // You might need a specific NIXL transfer function here.
//             // // Example: nixl::nixl_put_block(src_provider, dst_provider)
//             // Err(TransferError::ExecutionError(
//             //     "NIXL PUT not yet implemented".into(),
//             // ))
//         }

//         // TODO: Add NIXL GET cases (Nixl -> System/Pinned/Device)

//         // === Error Case ===
//         _ => Err(TransferError::IncompatibleTypes(format!(
//             "Unsupported storage combination for copy: {:?} -> {:?}",
//             std::any::type_name::<<RB as ReadableBlock>::StorageType>(), // Requires nightly or use debug print
//             std::any::type_name::<<WB as WritableBlock>::StorageType>()
//         ))),
//     }
// }

#[cfg(test)]
mod tests {

    use super::*;
    use crate::block_manager::storage::{
        DeviceAllocator, PinnedAllocator, StorageAllocator, SystemAllocator,
    };
    #[test]
    fn copy_to_strategy() {
        // System to ...
        assert_eq!(
            <SystemStorage as CopyToStrategy<SystemStorage>>::strategy(),
            CopyStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as CopyToStrategy<PinnedStorage>>::strategy(),
            CopyStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as CopyToStrategy<DeviceStorage>>::strategy(),
            CopyStrategy::CudaBlockingH2D
        );

        assert_eq!(
            <SystemStorage as CopyToStrategy<NixlStorage>>::strategy(),
            CopyStrategy::Nixl
        );

        // Pinned to ...
        assert_eq!(
            <PinnedStorage as CopyToStrategy<SystemStorage>>::strategy(),
            CopyStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as CopyToStrategy<PinnedStorage>>::strategy(),
            CopyStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as CopyToStrategy<DeviceStorage>>::strategy(),
            CopyStrategy::CudaAsyncH2D
        );
        assert_eq!(
            <PinnedStorage as CopyToStrategy<NixlStorage>>::strategy(),
            CopyStrategy::Nixl
        );

        // Device to ...
        assert_eq!(
            <DeviceStorage as CopyToStrategy<SystemStorage>>::strategy(),
            CopyStrategy::CudaBlockingD2H
        );
        assert_eq!(
            <DeviceStorage as CopyToStrategy<PinnedStorage>>::strategy(),
            CopyStrategy::CudaAsyncD2H
        );
        assert_eq!(
            <DeviceStorage as CopyToStrategy<DeviceStorage>>::strategy(),
            CopyStrategy::CudaAsyncD2D
        );
        assert_eq!(
            <DeviceStorage as CopyToStrategy<NixlStorage>>::strategy(),
            CopyStrategy::Nixl
        );

        // Nixl to ... should fail to compile
        // assert_eq!(
        //     <NixlStorage as CopyToStrategy<SystemStorage>>::strategy(),
        //     CopyStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as CopyToStrategy<PinnedStorage>>::strategy(),
        //     CopyStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as CopyToStrategy<DeviceStorage>>::strategy(),
        //     CopyStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as CopyToStrategy<NixlStorage>>::strategy(),
        //     CopyStrategy::Invalid
        // );
    }
}
