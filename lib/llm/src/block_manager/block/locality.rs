use super::{
    transfer::{TransferStrategy, WriteToStrategy},
    *,
};

pub trait BlockDataLocality: Send + Sync + std::fmt::Debug {}

pub trait Locality: Send + Sync + std::fmt::Debug + PartialEq + Eq + std::hash::Hash {
    fn is_compatible_storage(storage_type: &StorageType) -> Result<(), BlockError>;
    fn storage_type(&self) -> &StorageType;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Host {
    storage_type: StorageType,
}

impl Host {
    pub fn new(storage_type: StorageType) -> Result<Self, BlockError> {
        Self::is_compatible_storage(&storage_type)?;
        Ok(Self { storage_type })
    }
}

impl Locality for Host {
    fn is_compatible_storage(storage_type: &StorageType) -> Result<(), BlockError> {
        match storage_type {
            StorageType::System => Ok(()),
            StorageType::Pinned => Ok(()),
            _ => Err(BlockError::IncompatibleStorageType(
                "Host is not compatible with this storage type; only system and pinned are allowed"
                    .to_string(),
            )),
        }
    }

    fn storage_type(&self) -> &StorageType {
        &self.storage_type
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Device {
    storage_type: StorageType,
}

impl Device {
    pub fn new(storage_type: StorageType) -> Result<Self, BlockError> {
        Self::is_compatible_storage(&storage_type)?;
        Ok(Self { storage_type })
    }
}

impl Locality for Device {
    fn storage_type(&self) -> &StorageType {
        &self.storage_type
    }
    fn is_compatible_storage(storage_type: &StorageType) -> Result<(), BlockError> {
        match storage_type {
            StorageType::Device(_device) => Ok(()),
            _ => Err(BlockError::IncompatibleStorageType(
                "Device is not compatible with this storage type; only device are allowed"
                    .to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Disk;

impl Disk {
    pub fn new(storage_type: StorageType) -> Result<Self, BlockError> {
        Self::is_compatible_storage(&storage_type)?;
        Ok(Self)
    }
}

impl Locality for Disk {
    fn storage_type(&self) -> &StorageType {
        &StorageType::Disk
    }

    fn is_compatible_storage(storage_type: &StorageType) -> Result<(), BlockError> {
        match storage_type {
            StorageType::Disk => Ok(()),
            _ => Err(BlockError::IncompatibleStorageType(
                "Disk is not compatible with this storage type; only disk are allowed".to_string(),
            )),
        }
    }
}

#[derive(Debug)]
pub struct LocalBlockData<S: Storage> {
    block_data: BlockData<S>,
}

impl<S: Storage> BlockDataLocality for LocalBlockData<S> {}

pub mod nixl {
    use super::*;
    use crate::block_manager::storage::{nixl::NixlStorage, StorageType};
    use std::collections::HashSet;

    #[derive(Debug)]
    pub struct RemoteBlockData {
        block_data: BlockData<NixlStorage>,
    }

    impl BlockDataLocality for RemoteBlockData {}

    #[derive(Debug)]
    pub struct ActiveMessageClient {}

    #[derive(Debug)]
    pub struct ReplicatedBlockDataParallel<L: Locality> {
        block_data: Vec<RemoteBlockData>,
        am_client: Vec<ActiveMessageClient>,
        storage: std::marker::PhantomData<L>,
    }

    impl<L: Locality> ReplicatedBlockDataParallel<L> {
        pub fn new(
            block_data: Vec<RemoteBlockData>,
            am_client: Vec<ActiveMessageClient>,
        ) -> Result<Self, BlockError> {
            let storage_type = block_data
                .iter()
                .map(|bd| bd.block_data.storage_type())
                .collect::<HashSet<StorageType>>();

            // determine that all the block data have the same storage type
            if storage_type.len() != 1 {
                return Err(BlockError::MisconfiguredBlockDataParallelism(
                    "All block data must have the same storage type".to_string(),
                ));
            }

            let remote_storage_type = storage_type.iter().next().unwrap();
            L::is_compatible_storage(remote_storage_type)?;

            Ok(Self {
                block_data,
                am_client,
                storage: std::marker::PhantomData,
            })
        }
    }

    impl<L: Locality> BlockDataLocality for ReplicatedBlockDataParallel<L> {}
}

pub mod v2 {

    use super::{BlockDataLocality, BlockMetadata, BlockState};

    pub struct Block<L: BlockDataLocality, M: BlockMetadata> {
        locality: L,
        metadata: M,
        state: BlockState,
    }
}
