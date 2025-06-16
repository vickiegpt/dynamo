use super::{
    transfer::{TransferStrategy, WriteToStrategy},
    *,
};
use crate::block_manager::layout::{
    nixl::SerializedNixlBlockLayout, BlockLayoutConfig, LayoutConfig, LayoutType,
};

pub trait BlockDataLocality: BlockLayoutConfig + Send + Sync {}

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

impl<S: Storage> Deref for LocalBlockData<S> {
    type Target = BlockData<S>;
    fn deref(&self) -> &Self::Target {
        &self.block_data
    }
}

impl<S: Storage> DerefMut for LocalBlockData<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.block_data
    }
}

impl<S: Storage> From<BlockData<S>> for LocalBlockData<S> {
    fn from(block_data: BlockData<S>) -> Self {
        Self { block_data }
    }
}

impl<S: Storage> BlockDataLocality for LocalBlockData<S> {}

impl<S: Storage> BlockLayoutConfig for LocalBlockData<S> {
    fn layout_type(&self) -> LayoutType {
        self.block_data.layout.layout_type()
    }

    fn num_blocks(&self) -> usize {
        self.block_data.layout.num_blocks()
    }

    fn num_layers(&self) -> usize {
        self.block_data.layout.num_layers()
    }

    fn outer_dim(&self) -> usize {
        self.block_data.layout.outer_dim()
    }

    fn page_size(&self) -> usize {
        self.block_data.layout.page_size()
    }

    fn inner_dim(&self) -> usize {
        self.block_data.layout.inner_dim()
    }
}

pub mod nixl {
    use super::*;
    use crate::block_manager::storage::{nixl::NixlStorage, StorageType};
    use std::collections::HashSet;

    #[derive(Debug)]
    pub struct RemoteBlockData {
        block_data: BlockData<NixlStorage>,
    }

    impl BlockDataLocality for RemoteBlockData {}

    impl BlockLayoutConfig for RemoteBlockData {
        fn layout_type(&self) -> LayoutType {
            self.block_data.layout.layout_type()
        }

        fn num_blocks(&self) -> usize {
            self.block_data.layout.num_blocks()
        }

        fn num_layers(&self) -> usize {
            self.block_data.layout.num_layers()
        }

        fn outer_dim(&self) -> usize {
            self.block_data.layout.outer_dim()
        }

        fn page_size(&self) -> usize {
            self.block_data.layout.page_size()
        }

        fn inner_dim(&self) -> usize {
            self.block_data.layout.inner_dim()
        }
    }

    #[derive(Debug)]
    pub struct ActiveMessageClient {}

    #[derive(Debug)]
    pub struct ReplicatedBlockDataParallel<L: Locality> {
        layouts: Vec<Arc<dyn BlockLayout<StorageType = NixlStorage>>>,
        am_client: Vec<ActiveMessageClient>,
        locality: std::marker::PhantomData<L>,

        // extracted from the first layout and validated for continuity
        layout_config: LayoutConfig,
        layout_type: LayoutType,
        storage_type: StorageType,
    }

    impl<L: Locality> ReplicatedBlockDataParallel<L> {
        pub fn new(
            layouts: Vec<SerializedNixlBlockLayout>,
            am_client: Vec<ActiveMessageClient>,
        ) -> Result<Self, BlockError> {
            // num of am_clients should be equal to the number of layouts
            // there must be at least one am_client

            assert!(layouts.len() > 0);
            assert!(am_client.len() > 0);

            if layouts.len() != am_client.len() {
                return Err(BlockError::MisconfiguredBlockDataParallelism(
                    "Number of layouts must be equal to the number of am_clients".to_string(),
                ));
            }

            // deserialize the layouts
            let layouts = layouts
                .into_iter()
                .map(|layout| layout.deserialize())
                .collect::<Result<Vec<_>, _>>()?;

            // extract and validate for continuity
            let storage_type = layouts[0].storage_type();
            let layout_config = layouts[0].config().clone();
            let layout_type = layouts[0].layout_type();

            L::is_compatible_storage(&storage_type)?;

            for layout in layouts.iter().skip(1) {
                if layout.storage_type() != storage_type {
                    return Err(BlockError::MisconfiguredBlockDataParallelism(
                        "All layouts must have the same storage type".to_string(),
                    ));
                }

                if layout.config() != &layout_config {
                    return Err(BlockError::MisconfiguredBlockDataParallelism(
                        "All layouts must have the same config".to_string(),
                    ));
                }

                if layout.layout_type() != layout_type {
                    return Err(BlockError::MisconfiguredBlockDataParallelism(
                        "All layouts must have the same layout type".to_string(),
                    ));
                }
            }

            Ok(Self {
                layouts,
                layout_config,
                layout_type,
                storage_type,
                am_client,
                locality: std::marker::PhantomData,
            })
        }
    }

    impl<L: Locality> BlockDataLocality for ReplicatedBlockDataParallel<L> {}

    impl<L: Locality> BlockLayoutConfig for ReplicatedBlockDataParallel<L> {
        fn layout_type(&self) -> LayoutType {
            self.layout_type
        }

        fn num_blocks(&self) -> usize {
            self.layout_config.num_blocks
        }

        fn num_layers(&self) -> usize {
            self.layout_config.num_layers
        }

        fn outer_dim(&self) -> usize {
            self.layout_config.outer_dim
        }

        fn page_size(&self) -> usize {
            self.layout_config.page_size
        }

        fn inner_dim(&self) -> usize {
            self.layout_config.inner_dim
        }
    }
}

pub mod v2 {

    use super::{BlockDataLocality, BlockMetadata, BlockState};

    pub struct Block<L: BlockDataLocality, M: BlockMetadata> {
        locality: L,
        metadata: M,
        state: BlockState,
    }
}
