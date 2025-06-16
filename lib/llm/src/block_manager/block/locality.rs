use super::{
    transfer::{TransferStrategy, WriteToStrategy},
    *,
};
use crate::block_manager::{
    layout::{nixl::SerializedNixlBlockLayout, BlockLayoutConfig, LayoutConfig, LayoutType},
    DeviceStorage, DiskStorage, PinnedStorage,
};

pub trait BlockDataLocality: BlockLayoutConfig + Send + Sync {
    fn storage_type(&self) -> StorageType;
}
pub trait BlockDataStorage: BlockDataLocality {
    type StorageType: Storage;
}

pub trait LocalityProvider: Send + Sync + 'static + std::fmt::Debug {
    type Disk: BlockDataStorage;
    type Host: BlockDataStorage;
    type Device: BlockDataStorage;

    type BlockData<S: Storage>: BlockDataStorage;
}

#[derive(Debug)]
pub struct Local;

impl LocalityProvider for Local {
    type Disk = LocalBlockData<DiskStorage>;
    type Host = LocalBlockData<PinnedStorage>;
    type Device = LocalBlockData<DeviceStorage>;

    type BlockData<S: Storage> = LocalBlockData<S>;
}

pub trait Parallelism: Send + Sync + 'static + std::fmt::Debug {
    type Output<S: Storage>: BlockDataStorage;
}

#[derive(Debug)]
pub struct Logical<P: Parallelism> {
    _parallelism: std::marker::PhantomData<P>,
}

impl<P: Parallelism> LocalityProvider for Logical<P> {
    type Disk = P::Output<DiskStorage>;
    type Host = P::Output<PinnedStorage>;
    type Device = P::Output<DeviceStorage>;

    type BlockData<S: Storage> = P::Output<S>;
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

impl<S: Storage> BlockDataStorage for LocalBlockData<S> {
    type StorageType = S;
}

impl<S: Storage> BlockDataLocality for LocalBlockData<S> {
    fn storage_type(&self) -> StorageType {
        self.block_data.storage_type()
    }
}

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

    #[derive(Debug)]
    pub struct RemoteBlockData {
        block_data: BlockData<NixlStorage>,
    }

    impl BlockDataLocality for RemoteBlockData {
        fn storage_type(&self) -> StorageType {
            self.block_data.storage_type().clone()
        }
    }

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
    pub struct WorkerReplicated;

    #[derive(Debug)]
    pub struct ReplicatedBlockDataParallel<S: Storage> {
        layouts: Vec<Arc<dyn BlockLayout<StorageType = NixlStorage>>>,
        am_client: Vec<ActiveMessageClient>,
        storage: std::marker::PhantomData<S>,

        // extracted from the first layout and validated for continuity
        layout_config: LayoutConfig,
        layout_type: LayoutType,
        storage_type: StorageType,
    }

    impl Parallelism for WorkerReplicated {
        type Output<S: Storage> = ReplicatedBlockDataParallel<S>;
    }

    impl<S: Storage> ReplicatedBlockDataParallel<S> {
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
            let storage_type = layouts[0].storage_type().clone();
            let layout_config = layouts[0].config().clone();
            let layout_type = layouts[0].layout_type();

            for layout in layouts.iter().skip(1) {
                if layout.storage_type() != &storage_type {
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
                storage_type: storage_type.clone(),
                am_client,
                storage: std::marker::PhantomData,
            })
        }
    }

    impl<S: Storage> BlockDataStorage for ReplicatedBlockDataParallel<S> {
        type StorageType = NixlStorage;
    }

    impl<S: Storage> BlockDataLocality for ReplicatedBlockDataParallel<S> {
        fn storage_type(&self) -> StorageType {
            self.storage_type.clone()
        }
    }

    impl<S: Storage> BlockLayoutConfig for ReplicatedBlockDataParallel<S> {
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
