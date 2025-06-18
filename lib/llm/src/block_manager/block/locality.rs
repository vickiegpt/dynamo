use super::{
    // transfer::{TransferStrategy, WriteToStrategy},
    *,
};
use crate::block_manager::{
    layout::{nixl::SerializedNixlBlockLayout, BlockLayoutConfig, LayoutConfig, LayoutType},
    DeviceStorage, DiskStorage, PinnedStorage,
};
use nixl_sys::NixlDescriptor;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalityType {
    Local(StorageType),
    Remote,
    Logical(ParallelismType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelismType {
    WorkerSharded(usize),
}

pub trait BlockDataLocality: BlockLayoutConfig + Send + Sync {
    fn storage_type(&self) -> StorageType;
}
pub trait BlockDataStorage: BlockDataLocality {
    type StorageType: Storage;
    type BlockDataType<S: Storage>;

    fn block_data(&self) -> &Self::BlockDataType<Self::StorageType>;

    fn block_data_mut(&mut self) -> &mut Self::BlockDataType<Self::StorageType>;
}

// pub trait BlockDataProvider {
//     type StorageType: Storage;
//     type Locality: LocalityProvider;

//     fn block_data(&self) -> &Self::Locality::BlockData<Self::StorageType>;
// }

pub trait LocalityProvider: Send + Sync + 'static + std::fmt::Debug {
    type Disk: BlockDataStorage;
    type Host: BlockDataStorage;
    type Device: BlockDataStorage;

    type BlockData<S: Storage>: BlockDataStorage;

    /// Check if this locality is compatible with another for transfers
    /// Returns true if transfers between these localities are supported
    fn is_transfer_compatible_with<Other: LocalityProvider>() -> bool {
        // Default: only compatible with same locality type
        std::any::TypeId::of::<Self>() == std::any::TypeId::of::<Other>()
    }

    /// Get the transfer mechanism required for transfers to another locality
    fn transfer_mechanism_to<Other: LocalityProvider>() -> TransferMechanism {
        if Self::is_transfer_compatible_with::<Other>() {
            if std::any::TypeId::of::<Self>() == std::any::TypeId::of::<Local>() {
                TransferMechanism::DirectMemory
            } else {
                TransferMechanism::MockCounting // For MockLogical
            }
        } else {
            TransferMechanism::Unsupported
        }
    }
}

/// Transfer mechanism describes how transfers should be performed between locality types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMechanism {
    /// Direct memory access (Local-to-Local transfers using memcpy, CUDA, etc.)
    DirectMemory,
    /// Mock counting mechanism for testing (MockLogical-to-MockLogical)
    MockCounting,
    /// Remote RPC mechanism for cross-worker transfers (Logical localities)
    RemoteRpc,
    /// Unsupported transfer combination
    Unsupported,
}

/// Local locality provider for direct memory access
#[derive(Debug)]
pub struct Local;

impl LocalityProvider for Local {
    type Disk = Self::BlockData<DiskStorage>;
    type Host = Self::BlockData<PinnedStorage>;
    type Device = Self::BlockData<DeviceStorage>;

    type BlockData<S: Storage> = LocalBlockData<S>;

    fn is_transfer_compatible_with<Other: LocalityProvider>() -> bool {
        // Local is compatible with Local
        std::any::TypeId::of::<Other>() == std::any::TypeId::of::<Local>()
    }

    fn transfer_mechanism_to<Other: LocalityProvider>() -> TransferMechanism {
        if Self::is_transfer_compatible_with::<Other>() {
            TransferMechanism::DirectMemory
        } else {
            TransferMechanism::Unsupported
        }
    }
}

/// Mock logical locality for testing - computes transfer sizes without moving data
#[derive(Debug)]
pub struct MockLogical;

impl LocalityProvider for MockLogical {
    type Disk = Self::BlockData<DiskStorage>;
    type Host = Self::BlockData<PinnedStorage>;
    type Device = Self::BlockData<DeviceStorage>;

    type BlockData<S: Storage> = MockLogicalBlockData<S>;

    fn is_transfer_compatible_with<Other: LocalityProvider>() -> bool {
        // MockLogical is only compatible with MockLogical
        std::any::TypeId::of::<Other>() == std::any::TypeId::of::<MockLogical>()
    }

    fn transfer_mechanism_to<Other: LocalityProvider>() -> TransferMechanism {
        if Self::is_transfer_compatible_with::<Other>() {
            TransferMechanism::MockCounting
        } else {
            TransferMechanism::Unsupported
        }
    }
}

pub trait Parallelism: Send + Sync + 'static + std::fmt::Debug {
    type Output<S: Storage>: BlockDataStorage;
}

/// General logical locality for future RPC-based transfers
#[derive(Debug)]
pub struct Logical<P: Parallelism> {
    _parallelism: std::marker::PhantomData<P>,
}

impl<P: Parallelism> LocalityProvider for Logical<P> {
    type Disk = P::Output<DiskStorage>;
    type Host = P::Output<PinnedStorage>;
    type Device = P::Output<DeviceStorage>;

    type BlockData<S: Storage> = P::Output<S>;

    fn is_transfer_compatible_with<Other: LocalityProvider>() -> bool {
        // Future: implement proper compatibility logic for different logical patterns
        // For now, logical localities are only compatible with themselves
        std::any::TypeId::of::<Self>() == std::any::TypeId::of::<Other>()
    }

    fn transfer_mechanism_to<Other: LocalityProvider>() -> TransferMechanism {
        if Self::is_transfer_compatible_with::<Other>() {
            TransferMechanism::RemoteRpc
        } else {
            TransferMechanism::Unsupported
        }
    }
}

/// Local block data with direct memory access
#[derive(Debug)]
pub struct LocalBlockData<S: Storage> {
    block_data: BlockData<S>,
}

impl<S: Storage> LocalBlockData<S> {
    pub fn new(block_data: BlockData<S>) -> Self {
        Self { block_data }
    }

    pub fn block_data(&self) -> &BlockData<S> {
        &self.block_data
    }

    pub fn block_data_mut(&mut self) -> &mut BlockData<S> {
        &mut self.block_data
    }
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
    type BlockDataType<T: Storage> = BlockData<T>;

    fn block_data(&self) -> &Self::BlockDataType<Self::StorageType> {
        &self.block_data
    }

    fn block_data_mut(&mut self) -> &mut Self::BlockDataType<Self::StorageType> {
        &mut self.block_data
    }
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

/// Implementation of BlockDataExt for LocalBlockData to enable direct memory access
/// This delegates to the underlying BlockData implementation
impl<S: Storage + NixlDescriptor> BlockDataExt<S> for LocalBlockData<S> {
    fn num_layers(&self) -> usize {
        self.block_data.num_layers()
    }

    fn num_outer_dims(&self) -> usize {
        self.block_data.num_outer_dims()
    }

    fn layer_view(&self, layer_idx: usize, outer_idx: usize) -> BlockResult<view::LayerView<S>> {
        self.block_data.layer_view(layer_idx, outer_idx)
    }

    fn layer_view_mut(
        &mut self,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerViewMut<S>> {
        self.block_data.layer_view_mut(layer_idx, outer_idx)
    }

    fn block_view(&self) -> BlockResult<view::BlockView<S>> {
        self.block_data.block_view()
    }

    fn block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<S>> {
        self.block_data.block_view_mut()
    }
}

/// Mock logical block data that only stores metadata for size calculations
/// Does NOT provide access to actual memory - only computes transfer sizes
#[derive(Debug)]
pub struct MockLogicalBlockData<S: Storage> {
    // Store only the metadata needed to compute sizes
    num_layers: usize,
    outer_dim: usize,
    page_size: usize,
    inner_dim: usize,
    storage_type: StorageType,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Storage> MockLogicalBlockData<S> {
    pub fn new(
        num_layers: usize,
        outer_dim: usize,
        page_size: usize,
        inner_dim: usize,
        storage_type: StorageType,
    ) -> Self {
        Self {
            num_layers,
            outer_dim,
            page_size,
            inner_dim,
            storage_type,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute the total size in bytes for this block
    pub fn compute_total_size(&self) -> usize {
        // Compute based on layout: layers * outer_dim * page_size * inner_dim * element_size
        // For simplicity, assume 2 bytes per element (like f16)
        const ELEMENT_SIZE: usize = 2;
        self.num_layers * self.outer_dim * self.page_size * self.inner_dim * ELEMENT_SIZE
    }

    /// Compute the size of a specific layer
    pub fn compute_layer_size(&self) -> usize {
        const ELEMENT_SIZE: usize = 2;
        self.outer_dim * self.page_size * self.inner_dim * ELEMENT_SIZE
    }
}

impl<S: Storage> BlockDataStorage for MockLogicalBlockData<S> {
    type StorageType = S;
    type BlockDataType<T: Storage> = MockLogicalBlockData<T>;

    fn block_data(&self) -> &Self::BlockDataType<Self::StorageType> {
        self
    }

    fn block_data_mut(&mut self) -> &mut Self::BlockDataType<Self::StorageType> {
        self
    }
}

impl<S: Storage> BlockDataLocality for MockLogicalBlockData<S> {
    fn storage_type(&self) -> StorageType {
        self.storage_type.clone()
    }
}

impl<S: Storage> BlockLayoutConfig for MockLogicalBlockData<S> {
    fn layout_type(&self) -> LayoutType {
        LayoutType::FullyContiguous // Mock as fully contiguous for simplicity
    }

    fn num_blocks(&self) -> usize {
        1 // Mock blocks are always single blocks
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn outer_dim(&self) -> usize {
        self.outer_dim
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn inner_dim(&self) -> usize {
        self.inner_dim
    }
}

// NOTE: MockLogicalBlockData does NOT implement BlockDataExt
// This ensures it cannot access actual memory views, only compute sizes

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
        type StorageType = S;
        type BlockDataType<T: Storage> = Self;

        fn block_data(&self) -> &Self::BlockDataType<Self::StorageType> {
            &self
        }

        fn block_data_mut(&mut self) -> &mut Self::BlockDataType<Self::StorageType> {
            self
        }
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
