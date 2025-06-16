use super::*;

use derive_getters::Dissolve;

/// Collection that holds shared storage and layout
#[derive(Debug, Dissolve)]
pub struct LocalBlockDataFactoryMut<L: BlockLayout> {
    layout: L,
    block_set_idx: usize,
    worker_id: WorkerID,
}

pub struct LocalBlockDataFactory<L: BlockLayout> {
    layout: Arc<L>,
    block_set_idx: usize,
    worker_id: WorkerID,
}

impl<L: BlockLayout + 'static> LocalBlockDataFactoryMut<L> {
    /// Create a new block storage collection
    pub fn new(layout: L, block_set_idx: usize, worker_id: WorkerID) -> BlockResult<Self> {
        Ok(Self {
            layout,
            block_set_idx,
            worker_id,
        })
    }

    pub fn into_factory(self) -> LocalBlockDataFactory<L> {
        let (layout, block_set_idx, worker_id) = self.dissolve();
        let layout = Arc::new(layout);

        LocalBlockDataFactory {
            layout,
            block_set_idx,
            worker_id,
        }
    }
}

impl<L: BlockLayout + 'static> LocalBlockDataFactory<L> {
    pub fn create_block_data(&self, block_idx: BlockId) -> BlockResult<BlockData<L::StorageType>> {
        if block_idx >= self.layout.num_blocks() {
            return Err(BlockError::InvalidBlockID(block_idx));
        }

        let data = BlockData::new(
            self.layout.clone(),
            block_idx,
            self.block_set_idx,
            self.worker_id,
        );
        Ok(data)
    }
}

pub mod nixl {
    use super::*;
    use crate::block_manager::{
        layout::nixl::{NixlLayout, SerializedNixlBlockLayout},
        storage::nixl::{NixlRegisterableStorage, NixlStorage},
    };
    use nixl_sys::{Agent as NixlAgent, OptArgs};

    impl<L: NixlLayout> LocalBlockDataFactoryMut<L>
    where
        L::StorageType: NixlRegisterableStorage,
    {
        /// Register the blocks with an NIXL agent
        pub fn nixl_register(
            &mut self,
            agent: &NixlAgent,
            opt_args: Option<&OptArgs>,
        ) -> anyhow::Result<()> {
            self.layout.nixl_register(agent, opt_args)
        }

        /// Convert the factory into a remote factory
        pub fn into_remote_factory(self) -> BlockResult<RemoteBlockDataFactory> {
            let serialized_layout = self.layout.serialize()?;
            let layout = serialized_layout.deserialize()?;

            Ok(RemoteBlockDataFactory {
                layout,
                block_set_idx: self.block_set_idx,
                worker_id: self.worker_id,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct RemoteBlockDataFactory {
        layout: Arc<dyn BlockLayout<StorageType = NixlStorage>>,
        block_set_idx: usize,
        worker_id: WorkerID,
    }
}

// // /// Convert collection into Vec<Block> with default metadata/state
// // pub fn into_blocks(self) -> BlockResult<Vec<Block<L::StorageType, M>>> {
// //     // convert box to arc
// //     let layout: Arc<dyn BlockLayout<StorageType = L::StorageType>> = Arc::new(*self.layout);
// //     layout_to_blocks(layout, self.block_set_idx, self.worker_id)
// // }

// pub fn create_block_data(&self, block_id: BlockID) -> BlockResult<BlockData<L::StorageType>> {
//     let layout = Arc::new(self.layout);
//     let data = BlockData::new(layout, block_id, self.block_set_idx, self.worker_id);
//     Ok(data)
// }

pub(crate) fn layout_to_data<S: Storage, M: BlockMetadata>(
    layout: Arc<dyn BlockLayout<StorageType = S>>,
    block_set_idx: usize,
    worker_id: WorkerID,
) -> BlockResult<Vec<Block<S, M>>> {
    (0..layout.num_blocks())
        .map(|idx| {
            let data = BlockData::new(layout.clone(), idx, block_set_idx, worker_id);
            Block::new(data, M::default())
        })
        .collect()
}
