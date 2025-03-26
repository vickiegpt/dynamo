use super::*;

// Host memory but special class of host memory
pub struct PinnedBlockStorage {
    block_id: usize,
    block_storage: Arc<KvBlockStorage>,
}
pub struct DeviceBlockStorage {
    block_id: usize,
    block_storage: Arc<KvBlockStorage>,
}

// pub type KvBlockPinned = KvBlock<PinnedBlockStorage>;
// pub type KvBlockDevice = KvBlock<DeviceBlockStorage>;

impl PinnedBlockStorage {
    pub fn new(block_id: usize, block_storage: Arc<KvBlockStorage>) -> Self {
        Self {
            block_id,
            block_storage,
        }
    }
}

// impl BlockStorage for PinnedBlockStorage {

//     fn memory_layout(&self) -> MemoryLayout {

// }

// impl DeviceBlockStorage {
//     pub fn new(block_id: usize, block_storage: Arc<KvBlockStorage>) -> Self {
//         Self {
//             block_id,
//             block_storage,
//         }
//     }
// }
// impl BlockStorage for DeviceBlockStorage {
//     fn k_ptr(&self, layer_id: usize) -> Result<u64> {
//         self.block_storage.k_ptr(self.block_id, layer_id)
//     }

//     fn v_ptr(&self, layer_id: usize) -> Result<u64> {
//         self.block_storage.v_ptr(self.block_id, layer_id)
//     }

//     fn bytes_per_block_per_k_or_v(&self) -> usize {
//         self.block_storage.bytes_per_block_per_k_or_v()
//     }

//     fn k_and_v_are_contiguous(&self) -> bool {
//         self.block_storage.k_and_v_are_contiguous()
//     }
// }
