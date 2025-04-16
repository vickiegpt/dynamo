pub use super::block::*;
pub use super::events::*;
pub use super::pool::*;
pub use super::storage::*;

pub struct BlockStorageManager<S: Storage, M: BlockMetadata> {
    inactive: InactiveBlockPool<S, M>,
    active: ActiveBlockPool<S, M>,
    events: Arc<dyn EventManager>,
}
