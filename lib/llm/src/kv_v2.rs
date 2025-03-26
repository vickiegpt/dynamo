


pub trait BlockState {

    fn sequence_hash(&self) -> u64;
    fn block_hash(&self) -> u64;
}
