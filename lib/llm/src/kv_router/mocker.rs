use std::collections::{BinaryHeap, HashMap};
use std::cmp::{Eq, Ord, Ordering, PartialOrd};
use std::hash::Hash;
use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};
use crate::kv_router::protocols::{DirectRequest, SequenceHashWithDepth};
use crate::kv_router::indexer::RadixTree;

/// Wrapper for f64 that implements total ordering and panics when NaN is encountered
#[derive(Debug, Clone, Copy, PartialEq)]
struct TimeStamp(f64);

impl PartialOrd for TimeStamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Eq for TimeStamp {}

impl Ord for TimeStamp {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
} 
/// An LRU evictor that maintains objects and evicts them based on their
/// last accessed time.
pub struct LRUEvictor<T: Clone + Eq + Hash + Ord> {
    // Maps object keys to their metadata
    pub free_table: HashMap<T, f64>,
    // Min-heap of (last_accessed, object) for efficient eviction
    priority_queue: BinaryHeap<(Reverse<TimeStamp>, T)>,
    // Cleanup threshold as a ratio of priority_queue size to free_table size
    cleanup_threshold: usize,
}

impl<T: Clone + Eq + Hash + Ord> LRUEvictor<T> {
    /// Create a new LRUEvictor with the default cleanup threshold
    pub fn new() -> Self {
        Self::with_cleanup_threshold(50)
    }

    /// Create a new LRUEvictor with a custom cleanup threshold
    pub fn with_cleanup_threshold(cleanup_threshold: usize) -> Self {
        LRUEvictor {
            free_table: HashMap::new(),
            priority_queue: BinaryHeap::new(),
            cleanup_threshold,
        }
    }

    /// Check if the evictor contains the given object
    pub fn contains(&self, object: &T) -> bool {
        self.free_table.contains_key(object)
    }

    /// Evict an object based on LRU policy
    /// Returns the evicted object or an error if no objects are available
    pub fn evict(&mut self) -> Result<T, String> {
        if self.free_table.is_empty() {
            return Err("No usable cache memory left".to_string());
        }

        while let Some((Reverse(last_accessed), object)) = self.priority_queue.pop() {
            // Check if the entry is still valid (not outdated)
            if let Some(&current_last_accessed) = self.free_table.get(&object) {
                if current_last_accessed == last_accessed.0 {
                    // The entry is valid, remove it from the free table
                    self.free_table.remove(&object);
                    return Ok(object);
                }
                // Otherwise, this is an outdated entry and we skip it
            }
        }

        Err("No usable cache memory left".to_string())
    }

    /// Add a new object to the evictor
    pub fn add(&mut self, object: T, last_accessed: f64) {        
        self.free_table.insert(object.clone(), last_accessed);
        self.priority_queue.push((Reverse(TimeStamp(last_accessed)), object.clone()));
        self.cleanup_if_necessary();
    }

    /// Update the last_accessed time for an object with a specific timestamp
    pub fn update(&mut self, object: &T, last_accessed: f64) {
        if let Some(entry) = self.free_table.get_mut(object) {
            *entry = last_accessed;
        }
    }

    /// Remove an object from the evictor
    /// We don't remove from the priority queue immediately, as that would be inefficient
    /// Outdated entries will be filtered out during eviction or cleanup
    pub fn remove(&mut self, object: &T) -> Result<(), String> {
        match self.free_table.remove(object) {
            Some(_) => Ok(()),
            None => Err("Attempting to remove object that's not in the evictor".to_string()),
        }
    }

    /// Get the number of objects in the evictor
    pub fn num_objects(&self) -> usize {
        self.free_table.len()
    }

    /// Check if cleanup is necessary and perform it if needed
    fn cleanup_if_necessary(&mut self) {
        if self.priority_queue.len() > self.cleanup_threshold * self.free_table.len() {
            self.cleanup();
        }
    }

    /// Clean up the priority queue by removing outdated entries
    fn cleanup(&mut self) {
        let mut new_priority_queue = BinaryHeap::new();

        for (object, &last_accessed) in &self.free_table {
            new_priority_queue.push((Reverse(TimeStamp(last_accessed)), object.clone()));
        }

        self.priority_queue = new_priority_queue;
    }
}

/// Mock implementation of workers for testing and simulation
pub struct MockWorkers {
    pub num_workers: u64,
    pub active_blocks: Vec<HashSet<SequenceHashWithDepth>>,
    pub inactive_blocks: Vec<LRUEvictor<SequenceHashWithDepth>>,
    pub waiting_blocks: Vec<VecDeque<DirectRequest>>,
    pub radix_tree: RadixTree,
}

impl MockWorkers {
    /// Create a new MockWorkers instance
    pub fn new(num_workers: u64) -> Self {
        let mut active_blocks = Vec::with_capacity(num_workers as usize);
        let mut inactive_blocks = Vec::with_capacity(num_workers as usize);
        let mut waiting_blocks = Vec::with_capacity(num_workers as usize);
        
        for _ in 0..num_workers {
            active_blocks.push(HashSet::new());
            inactive_blocks.push(LRUEvictor::new());
            waiting_blocks.push(VecDeque::new());
        }
        
        MockWorkers {
            num_workers,
            active_blocks,
            inactive_blocks,
            waiting_blocks,
            radix_tree: RadixTree::new(),
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_lru_evictor_eviction_order() {
        // Create a new LRUEvictor with default cleanup threshold
        let mut evictor = LRUEvictor::<i32>::new();
        
        // Get current timestamp
        let timestamp1 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        
        // Add two integers with the same timestamp
        evictor.add(1, timestamp1);
        evictor.add(2, timestamp1);
        
        // Get a new timestamp (slightly later)
        let timestamp2 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        
        // Add three more integers with the new timestamp
        evictor.add(1, timestamp2); // This updates the timestamp for 1
        evictor.add(3, timestamp2);
        evictor.add(4, timestamp2);
        
        // Evict an object (should be 2 as it has the oldest timestamp)
        let evicted = evictor.evict().unwrap();
        assert_eq!(evicted, 2);
        
        // Check that free_table contains 1, 3, 4 and nothing else
        assert_eq!(evictor.num_objects(), 3);
        assert!(evictor.contains(&1));
        assert!(evictor.contains(&3));
        assert!(evictor.contains(&4));
    }
}