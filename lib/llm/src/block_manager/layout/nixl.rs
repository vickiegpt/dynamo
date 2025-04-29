use super::{BlockLayout, BlockLayoutConfig, LayoutError, Storage};

use super::super::storage::nixl::{MemType, NixlAgent, NixlEnabledStorage, NixlStorage, OptArgs};
use super::{FullyContiguous, FullyContiguousConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Extends [BlockLayout] with NIXL-specific methods for registering with an NIXL agent.
pub trait NixlLayout: BlockLayout + BlockLayoutNixlStorage {
    fn nixl_register(
        &mut self,
        agent: &NixlAgent,
        opt_args: Option<&OptArgs>,
    ) -> anyhow::Result<()>;
}

pub trait BlockLayoutNixlStorage {
    fn mem_type(&self) -> MemType;
    fn device_id(&self) -> u64;
}

// Umbrella impl for all BlockLayout types that are NixlEnabledStorage
impl<T> NixlLayout for T
where
    T: BlockLayout + BlockLayoutNixlStorage + ?Sized, // Implement for any T that is BlockLayout (potentially unsized)
    T::StorageType: NixlEnabledStorage, // T's associated StorageType must be NixlStorage
{
    fn nixl_register(
        &mut self,
        agent: &NixlAgent,
        opt_args: Option<&OptArgs>,
    ) -> anyhow::Result<()> {
        for storage in self.storage_mut() {
            storage.nixl_register(agent, opt_args)?;
        }
        Ok(())
    }
}

// impl LayoutConfig {
//     pub fn create_layout<S: Storage + NixlEnabledStorage>(
//         &self,
//         storage: Vec<S>,
//     ) -> Result<impl NixlLayout<StorageType = S>, LayoutError> {
//         let layout = FullyContiguous::new(self, storage)?;
//         Ok(Box::new(layout))
//     }
// }

/// Trait to convert a BlockLayout instance into its NIXL-specific serializable representation.
pub trait ToSerializedNixlBlockLayout: BlockLayout<StorageType: NixlEnabledStorage> {
    /// Converts the layout into a serializable format, ensuring it's backed by NIXL storage.
    /// Returns an error if the layout is not backed by storage providing NIXL descriptors.
    fn serialize(&self) -> Result<SerializedNixlBlockLayout, LayoutError>;
}

/// Serializable representation of a BlockLayout backed by NIXL storage.
pub struct SerializedNixlBlockLayout(Vec<u8>);

/// Enum representing the serializable state of different BlockLayout types
/// specifically when backed by NIXL-compatible storage.
#[derive(Serialize, Deserialize, Debug, Clone)]
enum NixlBlockLayoutKinds {
    FullyContiguous(SerializableNixlLayout<FullyContiguousConfig>),
    // Add variants for other layout types here
}

/// Serializable representation of FullyContiguous layout backed by NIXL storage.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerializableNixlLayout<C: BlockLayoutConfig> {
    config: C,
    base_offset: usize,
    storage_descriptors: Vec<NixlStorage>,
}

impl<C> SerializableNixlLayout<C>
where
    C: BlockLayoutConfig + Serialize + for<'de> Deserialize<'de> + Clone + std::fmt::Debug,
{
    /// Create a new SerializableNixlLayout
    fn new(config: C, base_offset: usize, storage_descriptors: Vec<NixlStorage>) -> Self {
        Self {
            config,
            base_offset,
            storage_descriptors,
        }
    }
}

impl<S: NixlEnabledStorage> ToSerializedNixlBlockLayout for FullyContiguous<S> {
    fn serialize(&self) -> Result<SerializedNixlBlockLayout, LayoutError> {
        // Use accessors added previously
        let config = self.config.clone();
        let base_offset = self.base_offset;

        let storages = self.storage();

        if storages.len() != 1 {
            return Err(LayoutError::InvalidConfig(
                "FullyContiguous reconstruction expects exactly one NixlStorage descriptor"
                    .to_string(),
            ));
        }

        // FullyContiguous uses a Vec<Storage>, but should only contain one element.
        let storage_instance = storages.first().ok_or_else(|| {
            LayoutError::OperationFailed("FullyContiguous requires one storage element".to_string())
        })?;

        let storage_descriptors = storage_instance.get_nixl_descriptors().ok_or_else(|| {
            LayoutError::OperationFailed(
                "Storage does not provide NIXL descriptors for serialization".to_string(),
            )
        })?;

        let serializable_data =
            SerializableNixlLayout::new(config, base_offset, vec![storage_descriptors]);

        let nixl_block_layout = NixlBlockLayoutKinds::FullyContiguous(serializable_data);

        Ok(SerializedNixlBlockLayout(serde_json::to_vec(
            &nixl_block_layout,
        )?))
    }
}

impl SerializedNixlBlockLayout {
    /// Reconstructs a dynamic BlockLayout trait object backed by NixlStorage
    /// from the serialized layout information.
    /// Assumes the NixlStorage regions described within already exist and are valid.
    pub fn deserialize(
        &self,
    ) -> Result<Arc<dyn BlockLayout<StorageType = NixlStorage>>, LayoutError> {
        let nixl_block_layout: NixlBlockLayoutKinds = serde_json::from_slice(&self.0)?;
        match nixl_block_layout {
            NixlBlockLayoutKinds::FullyContiguous(config) => {
                if config.storage_descriptors.len() != 1 {
                    return Err(LayoutError::InvalidConfig(
                        "FullyContiguous reconstruction expects exactly one NixlStorage descriptor"
                            .to_string(),
                    ));
                }
                // Clone the single NixlStorage descriptor to become the storage instance
                let storage = config.storage_descriptors[0].clone();

                // Use the internal constructor which skips allocation checks
                let layout = FullyContiguous::new_internal(
                    config.config.clone(),
                    storage, // Pass the NixlStorage instance
                    config.base_offset,
                )?;
                Ok(Arc::new(layout))
            } // Handle other variants when added...
        }
    }
}

impl<S> BlockLayoutNixlStorage for FullyContiguous<S>
where
    S: Storage + NixlEnabledStorage,
{
    fn mem_type(&self) -> MemType {
        self.storage.mem_type()
    }

    fn device_id(&self) -> u64 {
        self.storage.device_id()
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::block_manager::storage::SystemAllocator;
    use dynamo_runtime::logging::init as init_logging;

    #[test]
    fn test_nixl_layout() {
        init_logging();

        let config = LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(2)
            .page_size(4)
            .inner_dim(13)
            .build()
            .unwrap();

        config.validate().unwrap();

        let mut layout = FullyContiguous::allocate(config, &SystemAllocator::default()).unwrap();
        let agent = NixlAgent::new("test").unwrap();

        tracing::info!("Registering layout");
        layout.nixl_register(&agent, None).unwrap();
        tracing::info!("Layout registered");

        let serialized = layout.serialize().unwrap();

        let remote_layout = SerializedNixlBlockLayout::deserialize(&serialized).unwrap();
        println!("Nixl layout: {:?}", remote_layout);

        drop(layout);
        tracing::info!("Layout dropped");
    }
}
