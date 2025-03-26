// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

fn expected_storage_size_per_layer(dimensions: &Dimensions) -> usize {
    dimensions.expected_storage_per_block_per_layer() * dimensions.n_blocks
}

impl LayerWiseSlabContiguousBlockLayout {
    pub fn new(
        dimensions: Dimensions,
        layer_storage: Vec<OwnedStorage>,
    ) -> Result<Self> {
        dimensions.validate()?;

        // validate storage size
        let expected_layer_size = expected_storage_size_per_layer(&dimensions);

        for layer in &layer_storage {
            if layer.storage_size() < expected_layer_size {
                raise!(
                    "Invalid storage size: expected {}; got {}",
                    humanize_bytes_binary!(expected_layer_size),
                    humanize_bytes_binary!(layer.storage_size()),
                );
            }

            if layer.storage_size() > expected_layer_size {
                tracing::warn!("Storage size {} is larger than the total size required for the layout {}; {} wasted storage",
                    humanize_bytes_binary!(layer.storage_size()),
                    humanize_bytes_binary!(expected_layer_size),
                    humanize_bytes_binary!(layer.storage_size() - expected_layer_size)
                );
            }
        }


        Ok(Self {
            dimensions,
            layers: layer_storage,
        })
    }

    /// Create a new LayerWiseSlabContiguousBlockLayout by allocating memory per layer
    pub fn allocate(&self, dimensions: Dimensions, storage_type: StorageType) -> Result<Self> {
        let expected_layer_size = expected_storage_size_per_layer(&dimensions);
        let mut layers = Vec::new();
        for _ in 0..dimensions.n_layers {
            let layer = OwnedStorage::allocate(expected_layer_size, storage_type.clone())?;
            layers.push(layer);
        }
        Self::new(dimensions, layers)
    }

}

impl MemoryLayout for LayerWiseSlabContiguousBlockLayout {
    fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    fn is_contiguous(&self) -> bool {
        false
    }
}


