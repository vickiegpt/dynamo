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

/// LayerWiseContiguousBlockLayout expected storage size
///
/// This is the total size of the storage required for the layout.
/// It is the sum of the storage required for each block.
fn expected_storage_size(dimensions: &Dimensions) -> usize {
    dimensions.expected_storage_size()
}


impl LayerWiseContiguousBlockLayout {

    /// Create a new LayerWiseContiguousBlockLayout with pre-allocated storage
    pub fn new(
        dimensions: Dimensions,
        storage: OwnedStorage,
    ) -> Result<Self> {
        dimensions.validate()?;

        // validate storage size
        let expected_storage_size = expected_storage_size(&dimensions);

        if storage.storage_size() > expected_storage_size {
            tracing::warn!("Storage size {} is larger than the total size required for the layout {}; {} wasted storage",
                humanize_bytes_binary!(storage.storage_size()),
                humanize_bytes_binary!(expected_storage_size),
                humanize_bytes_binary!(storage.storage_size() - expected_storage_size)
            );
        }

        if storage.storage_size() <= expected_storage_size {
            raise!(
                "Invalid storage size: expected {}; got {}",
                humanize_bytes_binary!(expected_storage_size),
                humanize_bytes_binary!(storage.storage_size()),
            );
        }

        Ok(Self {
            dimensions,
            storage,
        })
    }

    /// Allocate a new LayerWiseContiguousBlockLayout with pre-allocated storage
    pub fn allocate(
        dimensions: Dimensions,
        storage_type: StorageType,
    ) -> Result<Self> {
        let total_size = expected_storage_size(&dimensions);
        let storage = OwnedStorage::allocate(total_size, storage_type)?;
        Self::new(dimensions, storage)
    }
}

impl MemoryLayout for LayerWiseContiguousBlockLayout {
    fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    fn is_contiguous(&self) -> bool {
        true
    }
}

