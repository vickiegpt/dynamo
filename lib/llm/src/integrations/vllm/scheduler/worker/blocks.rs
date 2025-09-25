// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker device blocks construction for scheduler.
//!
//! This module provides a simplified way to build device blocks from tensors
//! on the worker side without requiring leader/worker synchronization or network setup.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::block_manager::{
    BasicMetadata, LayoutConfigBuilder, NixlLayout,
    block::{Block, layout_to_blocks, locality},
    layout::LayoutType,
    storage::{DeviceAllocator, DeviceStorage, torch::TorchTensor},
};

/// Container for worker device blocks constructed locally.
pub struct WorkerDeviceBlocks {
    /// Device blocks constructed from KV cache tensors
    pub device_blocks: Vec<Block<DeviceStorage, locality::Local, BasicMetadata>>,

    /// Metadata about the layout
    pub num_device_blocks: usize,
    pub num_layers: usize,
    pub outer_dim: usize,
    pub page_size: usize,
    pub inner_dim: usize,
    pub dtype_width_bytes: usize,
    pub bytes_per_block: usize,
}

impl WorkerDeviceBlocks {
    /// Build device blocks locally from KV cache tensors.
    ///
    /// This is a simplified version of KvbmWorker's initialization that:
    /// - Validates tensor consistency
    /// - Infers layout configuration
    /// - Creates device blocks
    /// - Does NOT perform leader sync or network setup
    pub fn from_tensors(
        tensors: Vec<Arc<dyn TorchTensor>>,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        is_fully_contiguous_layout: bool,
    ) -> Result<Self> {
        if num_device_blocks == 0 {
            return Err(anyhow!("num_device_blocks must be greater than 0"));
        }

        if tensors.is_empty() {
            return Err(anyhow!("tensors cannot be empty"));
        }

        // Validate tensors and get device storage
        let (device_tensors, shape) = Self::load_and_validate_tensors(&tensors, device_id)?;

        if shape.len() < 3 {
            return Err(anyhow!(
                "Unsupported kv cache layout. Got shape: {:?}",
                shape
            ));
        }

        // Infer layout configuration
        let (layout_type, num_layers, outer_dim, inner_dim) = if !is_fully_contiguous_layout {
            let (outer_contiguous, outer_dim) = if shape[0] >= num_device_blocks {
                (false, shape[1])
            } else if shape[1] >= num_device_blocks {
                (true, shape[0])
            } else {
                return Err(anyhow!(
                    "Unsupported kv cache layout. Got shape: {:?}",
                    shape
                ));
            };
            let num_layers = device_tensors.len();
            let inner_dim = shape[2..].iter().product::<usize>() / page_size;

            (
                LayoutType::LayerSeparate { outer_contiguous },
                num_layers,
                outer_dim,
                inner_dim,
            )
        } else {
            let num_layers = shape[1];
            let outer_dim = shape[2];
            let inner_dim = shape[3..].iter().product::<usize>() / page_size;

            (
                LayoutType::FullyContiguous,
                num_layers,
                outer_dim,
                inner_dim,
            )
        };

        let bytes_per_block =
            num_layers * outer_dim * page_size * inner_dim * dtype_width_bytes;

        // Build layout
        let mut layout_builder_instance = LayoutConfigBuilder::default();
        let layout_builder = layout_builder_instance
            .num_layers(num_layers)
            .outer_dim(outer_dim)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype_width_bytes(dtype_width_bytes);

        let device_layout = layout_builder
            .num_blocks(num_device_blocks)
            .build()?
            .create_layout(layout_type, device_tensors)?;

        // Convert layout to blocks
        let device_blocks = Self::make_layout(device_layout)?;

        Ok(Self {
            device_blocks,
            num_device_blocks,
            num_layers,
            outer_dim,
            page_size,
            inner_dim,
            dtype_width_bytes,
            bytes_per_block,
        })
    }

    /// Validate tensors and create device storage.
    fn load_and_validate_tensors(
        tensors: &[Arc<dyn TorchTensor>],
        device_id: usize,
    ) -> Result<(Vec<DeviceStorage>, Vec<usize>)> {
        let mut shape = None;
        let mut device_tensors = Vec::with_capacity(tensors.len());
        let allocator = DeviceAllocator::new(device_id)?;

        for tensor in tensors {
            // Check the stride
            let stride = tensor.stride();
            for i in 1..stride.len() {
                if stride[i] > stride[i - 1] {
                    return Err(anyhow!(
                        "Tensor strides must be monotonically decreasing! Got {:?}",
                        stride
                    ));
                }
            }

            // Check that all tensors have the same shape
            if let Some(shape) = shape.as_ref() {
                if *shape != tensor.shape() {
                    return Err(anyhow!(
                        "All tensors must have the same shape! Got {:?} and {:?}",
                        *shape,
                        tensor.shape()
                    ));
                }
            } else {
                shape = Some(tensor.shape());
            }

            // Build the storage object from the tensor
            let device_tensor = DeviceStorage::new_from_torch(allocator.ctx(), tensor.clone())?;
            device_tensors.push(device_tensor);
        }

        Ok((device_tensors, shape.unwrap()))
    }

    /// Convert layout to blocks without NIXL registration.
    fn make_layout(
        layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
    ) -> Result<Vec<Block<DeviceStorage, locality::Local, BasicMetadata>>> {
        // Convert to Arc for layout_to_blocks
        let layout: Arc<dyn NixlLayout<StorageType = DeviceStorage>> = Arc::from(layout);

        // Create blocks with block_set_idx=0, worker_id=0 (local only)
        let blocks = layout_to_blocks::<_, BasicMetadata>(layout, 0, 0)?;

        Ok(blocks)
    }
}