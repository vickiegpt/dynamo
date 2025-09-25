# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal scheduler connector worker implementation for testing.

This is a barebones implementation that provides no-op responses,
used specifically for scheduler integration testing without actual KV transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

# Import our local block builder
from dynamo._core import scheduler_connector

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext


class SchedulerConnectorWorker:
    """
    Minimal scheduler connector worker that provides no-op implementations.

    This connector is used for scheduler integration where no actual
    KV transfer is needed. All methods are no-ops or return minimal responses.
    """

    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        """Initialize the scheduler connector worker."""
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self.local_blocks = None
        print(f"SchedulerConnectorWorker initialized with engine_id: {engine_id}")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Register KV caches - builds local blocks without leader sync.

        This creates device blocks locally from the provided tensors
        without requiring any network setup or synchronization.
        """
        if not kv_caches:
            print("Warning: register_kv_caches called with empty kv_caches")
            return

        print(
            f"SchedulerConnectorWorker.register_kv_caches called with {len(kv_caches)} layers"
        )

        # Extract configuration from vLLM config
        cache_config = self.vllm_config.cache_config

        # Sort tensors by layer index to ensure correct ordering
        ordered_kv_caches = sorted(
            kv_caches.items(), key=lambda item: extract_layer_index(item[0])
        )

        # Extract tensors in order
        tensors = [tensor for _, tensor in ordered_kv_caches]

        # Get first tensor to extract common properties
        first_tensor = tensors[0]
        shape = first_tensor.shape

        # Validate all tensors have same shape
        if not all(t.shape == shape for t in tensors):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

        # Extract parameters
        # TODO: Assume the block dimension is within the first 2. This will break if you're doing something weird
        num_device_blocks = max(shape[0], shape[1])
        page_size = cache_config.block_size
        device_id = (
            first_tensor.device.index if first_tensor.device.type == "cuda" else 0
        )

        # Determine cache dtype
        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = self.vllm_config.model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        dtype_width_bytes = kv_cache_dtype.itemsize

        # Build worker device blocks
        try:
            self.local_blocks = scheduler_connector.WorkerDeviceBlocks(
                tensors=tensors,
                num_device_blocks=num_device_blocks,
                page_size=page_size,
                device_id=device_id,
                dtype_width_bytes=dtype_width_bytes,
                is_fully_contiguous=False,  # Default to layer-separate layout
            )

            print(f"Successfully built worker device blocks: {self.local_blocks}")
            print(f"  - Blocks created: {self.local_blocks.num_blocks()}")
            print(f"  - Layers: {self.local_blocks.num_layers}")
            print(f"  - Outer dim: {self.local_blocks.outer_dim}")
            print(f"  - Page size: {self.local_blocks.page_size}")
            print(f"  - Inner dim: {self.local_blocks.inner_dim}")
            print(f"  - Bytes per block: {self.local_blocks.bytes_per_block}")

        except Exception as e:
            print(f"Failed to build worker device blocks: {e}")
            raise

    def bind_connector_metadata(self, data: bytes) -> None:
        """
        Bind connector metadata - no-op.

        Since our leader returns empty bytes, this is always a no-op.
        """
        pass

    def clear_connector_metadata(self) -> None:
        """
        Clear connector metadata - no-op.
        """
        pass

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading KV cache - no-op.

        No KV loading needed for scheduler connector.
        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Save KV layer - no-op.

        No KV saving needed for scheduler connector.
        """
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Get finished request IDs.

        Since request_finished() always returns False (never delays block freeing),
        we just acknowledge the finished requests but don't return any as finished
        for KV transfer purposes.

        Returns:
            (None, None): No finished sends/receives
        """
        # Just acknowledge the finished requests
        # Since our leader's request_finished() always returns False,
        # these requests have already had their blocks freed
        if len(finished_req_ids) > 0:
            print(
                f"SchedulerConnectorWorker.get_finished() acknowledging {len(finished_req_ids)} finished requests"
            )

        return (None, None)
