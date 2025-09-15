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
        print(f"SchedulerConnectorWorker initialized with engine_id: {engine_id}")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Register KV caches - no-op for now.

        Will be implemented in a later phase.
        """
        # TODO: Implement in future phase
        pass

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
