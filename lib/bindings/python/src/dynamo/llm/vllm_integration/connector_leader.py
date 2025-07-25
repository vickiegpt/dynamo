# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


# from dynamo.llm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
# from dynamo.llm.vllm_integration.rust import BlockManager, KvbmRequest
# from dynamo.llm.vllm_integration.rust import KvConnectorLeader as RustKvConnectorLeader
# from dynamo.llm.vllm_integration.rust import (
#     KvConnectorMetadata as RustKvConnectorMetadata,
# )
# from dynamo.llm.vllm_integration.rust import SchedulerOutput as RustSchedulerOutput

from dynamo.llm.vllm_integration.rust import KvConnectorLeader as RustKvConnectorLeader


class KvConnectorLeader:
    """
    Implements the vLLM KV cache manager protocol.

    This class is a wrapper around the Rust KvbmCacheManager class.
    It is used to convert the Rust KvbmCacheManager into a Python class
    that can be used in the vLLM KV cache manager protocol.
    """

    def __init__(self, vllm_config: "VllmConfig", engine_id: str):
        self.vllm_config = vllm_config
        print(f"KvConnectorLeader initialized with engine_id: {engine_id}")
        self._connector = RustKvConnectorLeader(engine_id)

    # KV Connector

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps).
        """
        self._create_slot(request)
        return self._connector.get_num_new_matched_tokens(
            request.request_id,
            request.num_tokens,
            num_computed_tokens,
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        self._connector.update_state_after_alloc(request.request_id)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        output = RustSchedulerOutput()

        for req in scheduler_output.new_requests:
            output.add_new_request(
                request_id=req.request_id,
                prompt_token_ids=req.prompt_token_ids,
                block_ids=req.block_ids,
                num_computed_tokens=req.num_computed_tokens,
            )

        for req in scheduler_output.cached_requests:
            output.add_cached_request(
                request_id=req.request_id,
                resumed_from_preemption=req.resumed_from_preemption,
                new_token_ids=req.new_token_ids,
                new_block_ids=req.new_block_ids,
                num_computed_tokens=req.num_computed_tokens,
            )

        for req in scheduler_output.num_scheduled_tokens:
            output.add_num_scheduled_tokens(
                request_id=req.request_id,
                num_scheduled_tokens=req.num_scheduled_tokens,
            )

        assert (
            scheduler_output.total_num_scheduled_tokens
            == output.get_num_scheduled_tokens()
        ), "Total number of scheduled tokens does not match"

        return DynamoConnectorMetadata(self.connector.build_connector_metadata(output))

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # note our working can communication with us oob and we can use that to know
        # ahead of time if the request is finished.
        status = self._connector.request_finished(request.request_id, block_ids)
        return status, None

    # Utility functions

    def _create_slot(self, request: Request) -> None:
        """Create a slot for the request"""

        if self.connector.has_slot(request.request_id):
            return None

        if bool(request.mm_positions):
            raise ValueError("Unsupported request - requires mm extra keys")

        all_token_ids = request.all_token_ids

        # extract the critial aspects of the request that effect how the tokens are hashed
        request = KvbmRequest(
            request_id=request.request_id,
            lora_name=request.lora_request.lora_name()
            if request.lora_request
            else None,
            salt_hash=request.cache_salt,
        )

        self._connector.create_slot(request, all_token_ids)
