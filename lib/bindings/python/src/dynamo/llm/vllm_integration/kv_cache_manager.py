# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from vllm.v1.request import Request

from dynamo.llm import BlockManager
from dynamo.llm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
from dynamo.llm.vllm_integration.rust import KvbmCacheManager as RustKvbmCacheManager
from dynamo.llm.vllm_integration.rust import KvbmRequest


class KvbmCacheManager:
    """
    Implements the vLLM KV cache manager protocol.

    This class is a wrapper around the Rust KvbmCacheManager class.
    It is used to convert the Rust KvbmCacheManager into a Python class
    that can be used in the vLLM KV cache manager protocol.
    """

    def __init__(self, block_manager: BlockManager):
        """
        Initializes the KvbmCacheManager.

        Args:
            block_manager: Python bound Dynamo KV Block Manager (KVBM).
        """
        # pass the python bound KVBM to the Rust KVBM cache manager
        # the rust cache manager will take ownership of the kvbm
        self.cache_manager = RustKvbmCacheManager(block_manager)
        self.block_size = block_manager.block_size()

    def get_computed_blocks(self, request: Request) -> tuple[KvbmCacheBlocks, int]:
        """
        Get the computed blocks for the request.
        """
        if bool(request.mm_positions):
            raise ValueError("Unsupported request - requires mm extra keys")

        # from dynamo_llm import KvRequestInputs
        request = KvbmRequest(
            request_id=request.request_id,
            tokens=request.all_token_ids,
            block_size=self.block_size,
            lora_name=request.lora_request.lora_name()
            if request.lora_request
            else None,
            salt_hash=request.cache_salt,
        )

        owned_blocks = self.cache_manager.get_computed_blocks(request)
        block_count = owned_blocks.block_count()

        return KvbmCacheBlocks(owned_blocks), block_count
