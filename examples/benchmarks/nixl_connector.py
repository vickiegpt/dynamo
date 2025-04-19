# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import uuid
from nixl._api import nixl_agent
import logging

logger = logging.getLogger(__name__)


class NixlConnector:
    def __init__(self, engine_id: str, rank: int):
        # Create unique NIXL agent for this worker
        self.nixl_wrapper = nixl_agent(str(uuid.uuid4()), None)
        self.engine_id = engine_id
        self.rank = rank
        self.block_len = None  # Will be set during registration

    def register_kv_caches(self, kv_cache: torch.Tensor):
        # Get block size from the KV cache tensor
        # Note: KV cache layout depends on specific attention implementation
        num_blocks, block_size, num_heads, head_dim = kv_cache.shape
        self.block_len = block_size * num_heads * head_dim * kv_cache.element_size()
        self.num_blocks = num_blocks

        # Register KV cache tensor with NIXL for sharing
        base_addr = kv_cache.data_ptr()
        region_len = num_blocks * self.block_len
        caches_data = [(base_addr, region_len, self.rank, "")]

        # Register memory regions with NIXL
        descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
        self.nixl_wrapper.register_memory(descs)

        # Prepare local side of the transfer
        blocks_data = []
        for block_id in range(num_blocks):
            block_offset = block_id * self.block_len
            blocks_data.append((base_addr + block_offset, self.block_len, self.rank))

        # Create transfer descriptors and prepare for transfers
        self.local_blocks_descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")

        # Create transfer handle with block descriptors for future transfers
        self.local_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist("", self.local_blocks_descs)  # descs ?

    def get_agent_metadata(self):
        # Get metadata for sharing with other agents
        return self.nixl_wrapper.get_agent_metadata(), self.local_blocks_descs

    def add_remote_agent(self, engine_id: str, agent_metadata: bytes, remote_blocks_descs: bytes):
        # Connect to remote NIXL agent
        agent_name = self.nixl_wrapper.add_remote_agent(agent_metadata)

        # Prepare remote side transfer handle using provided block descriptors
        self.remote_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(agent_name, remote_blocks_descs)

        return agent_name

    def write_blocks(self, local_block_ids, remote_block_ids, notify_msg):
        # Initiate asynchronous transfer using block IDs
        # Block descriptors were specified during transfer preparation
        handle = self.nixl_wrapper.make_prepped_xfer(
            "WRITE",
            self.local_xfer_side_handle,
            local_block_ids,
            self.remote_xfer_side_handle,
            remote_block_ids,
            notify_msg
        )
        status = self.nixl_wrapper.transfer(handle)
