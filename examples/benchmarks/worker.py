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

import base64
import logging
import pickle

import torch
from nixl_connector import NixlConnector
from pydantic import BaseModel

from dynamo.sdk import depends, dynamo_endpoint, service

logger = logging.getLogger(__name__)


class RequestType(BaseModel):
    text: str


class ResponseType(BaseModel):
    text: str


class KVRequestType(BaseModel):
    text: str
    decode_metadata: bytes
    decode_blocks_descs: bytes
    block_id: int


class KVResponseType(BaseModel):
    text: str


@service(
    dynamo={"enabled": True, "namespace": "benchmarks"},
    # resources={"gpu": 1},
    workers=1,
)
class RemoteWorker:
    def __init__(self) -> None:
        logger.info("Starting Remote Worker...")

        self.nixl_connector = NixlConnector(engine_id="1", rank=0)

        num_blocks, block_size, num_heads, head_dim = (1024, 128, 1, 1)
        self.kv_cache = (
            torch.randn(
                (num_blocks, block_size, num_heads, head_dim),
                dtype=torch.float16,
                device="cuda",
            )
            * 32767
        )
        self.nixl_connector.register_kv_caches(self.kv_cache)

    @dynamo_endpoint()
    async def generate(self, request: KVRequestType):
        logger.info(f"Remore Worker received: {request.text}")

        decode_engine_id = "1"
        self.nixl_connector.add_remote_agent(
            decode_engine_id,
            base64.b64decode(request.decode_metadata),
            pickle.loads(base64.b64decode(request.decode_blocks_descs)),
        )
        self.nixl_connector.write_blocks(
            [request.block_id], [request.block_id], "kv_transfer"
        )

        yield KVResponseType(text="done").model_dump_json()


@service(
    dynamo={"enabled": True, "namespace": "benchmarks"},
    resources={"gpu": 1},
    workers=1,
)
class MainWorker:
    remote_worker = depends(RemoteWorker)

    def __init__(self) -> None:
        logger.info("Starting Main Worker...")

        self.nixl_connector = NixlConnector(engine_id="0", rank=0)

        num_blocks, block_size, num_heads, head_dim = (1024, 128, 1, 1)
        self.kv_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=torch.float16,
            device="cuda",
        )
        self.nixl_connector.register_kv_caches(self.kv_cache)

        (
            self.decode_metadata,
            self.decode_blocks_descs,
        ) = self.nixl_connector.get_agent_metadata()
        # logger.info(f"Decode Metadata: {self.decode_metadata}")
        # logger.info(f"Decode Blocks Descs: {self.decode_blocks_descs}")

        self.request_count = 0

    @dynamo_endpoint()
    async def generate(self, request: RequestType):
        logger.info(f"Main Worker received: {request.text}")

        curr_request_count = self.request_count
        self.request_count += 1

        logger.info(
            f"kv_cache[{curr_request_count, 0, 0, 0}]: {self.kv_cache[curr_request_count, 0, 0, 0]}"
        )

        remote_request = KVRequestType(
            text=request.text,
            decode_metadata=base64.b64encode(self.decode_metadata),
            decode_blocks_descs=base64.b64encode(
                pickle.dumps(self.decode_blocks_descs)
            ),
            block_id=curr_request_count,
        )
        remote_responses = self.remote_worker.generate(remote_request.model_dump_json())
        async for response in remote_responses:
            logger.info(f"Main Worker remote response: {response}")

        if curr_request_count >= 1:
            logger.info(
                f"kv_cache[{curr_request_count - 1, 0, 0, 0}]: {self.kv_cache[curr_request_count - 1, 0, 0, 0]}"
            )
        logger.info(
            f"kv_cache[{curr_request_count, 0, 0, 0}]: {self.kv_cache[curr_request_count, 0, 0, 0]}"
        )
        logger.info(
            f"kv_cache[{curr_request_count + 1, 0, 0, 0}]: {self.kv_cache[curr_request_count + 1, 0, 0, 0]}"
        )

        for token in request.text.split():
            yield ResponseType(text=token).model_dump_json()
