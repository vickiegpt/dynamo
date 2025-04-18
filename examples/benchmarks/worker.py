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

import logging

from pydantic import BaseModel

from dynamo.sdk import depends, dynamo_endpoint, service

logger = logging.getLogger(__name__)


class RequestType(BaseModel):
    text: str


class ResponseType(BaseModel):
    text: str


@service(dynamo={"enabled": True, "namespace": "benchmarks"})
class RemoteWorker:
    def __init__(self) -> None:
        logger.info("Starting Remote Worker...")

    @dynamo_endpoint()
    async def generate(self, request: RequestType):
        logger.info(f"Remore Worker received: {request.text}")

        for token in request.text.split():
            yield ResponseType(text=token).model_dump_json()


@service(dynamo={"enabled": True, "namespace": "benchmarks"})
class MainWorker:
    remote_worker = depends(RemoteWorker)

    def __init__(self) -> None:
        logger.info("Starting Main Worker...")

    @dynamo_endpoint()
    async def generate(self, request: RequestType):
        logger.info(f"Main Worker received: {request.text}")

        # remote_request = RequestType(text="1 2 3 4 5 6 7 8 9")
        remote_responses = self.remote_worker.generate(request.model_dump_json())
        async for response in remote_responses:
            # logger.info(f"Main Worker received response: {response}")
            yield ResponseType(text=response).model_dump_json()
