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

from dynamo.sdk import service, endpoint, async_on_start, api, dynamo_context, serve
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

class RequestType(BaseModel):
    text: str


class ResponseType(BaseModel):
    text: str

# Simple user person: Does not need full clarity into runtime
@service(
    dynamo={
        "namespace": "dynamo",
    }
)
class Backend:
    def __init__(self) -> None:
        print("Starting backend")

    @async_on_start
    async def async_init(self):
        print("Async init")

    @endpoint()
    async def generate(self, req: RequestType):
        yield "Hello, world!"

# This whole block is run with `dynamo serve` and just used to get a client to the frontend
@service(
    dynamo={
        "name": "frontend",
        "namespace": "dynamo",
    }
)
class Frontend:
    def __init__(self) -> None:
        print("Starting frontend")
        self.runtime = dynamo_context["runtime"]

    @async_on_start
    async def async_init(self):
        print("Async init")
        # runtime is populated here because this is being called with dynamo serve
        self.client = await self.runtime.namespace("dynamo").component("Backend").endpoint("generate").client()
        print(f"Client: {self.client}")

    @api()
    async def generate(self, req: RequestType):

        async def content_generator():
            async for response in await self.client.generate(req.model_dump_json()):
                yield f"Frontend: {response}"

        return StreamingResponse(content_generator())


if __name__ == "__main__":
    import asyncio
    import uvloop

    asyncio.run(serve(Backend))