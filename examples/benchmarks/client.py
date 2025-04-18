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

import asyncio

from worker import RequestType

from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker()
async def client_worker(runtime: DistributedRuntime):
    client = (
        await runtime.namespace("benchmarks")
        .component("MainWorker")
        .endpoint("generate")
        .client()
    )
    request = RequestType(text="1 2 3 4 5 6 7 8 9")
    responses = await client.generate(request.model_dump_json())
    async for response in responses:
        print(response)


if __name__ == "__main__":
    asyncio.run(client_worker())
