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


import json
import logging
import signal

import requests
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server

from dynamo.sdk import dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class SglangWorker:
    def __init__(self):
        print("Initializing...")
        self.server_process, self.port = launch_server_cmd(
            """
            python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
            --host 0.0.0.0
            """
        )
        wait_for_server(f"http://localhost:{self.port}")

        print("Server started...")

        signal.signal(signal.SIGTERM, self.shutdown_server)
        signal.signal(signal.SIGINT, self.shutdown_server)

    def shutdown_server(self, signum, frame):
        terminate_process(self.server_process)

    async def generate(self, request: dict):
        print("Generating...")
        print(request)
        response = requests.post(
            f"http://localhost:{self.port}/v1/chat/completions",
            json=request,
            stream=True,
        )

        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                yield data

    @dynamo_endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest):
        print("Chat completions...")
        print(raw_request)
        async for response in self.generate(raw_request.model_dump()):
            yield response
