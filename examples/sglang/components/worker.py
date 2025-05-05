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
import socket

import requests
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server

from dynamo.sdk import dynamo_endpoint, service

logger = logging.getLogger(__name__)


class SglangBaseWorker:
    def __init__(self, additional_args: str = ""):
        print("Initializing...")
        self.hostname = socket.gethostname()
        self.server_process, self.port = launch_server_cmd(
            f"""
            python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
            --host 0.0.0.0 {additional_args}
            """
        )
        wait_for_server(f"http://localhost:{self.port}")

        print("Server started...")

        signal.signal(signal.SIGTERM, self.shutdown_server)
        signal.signal(signal.SIGINT, self.shutdown_server)

    def shutdown_server(self, signum, frame):
        terminate_process(self.server_process)

    @dynamo_endpoint(name="generate")
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

    @dynamo_endpoint(name="get_url")
    async def get_url(self, request: dict):
        return f"http://{self.hostname}:{self.port}"


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class SglangPrefillWorker(SglangBaseWorker):
    def __init__(self):
        super().__init__(
            "--disaggregation-mode prefill --disaggregation-transfer-backend nixl"
        )


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class SglangDecodeWorker(SglangBaseWorker):
    def __init__(self):
        super().__init__(
            "--disaggregation-mode decode --disaggregation-transfer-backend nixl"
        )


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class SglangPrefillDecodeWorker(SglangBaseWorker):
    pass
