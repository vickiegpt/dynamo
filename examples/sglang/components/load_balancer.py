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
import random

from components.worker import (
    SglangDecodeWorker,
    SglangPrefillDecodeWorker,
    SglangPrefillWorker,
)
from sglang.srt.openai_api.protocol import ChatCompletionRequest

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class SimpleLoadBalancer:
    prefill_decode_worker = depends(SglangPrefillDecodeWorker)
    decode_worker = depends(SglangDecodeWorker)
    prefill_worker = depends(SglangPrefillWorker)

    def __init__(self):
        load_balancer_config = ServiceConfig.get_instance().get(
            "SimpleLoadBalancer", {}
        )
        self.disaggregation_enabled = load_balancer_config.get(
            "disaggregation_enabled", False
        )

        print(f"Disaggregation enabled: {self.disaggregation_enabled}")
        print(f"Prefill decode worker: {self.prefill_decode_worker}")
        print(f"Decode worker: {self.decode_worker}")
        print(f"Prefill worker: {self.prefill_worker}")

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]

        if not self.disaggregation_enabled:
            self.prefill_decode_client = (
                await runtime.namespace("dynamo")
                .component("router")
                .endpoint("generate")
                .client()
            )

        if self.disaggregation_enabled:
            self.decode_client = (
                await runtime.namespace("dynamo")
                .component("router")
                .endpoint("generate")
                .client()
            )
            self.prefill_client = (
                await runtime.namespace("dynamo")
                .component("router")
                .endpoint("generate")
                .client()
            )
            self.prefill_get_url_client = (
                await runtime.namespace("dynamo")
                .component("router")
                .endpoint("get_url")
                .client()
            )

        print("Clients initialized")

    @dynamo_endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest):
        print("Chat completions...")
        print(raw_request)

        if self.disaggregation_enabled:
            async for response in self._run_disaggregated(raw_request):
                yield response
        else:
            async for response in self._run_aggregated(raw_request):
                yield response

    async def _run_aggregated(self, raw_request: ChatCompletionRequest):
        async for response in self.prefill_decode_worker.generate(
            raw_request.model_dump()
        ):
            yield response

    async def _run_disaggregated(self, raw_request: ChatCompletionRequest):
        raise NotImplementedError("Disaggregated mode not implemented")
        # request_data = raw_request.model_dump()
        # modified_request = request_data.copy()
        # modified_request.update(
        #     {
        #         "bootstrap_host": hostname,
        #         "bootstrap_port": bootstrap_port,
        #         "bootstrap_room": self._generate_bootstrap_room(),
        #     }
        # )

        # self.prefill_worker.generate(modified_request)
        # async for response in self.decode_worker.generate(modified_request):
        #     yield response

    def _generate_bootstrap_room(self):
        return random.randint(0, 2**63 - 1)
