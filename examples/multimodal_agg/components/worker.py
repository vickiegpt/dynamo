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

from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import asyncio
import logging
import os
import signal
import torch

from components.disagg_router import PyDisaggregatedRouter
from components.prefill_worker import PrefillWorker
from components.encode_worker import EncodeWorker
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import MyRequestOutput, vLLMGenerateRequest, vLLMMultimodalRequest
from utils.protocol import EncodeRequest, EncodeResponse
from utils.logging import check_required_workers

from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest
from vllm.sampling_params import RequestOutputKind

from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.worker import Worker
from vllm.inputs.data import TokensPrompt


from dynamo.llm import KvMetricsPublisher
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmWorker:
    prefill_worker = depends(PrefillWorker)
    encode_worker = depends(EncodeWorker)

    def __init__(self):
        self.client = None
        self.min_workers = 1
        self.disaggregated_router: PyDisaggregatedRouter = None  # type: ignore
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        # Set "trust_remote_code" to True to allow loading models with custom tokens
        # TODO: Add documentation for this
        # self.engine_args.trust_remote_code = True
        self.do_remote_prefill = self.engine_args.remote_prefill
        self.model_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = self.model_name
        logger.info(
            f"Prefill queue: {self._prefill_queue_nats_server}:{self._prefill_queue_stream_name}"
        )

        if self.engine_args.remote_prefill:
            if self.engine_args.enable_chunked_prefill is not False:
                logger.info("Chunked prefill is not supported yet, setting to False")
                self.engine_args.enable_chunked_prefill = False

            if self.engine_args.preemption_mode != "swap":
                logger.info("Preemption mode is not supported yet, setting to swap")
                self.engine_args.preemption_mode = "swap"

            if self.engine_args.pipeline_parallel_size != 1:
                logger.info("Pipeline parallel size is not supported yet, setting to 1")
                self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.router == "kv":
            if not self.engine_args.enable_prefix_caching:
                logger.info(
                    "When using KV router, prefix caching must be enabled, setting to True"
                )
                self.engine_args.enable_prefix_caching = True

            VLLM_WORKER_ID = dynamo_context["endpoints"][0].lease_id()
            os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
            os.environ["VLLM_KV_NAMESPACE"] = "dynamo"
            os.environ["VLLM_KV_COMPONENT"] = class_name
            logger.info(f"Generate endpoint ID: {VLLM_WORKER_ID}")
        self.metrics_publisher = KvMetricsPublisher()

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")
        if self.engine_args.router == "kv":
            assert self.engine_client is not None, "engine_client was not initialized"
            self.engine_client.set_metrics_publisher(self.metrics_publisher)
            # Initially send dummy metrics to kick start,
            # vLLM will not update stat until forward pass is triggered
            self.metrics_publisher.publish(
                0,  # request_active_slots
                1024,  # request_total_slots
                0,  # kv_active_blocks
                1024,  # kv_total_blocks
                0,  # num_requests_waiting
                0.0,  # gpu_cache_usage_perc
                0.0,  # gpu_prefix_cache_hit_rate
            )
            task = asyncio.create_task(self.create_metrics_publisher_endpoint())
            task.add_done_callback(
                lambda _: logger.info("metrics publisher endpoint created")
            )

        runtime = dynamo_context["runtime"]

        enc_comp_ns, enc_comp_name = EncodeWorker.dynamo_address()  # type: ignore
        self.encode_worker_client = (
            await runtime.namespace(enc_comp_ns)
            .component(enc_comp_name)
            .endpoint("encode")
            .client()
        )

        await check_required_workers(self.encode_worker_client, self.min_workers)

        self.disaggregated_router = None
        logger.info("VllmWorker has been initialized")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("VllmWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    async def create_metrics_publisher_endpoint(self):
        component = dynamo_context["component"]
        await self.metrics_publisher.create_endpoint(component)

    @dynamo_endpoint()
    async def generate(self, request: vLLMGenerateRequest):
    # async def generate(self, request: vLLMMultimodalRequest):
        # Get embedding for image
        # image_url = raw_request.image_url
        # Use a dummy image url for now
        # image_url = "http://images.cocodataset.org/test2017/000000155781.jpg"
        image_url = "/workspace/examples/e_agg/coffee_2.JPG"

        print(f"worker generate request: {request.model_dump_json()}")

        encode_generator = await self.encode_worker_client.round_robin(
            EncodeRequest(
                request_id=request.request_id,
                image_url=image_url,
            ).model_dump_json()
        )
        print("after encode_generator")
        async for encode_response in encode_generator:
            # print(f"Encode response: {encode_response}")
            encode_output = EncodeResponse.model_validate_json(encode_response.data())
            # print("after encode_output")
            # image_features = encode_output.image_features
            image_features = torch.tensor(encode_output.image_features, device="cpu", dtype=torch.float16)

        print(f"Image features: {image_features}")

        remote_prefill_params = None
        logger.info(
            f"Prefilling locally for request {request.request_id} with length {len(request.engine_prompt['prompt_token_ids'])}"
        )
        async for response in self.engine_client.generate(
            prompt=TokensPrompt(prompt_token_ids=request.engine_prompt["prompt_token_ids"], multi_modal_data={"image": image_features}),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
            remote_prefill_params=remote_prefill_params,
        ):
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            ).model_dump_json()
