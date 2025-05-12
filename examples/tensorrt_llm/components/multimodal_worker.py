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
import logging

import torch
from common.base_engine import BaseTensorrtLLMEngine
from common.parser import parse_tensorrt_llm_args
from common.protocol import EncodeRequest, EncodeResponse, TRTLLMWorkerRequest
from common.utils import ServerType
from components.encode_worker import EncodeWorker
from components.prefill_worker import TensorRTLLMPrefillWorker
from utils.logging import check_required_workers

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class TensorRTLLMWorker(BaseTensorrtLLMEngine):
    prefill_worker = depends(TensorRTLLMPrefillWorker)
    encode_worker = depends(EncodeWorker)

    def __init__(self):
        logger.info("Initializing TensorRT-LLM Worker")
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        config_args = config.as_args(class_name, prefix="")
        args, engine_config = parse_tensorrt_llm_args(config_args)
        worker_id = dynamo_context["endpoints"][0].lease_id()
        self._min_prefill_workers = args.min_prefill_workers
        super().__init__(
            namespace_str="dynamo",
            component_str=class_name,
            worker_id=worker_id,
            engine_config=engine_config,
            remote_prefill=args.remote_prefill,
            min_workers=args.min_workers,
            disagg_config_file=args.llmapi_disaggregated_config,
            block_size=args.block_size,
            router=args.router,
            server_type=ServerType.GEN,
        )

    @async_on_start
    async def async_init(self):
        self._init_engine()

        if self._remote_prefill:
            runtime = dynamo_context["runtime"]
            comp_ns, comp_name = TensorRTLLMPrefillWorker.dynamo_address()  # type: ignore
            self._prefill_client = (
                await runtime.namespace(comp_ns)
                .component(comp_name)
                .endpoint("generate")
                .client()
            )
            while len(self._prefill_client.endpoint_ids()) < self._min_prefill_workers:
                logger.info(
                    f"Waiting for prefill workers to be ready.\n"
                    f" Current: {len(self._prefill_client.endpoint_ids())},"
                    f" Required: {self._min_prefill_workers}"
                )
                await asyncio.sleep(2)

        if self._kv_metrics_publisher is not None:
            task = asyncio.create_task(self.create_metrics_publisher_endpoint())
            task.add_done_callback(
                lambda _: logger.info("metrics publisher endpoint created")
            )

        enc_comp_ns, enc_comp_name = EncodeWorker.dynamo_address()  # type: ignore
        self.encode_worker_client = (
            await runtime.namespace(enc_comp_ns)
            .component(enc_comp_name)
            .endpoint("encode")
            .client()
        )

        await check_required_workers(self.encode_worker_client, self.min_workers)

        self.disaggregated_router = None

        logger.info("TensorRT-LLM Worker initialized")

    async def create_metrics_publisher_endpoint(self):
        component = dynamo_context["component"]
        await self._kv_metrics_publisher.create_endpoint(component)

    @dynamo_endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        image_url = request.image_url

        encode_generator = await self.encode_worker_client.round_robin(
            EncodeRequest(
                image_url=image_url,
            ).model_dump_json()
        )
        async for encode_response in encode_generator:
            encode_output = EncodeResponse.model_validate_json(encode_response.data())
            image_features_tensor = torch.tensor(
                encode_output.image_features, device="cpu", dtype=torch.float16
            )

        # --- Assign features to the request object's dedicated field ---
        # NOTE: Based on TRTLLMWorkerRequest definition, super().generate
        # likely does NOT know about this field. This approach might require
        # either modifying the base class or implementing generation logic directly
        # instead of calling super().generate.
        if image_features_tensor is not None:
            # Convert tensor to list for Pydantic/JSON serialization
            request.image_features = image_features_tensor.tolist()
            logger.info(
                f"Assigned image features (shape: {image_features_tensor.shape}) to request.image_features field."
            )
        else:
            request.image_features = None
            logger.warning("Could not get image features to assign to request.")
        # -------------------------------------------------------------

        logger.info(
            f"Calling super().generate for request {request.request_id} with multimodal data."
        )

        # Call base class. It receives the TRTLLMWorkerRequest object
        # but likely won't use request.image_features unless modified.
        # Defined at common/base_engine.py
        async for response in super().generate(request):
            # Log the type and content of the response from the base engine
            logger.info(
                f"Raw response type from BaseTensorrtLLMEngine: {type(response)}"
            )
            logger.info(f"Raw response content: {response}")

            # Yield the response object directly
            yield response
