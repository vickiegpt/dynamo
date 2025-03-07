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
import json
import os
import signal

import uvloop
from common.base_engine import BaseTensorrtLLMEngine
from common.disagg_processor import ChatProcessor, parse_chat_message_content
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.processor import merge_promises
from common.protocol import (
    DisaggChatCompletionRequest,
    DisaggChatCompletionStreamResponse,
    DisaggCompletionStreamResponse,
    DisaggregatedTypeConverter,
)
from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD
from tensorrt_llm._utils import set_mpi_comm
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi import MpiCommSession
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    parse_disagg_config_file,
    split_world_comm,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import CompletionRequest

from triton_distributed.llm import KvMetricsPublisher
from triton_distributed.runtime import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)

import traceback
import threading

logger.set_level("debug")


class WorkerId:
    def __init__(self, completions_lease_id: str, chat_lease_id: str):
        self.completions_lease_id = completions_lease_id
        self.chat_lease_id = chat_lease_id

    def id(self):
        return "%s^%s" % (self.completions_lease_id, self.chat_lease_id)


def update_args_from_disagg_config(
    engine_config: LLMAPIConfig, server_config: CtxGenServerConfig
):
    # Overwrite the LLM API config with the disaggregated config
    # Allows for different configs for context and generation servers
    engine_config.extra_args.update(**server_config.other_args)
    engine_config.update_sub_configs(server_config.other_args)
    return engine_config


class TensorrtLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(
        self,
        engine_config: LLMAPIConfig,
        disagg_config: DisaggServerConfig,
        instance_idx: int,
        sub_comm,
        metrics_publisher: KvMetricsPublisher,
        worker_id: WorkerId,
    ):
        self.disagg_config = disagg_config
        self.instance_idx = instance_idx
        self.server_config: CtxGenServerConfig = disagg_config.server_configs[
            instance_idx
        ]
        engine_config = update_args_from_disagg_config(
            engine_config, self.server_config
        )

        # needed for disagg
        self._mpi_session = MpiCommSession(sub_comm, n_workers=sub_comm.Get_size())
        engine_config.extra_args["_mpi_session"] = self._mpi_session
        super().__init__(engine_config)

        self.metrics_publisher = metrics_publisher

        self.request_active_slots = 0
        self.request_total_slots = 4
        self.kv_active_block = 0
        self.kv_total_blocks = 4

        self.worker_id = worker_id

        self._publishing_cv = threading.Condition()
        self._token_generated = False

        if self.metrics_publisher is not None:
            # [NOTE] Now that the component must has proper metrics reported
            # to be properly selected by the router
            self.metrics_publisher.publish(
                self.request_active_slots,
                self.request_total_slots,
                self.kv_active_block,
                self.kv_total_blocks,
            )

            self._init_publishing_loop()
            

    def _init_publishing_loop(self):
        self._publishing_thread = threading.Thread(
            target=asyncio.run, args=(self._publishing_loop(),)
        )
        self._publishing_thread.start()
    
    async def _publishing_loop(self):

        # Wait for the first token to be generated to start query
        # metrics and kv cache events from the engine.
        with self._publishing_cv:
            while not self._token_generated:
                self._publishing_cv.wait()

        logger.debug("TTanmay:: Start publishing loop")
        try:
            #stats = self._llm_engine.get_stats(timeout=5)
            stats = "dummy"
            logger.info(f"TTanmay:: Stats: {stats}")
            #async for stat in stats:
            #    logger.info(f"TTanmay:: Stats: {stat}")
        except Exception as e:
            logger.error(f"TTanmay:: Error in publishing loop: {traceback.format_exc()}")



    @triton_endpoint(DisaggChatCompletionRequest, DisaggChatCompletionStreamResponse)
    async def generate_chat(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        logger.debug(f"Received request: {request}")
        chat_processor = ChatProcessor(self._model, self._tokenizer, request)

        self._ongoing_request_count += 1

        try:
            conversation = []
            for message in request.messages:
                conversation.extend(parse_chat_message_content(message))
            tool_dicts = (
                None
                if request.tools is None
                else [tool.model_dump() for tool in request.tools]
            )
            prompt: str = self._tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                **(request.chat_template_kwargs or {}),
            )
            sampling_params = request.to_sampling_params()
            disaggregated_params = (
                DisaggregatedTypeConverter.to_llm_disaggregated_params(
                    request.disaggregated_params
                )
            )

            final_result = None
            async for result in self._llm_engine.generate_async(
                prompt,
                sampling_params,
                streaming=request.stream,
                disaggregated_params=disaggregated_params,
            ):
                self.generate_event.set()
                final_result = result
                logger.debug(f"Generated result: {result}")
                if self.server_config.type == "ctx":
                    disaggregated_response = chat_processor.get_chat_stream_response(
                        request.id,
                        result,
                        first_iteration=True,
                    )
                    disaggregated_response.disaggregated_params = (
                        DisaggregatedTypeConverter.to_oai_disaggregated_params(
                            result.outputs[0].disaggregated_params
                        )
                    )
                    if self.metrics_publisher is not None:
                        # TODO:Tanmay: Asynchronously publish KV events for ctx server
                        # and publish metrics.
                        pass
                    yield disaggregated_response.model_dump_json()
                else:
                    yield chat_processor.get_chat_stream_response(
                        request.id,
                        result,
                        first_iteration=False,
                    ).model_dump_json(
                        exclude_unset=True, exclude={"disaggregated_params"}
                    )

            if request.stream_options and request.stream_options.include_usage:
                yield chat_processor.create_final_stream_response(
                    request.id,
                    final_result,
                ).model_dump_json(exclude_unset=True, exclude={"disaggregated_params"})

        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))

    @triton_endpoint(CompletionRequest, DisaggCompletionStreamResponse)
    async def generate_completions(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        self._ongoing_request_count += 1
        logger.debug(f"[worker] Received completions request: {request}")

        if not isinstance(request.prompt, str):
            # Check if it's a list and contains integers
            if isinstance(request.prompt, list) and len(request.prompt) == 1:
                request.prompt = request.prompt[0]
            elif not isinstance(request.prompt, list) or not all(
                isinstance(x, int) for x in request.prompt
            ):
                raise ValueError(
                    "Disaggregated server currently only supports single string prompt or list of integers in request"
                )

        sampling_params = request.to_sampling_params()
        llm_disaggregated_params = (
            DisaggregatedTypeConverter.to_llm_disaggregated_params(
                request.disaggregated_params
            )
        )

        # only 1 prompt is supported for now
        promise = self._llm_engine.generate_async(
            request.prompt,
            sampling_params,
            streaming=request.stream,
            disaggregated_params=llm_disaggregated_params,
        )
        generator = merge_promises([promise])
        num_choices = 1 if request.n is None else request.n
        if request.stream:
            response_generator = self.completions_processor.create_completion_generator(
                request, generator, num_choices
            )
            async for response in response_generator:
                yield json.loads(response)
        else:
            raise RuntimeError("Non-streaming is not supported")
        
        if self.metrics_publisher is not None:
            with self._publishing_cv:
                if not self._token_generated:
                    self._token_generated = True
                    self._publishing_cv.notify_all()
            try:
                logger.info("DTTanmay:: Testing testing")
                s = self._llm_engine.get_kv_cache_events_async(timeout=5)
                logger.info(f"DTTanmay:: Stats expect returned: {s}")
                async for stat in s:
                    logger.info(f"DTTanmay:: Stats expect generated: {stat}")
            except Exception:
                logger.error(f"DTTanmay:: Error in publishing loop: {traceback.format_exc()}")

        self._ongoing_request_count -= 1

    async def run_metrics_loop(self):
        stats = self._llm_engine.get_stats_async(timeout=5)
        async for stat in stats:
            logger.info(f"Tanmay:: Stats: {stat}")

    async def run_kv_cache_event_loop(self):
        stats = self._llm_engine.get_kv_cache_events_async(timeout=5)
        async for stat in stats:
            logger.info(f"Tanmay:: Events: {stat}")


@triton_worker()
async def worker(
    runtime: DistributedRuntime,
    engine_config: LLMAPIConfig,
    disagg_config: DisaggServerConfig,
    instance_idx: int,
    sub_comm,
    publish_kv_events: bool,
):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    server_type = disagg_config.server_configs[instance_idx].type
    logger.info(f"Starting {server_type} server")

    component = runtime.namespace("triton-init").component(
        f"tensorrt-llm-{server_type}"
    )
    await component.create_service()

    completions_endpoint = component.endpoint("completions")
    chat_endpoint = component.endpoint("chat/completions")

    # Only publish events for ctx server.
    metrics_publisher = None
    if publish_kv_events:
        if server_type == "ctx":
            metrics_publisher = KvMetricsPublisher()
        else:
            logger.warning("KV events can only be published for ctx server")

    worker_id = WorkerId(completions_endpoint.lease_id(), chat_endpoint.lease_id()).id()

    logger.debug(f"Tanmay:: Worker ID: {worker_id}")

    engine = TensorrtLLMEngine(
        engine_config,
        disagg_config,
        instance_idx,
        sub_comm,
        metrics_publisher,
        worker_id,
    )

    coros = [
        completions_endpoint.serve_endpoint(engine.generate_completions),
        chat_endpoint.serve_endpoint(engine.generate_chat),
    ]

    await asyncio.gather(*coros)


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()

    if args.llmapi_disaggregated_config is None or not os.path.exists(
        args.llmapi_disaggregated_config
    ):
        raise ValueError(
            "llmapi_disaggregated_config file does not exist or not provided"
        )

    disagg_config: DisaggServerConfig = parse_disagg_config_file(
        args.llmapi_disaggregated_config
    )

    logger.info(f"Parsed disaggregated config: {disagg_config}")

    is_leader, instance_idx, sub_comm = split_world_comm(disagg_config.server_configs)
    os.environ["TRTLLM_USE_MPI_KVCACHE"] = "1"
    set_mpi_comm(sub_comm)

    logger.info(f"is_leader: {is_leader}, instance_idx: {instance_idx}")

    if is_leader:
        asyncio.run(
            worker(
                engine_config,
                disagg_config,
                instance_idx,
                sub_comm,
                args.publish_kv_events,
            )
        )
    else:
        with MPICommExecutor(sub_comm) as executor:
            if not is_leader and executor is not None:
                raise RuntimeError(f"rank{COMM_WORLD} should not have executor")

