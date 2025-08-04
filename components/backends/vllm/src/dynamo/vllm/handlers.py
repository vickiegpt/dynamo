# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import AsyncGenerator, Optional, Set

import msgspec
import zmq
import zmq.asyncio

from dynamo.runtime.logging import configure_dynamo_logging
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

from .ports import get_host_ip
from .protocol import MyRequestOutput

configure_dynamo_logging()
logger = logging.getLogger(__name__)


# ZMQ message type definition
class NixlMsg(msgspec.Struct):
    """ZMQ message structure for receiving prefill completion notifications"""

    req_id: str
    status: str = "completed"


class BaseWorkerHandler(ABC):
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    def __init__(self, component, engine, default_sampling_params):
        self.component = component
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publisher = None

    @abstractmethod
    async def generate(self, request) -> AsyncGenerator[dict, None]:
        raise NotImplementedError

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    def cleanup(self):
        """Override in subclasses if cleanup is needed."""
        pass

    async def generate_tokens(self, prompt, sampling_params, request_id):
        gen = self.engine_client.generate(prompt, sampling_params, request_id)

        num_output_tokens_so_far = 0
        async for res in gen:
            # res is vllm's RequestOutput

            # This is the expected way for a request to end.
            # The new token ID will be eos, don't forward it.
            if res.finished:
                yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
            yield out
            num_output_tokens_so_far = next_total_toks


class DecodeWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        component,
        engine,
        default_sampling_params,
        prefill_worker_client=None,
        proxy_host: Optional[str] = None,
        proxy_port: Optional[int] = None,
    ):
        super().__init__(component, engine, default_sampling_params)
        self.prefill_worker_client = prefill_worker_client
        self.can_prefill = 0
        self._prefill_check_task = None

        # Proxy server configuration
        print(f"proxy_host: {proxy_host}, proxy_port: {proxy_port}")
        self.proxy_host = proxy_host or "localhost"
        self.proxy_port = proxy_port or 7500  # default port
        self.finished_reqs: Set[str] = set()
        self._zmq_context = None
        self._proxy_task = None
        self._run_proxy = False

        if self.prefill_worker_client is not None:
            self._prefill_check_task = asyncio.create_task(self._prefill_check_loop())

        # Start proxy server
        if self.proxy_host and self.proxy_port:
            self._start_proxy_server()

    def _start_proxy_server(self):
        """Start ZMQ proxy server"""
        self._zmq_context = zmq.asyncio.Context()
        self._run_proxy = True
        self._proxy_task = asyncio.create_task(self._zmq_pull_server())
        logger.info(f"Starting ZMQ proxy server on {self.proxy_host}:{self.proxy_port}")

    async def _zmq_pull_server(self):
        """ZMQ PULL server to receive prefill completion notifications"""
        socket = self._zmq_context.socket(zmq.PULL)
        proxy_url = f"{self.proxy_host}:{self.proxy_port}"

        try:
            socket.bind(f"tcp://{proxy_url}")
            logger.info(f"ZMQ proxy server started on {proxy_url}")

            while self._run_proxy:
                try:
                    # Set timeout to respond to shutdown
                    msg_bytes = await asyncio.wait_for(socket.recv(), timeout=1.0)
                    msg = msgspec.msgpack.decode(msg_bytes, type=NixlMsg)
                    req_id = msg.req_id
                    self.finished_reqs.add(req_id)
                    logger.info(
                        f"Prefill of req {req_id} done, decoder IP: {self.proxy_host}"
                    )
                except asyncio.TimeoutError:
                    continue  # timeout, continue loop
                except zmq.Again:
                    await asyncio.sleep(0.01)  # Avoid busy loop
                except Exception as e:
                    logger.error(f"ZMQ Error: {e}")
                    break

        except Exception as e:
            logger.error(f"Failed to start ZMQ proxy server: {e}")
        finally:
            socket.close()
            logger.info("ZMQ PULL server stopped.")

    async def _prefill_check_loop(self):
        """Background task that checks prefill worker availability every 5 seconds."""
        while True:
            try:
                if self.prefill_worker_client is not None:
                    self.can_prefill = len(self.prefill_worker_client.instance_ids())
                    logger.debug(f"Current Prefill Workers: {self.can_prefill}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in prefill check loop: {e}")
                await asyncio.sleep(5)  # Still sleep on error to avoid tight loop

    def cleanup(self):
        """Cancel background tasks and cleanup resources."""
        # Stop proxy server
        if self._proxy_task is not None:
            self._run_proxy = False
            self._proxy_task.cancel()

        # Cleanup ZMQ context
        if self._zmq_context is not None:
            self._zmq_context.term()

        # Stop prefill check task
        if self._prefill_check_task is not None:
            self._prefill_check_task.cancel()

        super().cleanup()

    def is_request_finished(self, request_id: str) -> bool:
        """Check if the specified request has completed prefill"""
        return request_id in self.finished_reqs

    def mark_request_processed(self, request_id: str):
        """Mark request as processed, remove from finished_reqs"""
        self.finished_reqs.discard(request_id)

    async def generate(self, request):
        request_id = str(uuid.uuid4().hex)

        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])

        sampling_params = SamplingParams(**self.default_sampling_params)

        sampling_params.detokenize = False
        for key, value in request["sampling_options"].items():
            if value is not None and hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        for key, value in request["stop_conditions"].items():
            if value is not None and hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        if self.can_prefill:
            # Create a copy for prefill with specific modifications
            prefill_sampling_params = deepcopy(sampling_params)
            decode_host_ip = get_host_ip()

            if prefill_sampling_params.extra_args is None:
                prefill_sampling_params.extra_args = {}

            disagg_spec = {
                "req_id": request_id,
                "receiver_host": decode_host_ip,
                "receiver_init_port": 7300,
                "receiver_alloc_port": 7400,
            }

            # req_data["kv_transfer_params"] = {
            #     "ret_first_tok": True,
            #     "disagg_spec": disagg_spec,
            # }
            prefill_sampling_params.extra_args["kv_transfer_params"] = {
                "ret_first_tok": True,
                "disagg_spec": disagg_spec,
            }
            prefill_sampling_params.max_tokens = 1
            prefill_sampling_params.min_tokens = 1

            prefill_request = {
                "token_ids": request["token_ids"],
                "sampling_params": msgspec.to_builtins(prefill_sampling_params),
                "request_id": request_id,
            }

            # TODO Change to prefill queue
            if self.prefill_worker_client is not None:
                prefill_response = await anext(
                    await self.prefill_worker_client.round_robin(prefill_request)
                )
                prefill_response = MyRequestOutput.model_validate_json(
                    prefill_response.data()
                )

                # Modify original sampling_params for decode
                if sampling_params.extra_args is None:
                    sampling_params.extra_args = {}
                sampling_params.extra_args[
                    "kv_transfer_params"
                ] = prefill_response.kv_transfer_params

                logger.info(
                    f"Request {request_id} sent to prefill, decoder IP: {decode_host_ip}"
                )

        async for tok in self.generate_tokens(prompt, sampling_params, request_id):
            yield tok


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(self, component, engine, default_sampling_params):
        super().__init__(component, engine, default_sampling_params)

    async def generate(self, request):
        request_id = request["request_id"]
        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])
        sampling_params = msgspec.convert(request["sampling_params"], SamplingParams)

        gen = self.engine_client.generate(prompt, sampling_params, request_id)

        # Generate only 1 token in prefill
        async for res in gen:
            logger.debug(f"kv transfer params: {res.kv_transfer_params}")
            logger.info(f"kv transfer params: {res.kv_transfer_params}")
            yield MyRequestOutput(
                request_id=res.request_id,
                prompt=res.prompt,
                prompt_token_ids=res.prompt_token_ids,
                prompt_logprobs=res.prompt_logprobs,
                outputs=res.outputs,
                finished=res.finished,
                metrics=res.metrics,
                kv_transfer_params=res.kv_transfer_params,
            ).model_dump_json()
