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
import os
import signal
import socket
from typing import Optional

from utils.args import parse_vllm_args
from utils.protocol import MyRequestOutput, vLLMGenerateRequest
from vllm.config import VllmConfig
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from dynamo.llm import (
    WorkerMetricsPublisher,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
)
from dynamo.runtime import Component
from dynamo.sdk import async_on_start, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)


class DynamoStatLoggerPublisher(StatLoggerBase):
    """Stat logger publisher. Wrapper for the WorkerMetricsPublisher to match the StatLoggerBase interface."""

    def __init__(self, component: Component, dp_rank: int) -> None:
        self.inner = WorkerMetricsPublisher()
        self.inner.create_endpoint(component, dp_rank=dp_rank)
        self.dp_rank = dp_rank

    def record(
        self, scheduler_stats: SchedulerStats, iteration_stats: Optional[IterationStats]
    ):
        # request_total_slots and kv_total_blocks are properties of model + gpu
        # we should only publish them once, not every metric update
        # they should be part of some runtime metadata tied to MDC or put in etcd ?
        hit_rate = 0
        if scheduler_stats.prefix_cache_stats.queries > 0:
            hit_rate = (
                scheduler_stats.prefix_cache_stats.hits
                / scheduler_stats.prefix_cache_stats.queries
            )

        # TODO Manage DP Ranks in metrics aggregation.
        self.inner.publish(
            request_active_slots=scheduler_stats.num_running_reqs,
            request_total_slots=0,  # TODO - remove from metrics
            kv_active_blocks=0,  # TODO - need to calculate this
            kv_total_blocks=0,  # TODO - remove from metrics
            num_requests_waiting=scheduler_stats.num_waiting_reqs,  # used in current cost function
            gpu_cache_usage_perc=scheduler_stats.gpu_cache_usage,  # used in current cost function
            gpu_prefix_cache_hit_rate=hit_rate,
            data_parallel_rank=self.dp_rank,
        )

    def log_engine_initialized(self) -> None:
        pass


class StatLoggerFactory:
    """Factory for creating stat logger publishers. Required by vLLM."""

    def __init__(self, component: Component) -> None:
        self.component = component

    def create_stat_logger(self, dp_rank: int) -> StatLoggerBase:
        return DynamoStatLoggerPublisher(self.component, dp_rank)

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return self.create_stat_logger(dp_rank=dp_rank)


class VllmBaseWorker:
    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.kv_publishers = []

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

        self.set_side_channel_port()

    async def async_init(self):
        # Taken from build_async_engine_client_from_engine_args()
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = self.engine_args.create_engine_config(usage_context=usage_context)

        # Explicitly pass our custom stat logger for metrics
        self.engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=[StatLoggerFactory(dynamo_context["component"])],
            disable_log_requests=self.engine_args.disable_log_requests,
            disable_log_stats=self.engine_args.disable_log_stats,
        )

        logger.info("VllmWorker has been initialized")

        base_zmq_endpoint = "tcp://127.0.0.1:5557"
        dp_rank_size = vllm_config.parallel_config.data_parallel_size

        # Store references to prevent garbage collection

        for dp_rank in range(dp_rank_size):
            zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
                base_zmq_endpoint, data_parallel_rank=dp_rank
            )
            zmq_config = ZmqKvEventPublisherConfig(
                worker_id=dynamo_context["endpoints"][0].lease_id(),
                kv_block_size=self.engine_args.block_size,
                zmq_endpoint=zmq_endpoint,
            )

            try:
                publisher = ZmqKvEventPublisher(
                    component=dynamo_context["component"], config=zmq_config
                )
                self.kv_publishers.append(publisher)
            except Exception as e:
                logger.error(
                    f"Failed to create ZmqKvEventPublisher for dp_rank {dp_rank}: {e}"
                )

        logger.debug(
            f"Successfully created {len(self.kv_publishers)} ZmqKvEventPublishers out of {dp_rank_size} expected"
        )

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            for publisher in self.kv_publishers:
                publisher.shutdown()
            logger.info("VllmWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    @endpoint()
    async def generate(self, request: vLLMGenerateRequest):
        gen = self.engine_client.generate(
            prompt=request.prompt,
            sampling_params=request.sampling_params,
            request_id=request.request_id,
        )

        async for response in gen:
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
                metrics=response.metrics,
                kv_transfer_params=response.kv_transfer_params,
            ).model_dump_json()

    def set_side_channel_port(self, port: Optional[int] = None):
        """vLLM V1 NixlConnector creates a side channel to exchange metadata with other NIXL connectors.
        This sets the port number for the side channel.
        """
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))  # Bind to a free port provided by the host.
                port = s.getsockname()[1]  # Get the port number assigned.
        logger.debug("Setting VLLM_NIXL_SIDE_CHANNEL_PORT to %s", port)
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(port)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmPrefillWorker(VllmBaseWorker):
    @async_on_start
    async def async_init(self):
        await super().async_init()
        logger.info("VllmPrefillWorker has been initialized")


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmDecodeWorker(VllmBaseWorker):
    @async_on_start
    async def async_init(self):
        await super().async_init()
        logger.info("VllmDecodeWorker has been initialized")
