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
from argparse import Namespace
from enum import Enum
from typing import AsyncIterator

import uvloop
from common.protocol import Tokens
from vllm.logger import logger as vllm_logger

from dynemo.llm import KvIndexer, KvMetricsAggregator, KvRouter
from dynemo.runtime import DistributedRuntime, dynemo_endpoint, dynemo_worker

WorkerId = str


class RoutingStrategy(Enum):
    PREFIX = "prefix"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


class Router:
    """
    Request handler for the generate endpoint
    """

    def __init__(
        self,
        router: KvRouter,
        routing_strategy: RoutingStrategy = RoutingStrategy.PREFIX,
    ):
        vllm_logger.info(
            f"Initializing KV Router with strategy: {routing_strategy.value}"
        )
        self.router = router
        self.routing_strategy = routing_strategy

    @dynemo_endpoint(Tokens, WorkerId)
    async def generate(self, request) -> AsyncIterator[WorkerId]:
        lora_id = 0
        worker_id = None
        if self.routing_strategy == RoutingStrategy.PREFIX:
            try:
                worker_id = await self.router.schedule(request.tokens, lora_id)
            # [NOTE][TODO] Now that the scheduler may return more error messages,
            # now we are catching all exceptions and logging them. Should have
            # catch specific router exceptions once we have dedicated types.
            except Exception as e:
                vllm_logger.info(f"{e}")
                worker_id = ""
                vllm_logger.exception(f"Error during worker selection: {e}")

            vllm_logger.info(f"Scheduling to worker_id: {worker_id}")

            yield str(worker_id)

        else:
            # TODO: Do we implement round_robin and random here?
            # or just skip this router and directly enable in preprocess?
            raise NotImplementedError(
                f"Routing strategy {self.routing_strategy} not implemented"
            )

def normalize_values(values, transform=None):
    """Normalize values to a 0-1 range with optional transformation after normalization
    
    Args:
        values: List of values to normalize
        transform: Optional function to apply to normalized values
        
    Returns:
        List of normalized values with optional transformation applied
    """
    max_value = max(values) if values and max(values) > 0 else 1
    normalized = [value / max_value for value in values]
    
    if transform:
        normalized = [transform(value) for value in normalized]
    
    return normalized


class CustomRouter:
    """
    Request handler for the generate endpoint
    """

    def __init__(
        self,
        indexer: KvIndexer,
        metrics_aggregator: KvMetricsAggregator,
    ):
        vllm_logger.info("Initializing Custom Router")
        self.indexer = indexer
        self.metrics_aggregator = metrics_aggregator

    def _cost_function(self, scores, metrics):
        # Get worker IDs and their corresponding scores
        worker_ids = list(scores.scores.keys())
        
        # Normalize scores (higher is better)
        max_score = max(scores.scores.values()) if scores.scores else 1.0
        normalized_scores = [
            score / max_score if max_score > 0 else 0.0
            for _, score in scores.scores.items()
        ]

        # Initialize lists to store metrics
        kv_usages = []
        waiting_requests = []
        cache_metrics = []

        # Collect metrics in a single loop
        for endpoint in metrics.endpoints:
            if endpoint.worker_id in worker_ids:
                kv_usages.append(endpoint.gpu_cache_usage_perc if hasattr(endpoint, 'gpu_cache_usage_perc') else 0.0)
                waiting_requests.append(endpoint.num_requests_waiting if hasattr(endpoint, 'num_requests_waiting') else 0.0)
                cache_metrics.append(endpoint.gpu_prefix_cache_hit_rate if hasattr(endpoint, 'gpu_prefix_cache_hit_rate') else 0.0)

        normalized_waiting_requests = normalize_values(waiting_requests) #, transform=lambda x: x**2

        # Calculate logits using the formula:
        # 2 * overlap - 4 * cache_len - kv - waiting
        worker_logits = {
            worker_id: (
                2.0 * overlap - 
                4.0 * cache_len -  # seems a bit high, lets see
                kv - 
                waiting
            )
            for worker_id, overlap, kv, waiting, cache_len in zip(
                worker_ids, normalized_scores, kv_usages, normalized_waiting_requests, cache_metrics
            )
        }
        
        # Select the worker with the highest logit
        best_worker_id = max(worker_logits.items(), key=lambda x: x[1])[0] if worker_logits else ""
        
        # Log the metrics for the selected worker
        if best_worker_id:
            vllm_logger.info(f"Selected worker: {best_worker_id}, logit: {worker_logits[best_worker_id]:.4f}")
            vllm_logger.info(f"Score: {scores.scores.get(best_worker_id, 0.0):.4f}")
            for endpoint in metrics.endpoints:
                if endpoint.worker_id == best_worker_id:
                    # Log all relevant metrics
                    vllm_logger.info(f"Overlap score: {normalized_scores[worker_ids.index(best_worker_id)]:.4f}")
                    
                    if hasattr(endpoint, 'num_requests_waiting'):
                        vllm_logger.info(f"Waiting requests: {endpoint.num_requests_waiting}")
                    
                    if hasattr(endpoint, 'gpu_cache_usage_perc'):
                        vllm_logger.info(f"GPU cache usage: {endpoint.gpu_cache_usage_perc:.2f}%")
                    
                    if hasattr(endpoint, 'gpu_prefix_cache_hit_rate'):
                        vllm_logger.info(f"GPU prefix cache hit rate: {endpoint.gpu_prefix_cache_hit_rate:.2f}")
                    
                    # Also log the original metrics for reference
                    vllm_logger.info(f"Request slot usage: {endpoint.request_active_slots} / {endpoint.request_total_slots}")
                    vllm_logger.info(f"KV block usage: {endpoint.kv_active_blocks} / {endpoint.kv_total_blocks}")
        
        return best_worker_id

    @dynemo_endpoint(Tokens, WorkerId)
    async def generate(self, request) -> AsyncIterator[WorkerId]:
        lora_id = 0
        worker_id = ""
        try:
            scores = await self.indexer.find_matches_for_request(
                request.tokens, lora_id
            )
            metrics = await self.metrics_aggregator.get_metrics()
            worker_id = self._cost_function(scores, metrics)

        # [NOTE][TODO] Now that the scheduler may return more error messages,
        # now we are catching all exceptions and logging them. Should have
        # catch specific router exceptions once we have dedicated types.
        except Exception as e:
            vllm_logger.info(f"{e}")
            worker_id = ""
            vllm_logger.exception(f"Error during worker selection: {e}")

        vllm_logger.info(f"Scheduling to worker_id: {worker_id}")

        yield str(worker_id)


@dynemo_worker()
async def worker(runtime: DistributedRuntime, args: Namespace):
    """
    Set up the worker clients.
    Serve the dynemo.router.generate endpoint.
    """
    workers_client = (
        await runtime.namespace("dynemo")
        .component("vllm")
        .endpoint("generate")
        .client()
    )

    while len(workers_client.endpoint_ids()) < args.min_workers:
        vllm_logger.info(
            f"Waiting for more workers... Current: {len(workers_client.endpoint_ids())}, Required: {args.min_workers}"
        )
        await asyncio.sleep(5)

    vllm_logger.info(
        f"Required number of workers ({args.min_workers}) are ready:\n"
        + "\n".join(f"id: {id}" for id in workers_client.endpoint_ids())
    )

    kv_listener = runtime.namespace("dynemo").component("vllm")
    await kv_listener.create_service()

    router_component = runtime.namespace("dynemo").component("router")
    await router_component.create_service()

    endpoint = router_component.endpoint("generate")

    if args.custom_router:
        indexer = KvIndexer(kv_listener)
        metrics_aggregator = KvMetricsAggregator(kv_listener)
        await endpoint.serve_endpoint(
            CustomRouter(indexer, metrics_aggregator).generate
        )
    else:
        router = KvRouter(runtime, kv_listener)
        await endpoint.serve_endpoint(Router(router, args.routing_strategy).generate)


if __name__ == "__main__":
    uvloop.install()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routing-strategy",
        type=RoutingStrategy,
        default=RoutingStrategy.PREFIX,
        choices=list(RoutingStrategy),
        help="Routing strategy to use",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum number of workers required before proceeding",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model that is being served",
    )
    parser.add_argument(
        "--custom-router",
        action="store_true",
        help="Whether to use custom router or not",
    )
    args = parser.parse_args()

    asyncio.run(worker(args))
