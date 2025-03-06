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
import random
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
        workers_client,
        indexer: KvIndexer,
        metrics_aggregator: KvMetricsAggregator,
    ):
        vllm_logger.info("Initializing Custom Router")
        self.indexer = indexer
        self.metrics_aggregator = metrics_aggregator
        self.workers_client = workers_client

    def _cost_function(self, scores, metrics):
        # Normalize scores (higher is better) into a dict
        max_score = max(scores.scores.values()) if scores.scores else 1.0
        worker_scores = {
            worker_id: (score / max_score if max_score > 0 else 0.0)
            for worker_id, score in scores.scores.items()
        }

        # Initialize metrics dictionary
        worker_metrics = {}
        for endpoint in metrics.endpoints:
            worker_id = endpoint.worker_id
            worker_metrics[worker_id] = {
                'gpu_cache_usage_perc': endpoint.gpu_cache_usage_perc if hasattr(endpoint, 'gpu_cache_usage_perc') else 0.0,
                'num_requests_waiting': endpoint.num_requests_waiting if hasattr(endpoint, 'num_requests_waiting') else 0.0,
                'gpu_prefix_cache_hit_rate': endpoint.gpu_prefix_cache_hit_rate if hasattr(endpoint, 'gpu_prefix_cache_hit_rate') else 0.0
            }

        # Get all worker IDs from the client
        worker_ids = self.workers_client.endpoint_ids()

        max_waiting = max([worker_metrics.get(worker_id, {}).get('num_requests_waiting', 0.0) for worker_id in worker_ids], default=1.0)

        
        # Calculate logits for each worker
        worker_logits = {}
        for worker_id in worker_ids:
            # Use default values if worker not in scores or metrics
            score = worker_scores.get(worker_id, 0.0)
            metrics_dict = worker_metrics.get(worker_id, {
                'gpu_cache_usage_perc': 0.0,
                'num_requests_waiting': 0.0,
                'gpu_prefix_cache_hit_rate': 0.0
            })

            normalized_waiting = metrics_dict['num_requests_waiting'] / max_waiting if max_waiting > 0 else 0.0

            # Calculate logit using the metrics
            worker_logits[worker_id] = (
                2.0 * score - 
                metrics_dict['gpu_cache_usage_perc'] - 
                normalized_waiting -
                2 * metrics_dict['gpu_prefix_cache_hit_rate'] * normalized_waiting
            )

            # Print out the formula for each worker
            vllm_logger.info(f"Formula for {worker_id}: {worker_logits[worker_id]:.4f} = 2.0 * {score:.4f} - 4.0 * {metrics_dict['gpu_prefix_cache_hit_rate']:.4f} - {metrics_dict['gpu_cache_usage_perc']:.4f} - {normalized_waiting:.4f}")

        
        if not worker_logits or all(logit == 0 for logit in worker_logits.values()):
            return ""
            
        vllm_logger.info(f"Logit Length: {len(worker_logits)}")
        
        # Select the worker with the highest logit
        if worker_logits:
            max_logit = max(worker_logits.values())
            # Get all workers that have the max logit value
            best_workers = [wid for wid, logit in worker_logits.items() if logit == max_logit]
            # Randomly select one of the workers with max logit
            best_worker_id = random.choice(best_workers)
        else:
            best_worker_id = ""
        
        # Log the metrics for the selected worker
        if best_worker_id:
            vllm_logger.info(f"Selected worker: {best_worker_id}, logit: {worker_logits[best_worker_id]:.4f}")
            vllm_logger.info(f"Score: {scores.scores.get(best_worker_id, 0.0):.4f}")
            
            metrics_dict = worker_metrics.get(best_worker_id, {})
            vllm_logger.info(f"overlap: {worker_scores.get(best_worker_id, 0.0):.4f}")
            vllm_logger.info(f"cache_len: {metrics_dict.get('gpu_prefix_cache_hit_rate', 0.0):.4f}")
            vllm_logger.info(f"kv: {metrics_dict.get('gpu_cache_usage_perc', 0.0):.4f}")
            vllm_logger.info(f"waiting: {metrics_dict.get('num_requests_waiting', 0.0) / max_waiting if max_waiting > 0 else 0.0:.4f}")
        
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
        vllm_logger.info("########")

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
            CustomRouter(workers_client, indexer, metrics_aggregator).generate
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
