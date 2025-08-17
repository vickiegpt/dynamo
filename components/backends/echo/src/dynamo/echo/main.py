# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import logging
import signal

import uvloop

from dynamo.llm import ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def graceful_shutdown(runtime):
    """
    Shutdown dynamo distributed runtime.
    The endpoints will be immediately invalidated so no new requests will be accepted.
    For endpoints served with graceful_shutdown=True, the serving function will wait until all in-flight requests are finished.
    For endpoints served with graceful_shutdown=False, the serving function will return immediately.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


class MockLLM:
    def __init__(self):
        pass

    async def generate(self, request):
        time_per_input_token = 0.1
        time_per_output_token = 0.1

        tokens = request["token_ids"]

        for token in tokens:
            await asyncio.sleep(time_per_input_token)

        # prefill complete

        for token in tokens:
            yield {"token_ids": [token]}
            await asyncio.sleep(time_per_output_token)
        yield {"finish_reason": "stop", "token_ids": []}


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime, prefill_worker=False):
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, functools.partial(graceful_shutdown, runtime))

    logging.info("Signal handlers set up for graceful shutdown")

    handler = MockLLM()

    component = runtime.namespace("dynamo").component("MockLLMWorker")
    await component.create_service()

    generate_endpoint = component.endpoint("generate")

    try:
        await register_llm(
            ModelType.Backend,
            generate_endpoint,
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            kv_cache_block_size=0,
            migration_limit=0,
            runtime_config=ModelRuntimeConfig(),
        )

        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate, graceful_shutdown=True),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        pass


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
