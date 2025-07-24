# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import signal
import sys
from fastapi import FastAPI
import uvicorn


import uvloop
from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory

from dynamo.llm import (
    ModelType,
    get_tensorrtllm_engine,
    get_tensorrtllm_publisher,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.utils.request_handlers.handlers import (
    RequestHandlerConfig,
    RequestHandlerFactory,
)
from dynamo.trtllm.utils.trtllm_utils import (
    Config,
    cmd_line_args,
    is_first_worker,
    parse_endpoint,
)

# Default buffer size for kv cache events.
DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 1024

configure_dynamo_logging()


async def graceful_shutdown(runtime):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        # Schedule the shutdown coroutine instead of calling it directly
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    config = cmd_line_args()
    await init(runtime, config)


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    logging.info(f"Initializing the worker with config: {config}")

    next_client = None
    if config.next_endpoint:
        logging.info(
            f"Initializing next worker client for endpoint: {config.next_endpoint}"
        )
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            config.next_endpoint
        )
        next_client = (
            await runtime.namespace(parsed_namespace)
            .component(parsed_component_name)
            .endpoint(parsed_endpoint_name)
            .client()
        )

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    # Convert model path to Path object if it's a local path, otherwise keep as string
    model_path = str(config.model_path)

    arg_map = {
        "model": model_path,
        "tensor_parallel_size": config.tensor_parallel_size,
        "backend": "pytorch",
        "skip_tokenizer_init": True,
    }
    if config.extra_engine_args != "":
        # TODO: Support extra engine args from json file as well.
        arg_map = update_llm_args_with_extra_options(arg_map, config.extra_engine_args)
    if config.publish_events_and_metrics:
        # 'event_buffer_max_size' is required to enable TRTLLM to publish kv cache events.
        kv_cache_config = None
        if "kv_cache_config" not in arg_map:
            kv_cache_config = {}
            kv_cache_config["event_buffer_max_size"] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        else:
            kv_cache_config = arg_map["kv_cache_config"]
            if not kv_cache_config.event_buffer_max_size:
                kv_cache_config.event_buffer_max_size = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        arg_map["kv_cache_config"] = kv_cache_config

        # Only pytorch backend is supported for now to publish events and metrics.
        if "backend" not in arg_map:
            arg_map["backend"] = "pytorch"
        elif arg_map["backend"] != "pytorch":
            logging.error(
                "Only pytorch backend is supported for now to publish events and metrics."
            )
            sys.exit(1)

    logging.info(f"TensorRT-LLM engine args: {arg_map}")
    engine_args = arg_map

    # Populate default sampling params from the model
    tokenizer = tokenizer_factory(arg_map["model"])
    default_sampling_params = SamplingParams()
    default_sampling_params._setup(tokenizer)
    default_sampling_params.stop = None

    # We already detokenize inside HandlerBase. No need to also do it in TRTLLM.
    default_sampling_params.detokenize = False

    async with get_tensorrtllm_engine(engine_args) as engine:
        logging.info(f"TensorRT-LLM Debug Drafter 1")
        endpoint = component.endpoint(config.endpoint)

        logging.info(f"TensorRT-LLM Debug Drafter 2")
        # publisher will be set later if publishing is enabled.
        handler_config = RequestHandlerConfig(
            component=component,
            engine=engine,
            default_sampling_params=default_sampling_params,
            publisher=None,
            disaggregation_mode=config.disaggregation_mode,
            disaggregation_strategy=config.disaggregation_strategy,
            next_client=next_client,
        )
        logging.info(f"TensorRT-LLM Debug Drafter 3")
        if config.publish_events_and_metrics and is_first_worker(config):
            logging.info(f"TensorRT-LLM Debug Drafter 4")
            # Initialize and pass in the publisher to the request handler to
            # publish events and metrics.
            kv_listener = runtime.namespace(config.namespace).component(
                config.component
            )
            async with get_tensorrtllm_publisher(
                component,
                engine,
                kv_listener,
                int(endpoint.lease_id()),
                config.kv_block_size,
            ) as publisher:
                handler_config.publisher = publisher
                handler = RequestHandlerFactory().get_request_handler(handler_config)
                await endpoint.serve_endpoint(handler.generate)
        else:
            logging.info(f"TensorRT-LLM Debug Drafter 5")
            handler = RequestHandlerFactory().get_request_handler(handler_config)
            logging.error(f"DEBUG DRAFTER ENDPOINT: {config.namespace}.{config.component}.{config.endpoint}")
            await endpoint.serve_endpoint(handler.generate)


async def start_http_server(runtime: DistributedRuntime, config: Config, tokenizer):
    app = FastAPI(title="Drafter HTTP Gateway")
    
    # Create client to internal endpoint
    client = await runtime.namespace(config.namespace)\
                     .component(config.component)\
                     .endpoint(config.endpoint)\
                     .client()
    
    def convert_http_to_dynamo(request: dict):
        """Convert HTTP request to Dynamo internal format"""
        # Extract text from request
        if "prompt" in request:
            text = request["prompt"]
        elif "messages" in request:
            text = request["messages"][-1]["content"]  # Last message
        else:
            text = str(request)
        
        # Tokenize
        token_ids = tokenizer.encode(text)
        
        return {
            "token_ids": token_ids,
            "sampling_options": {
                "temperature": request.get("temperature", 0.7),
                "top_p": request.get("top_p", 1.0),
            },
            "stop_conditions": {
                "max_tokens": request.get("max_tokens", 100),
            }
        }
    
    def convert_dynamo_to_http(tokens: list):
        """Convert token list back to HTTP response"""
        # Detokenize
        text = tokenizer.decode(tokens)
        return {
            "choices": [{
                "message": {"content": text},
                "finish_reason": "stop"
            }]
        }
    
    @app.post("/draft/chat/completions")
    async def generate_draft(request: dict):
        try:
            print(f"🔵 Received request: {request}")
            
            # Convert HTTP request to Dynamo format
            dynamo_request = convert_http_to_dynamo(request)
            print(f"🔵 Converted to Dynamo: {dynamo_request}")
            
            # Call internal endpoint and collect tokens
            response_tokens = []
            print(f"🔵 Calling client.round_robin...")
            response = await client.round_robin(dynamo_request)
            print(f"🔵 Got response object: {response}")
            
            async for chunk in response:
                print(f"🔵 Got chunk: {chunk}")

                # Extract data from Annotated object
                chunk_data = chunk.data()
                print(f"🔵 Chunk data: {chunk_data}")
                
                if chunk_data.get("finish_reason"):
                    break
                response_tokens.extend(chunk_data.get("token_ids", []))
            
            print(f"🔵 Collected tokens: {response_tokens}")
            
            # Convert back to HTTP format
            result = convert_dynamo_to_http(response_tokens)
            print(f"🔵 Final result: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Error in generate_draft: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    # Start HTTP server
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8001))
    await server.serve()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
