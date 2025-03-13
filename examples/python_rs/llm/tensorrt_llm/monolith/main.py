import uvloop
import asyncio
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from .worker import TensorrtLLMEngineConfig, TensorrtLLMEngine
from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker()
async def decorated_worker(runtime: DistributedRuntime, engine_config: LLMAPIConfig):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    namespace_str = "dynamo"
    component_str = "tensorrt-llm"

    component = runtime.namespace(namespace_str).component(component_str)
    await component.create_service()

    completions_endpoint = component.endpoint("completions")
    chat_completions_endpoint = component.endpoint("chat/completions")

    trt_llm_engine_config = TensorrtLLMEngineConfig(
        namespace_str=namespace_str,
        component_str=component_str,
        engine_config=engine_config,
    )
    engine = TensorrtLLMEngine(trt_llm_engine_config)

    await asyncio.gather(
        completions_endpoint.serve_endpoint(engine.generate_completion),
        chat_completions_endpoint.serve_endpoint(engine.generate_chat),
    )

if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()
    asyncio.run(decorated_worker(engine_config))
