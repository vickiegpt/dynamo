import sys

from common.parser import parse_tensorrt_llm_args, LLMAPIConfig
from common.base_engine import TensorrtLLMEngineConfig, BaseTensorrtLLMEngine


from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
from tensorrt_llm.executor import CppExecutorError

from dynamo.runtime import dynamo_endpoint
from tensorrt_llm.logger import logger

logger.set_level("debug")

class DynamoTRTLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, trt_llm_engine_config: TensorrtLLMEngineConfig):
        super().__init__(trt_llm_engine_config)

engine = None # Global variable to store the engine instance. This is initialized in the main function.

def init_global_engine(args):
    global engine
    args, engine_config = parse_tensorrt_llm_args()
    # Hard-coding for now.
    # Make it configurable via rust cli args.
    engine_config = LLMAPIConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tensor_parallel_size=1,
    )

@dynamo_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
async def generate(request):
    async for response in engine.generate_chat_helper(request):
        yield response


if __name__ == "__main__":
    init_global_engine(sys.argv)
