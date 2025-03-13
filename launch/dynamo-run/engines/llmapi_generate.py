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

import sys
import signal
import uuid

from llmapi.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from llmapi.parser import LLMAPIConfig
from llmapi.processor import parse_chat_message_content
from llmapi.trtllm_engine import TensorrtLLMEngine
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)

from dynamo.runtime import dynamo_endpoint

logger.set_level("info")

# Hard-coding for now.
# Make it configurable via rust cli args.
engine_config = LLMAPIConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tensor_parallel_size=1,
)
trt_llm_engine_config = TensorrtLLMEngineConfig(
    engine_config=engine_config,
)

engine = None

@dynamo_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
async def generate(request):
    if engine is None:
        raise RuntimeError("Engine not initialized")

    logger.debug(f"Received chat request: {request}")
    request_id = str(uuid.uuid4())

    try:
        conversation = []
        for message in request.messages:
            conversation.extend(parse_chat_message_content(message))
        tool_dicts = (
            None
            if request.tools is None
            else [tool.model_dump() for tool in request.tools]
        )
        prompt: str = engine._tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=request.add_generation_prompt,
            tools=tool_dicts,
            documents=request.documents,
            chat_template=request.chat_template,
            **(request.chat_template_kwargs or {}),
        )
        sampling_params = request.to_sampling_params()

        promise = engine._llm_engine.generate_async(
            prompt,
            sampling_params,
            streaming=request.stream,
        )
        # NOTE: somehow stream and non-stream is working with the same path
        response_generator = engine.chat_processor.stream_response(
            request, request_id, conversation, promise
        )
        async for response in response_generator:
            logger.debug(f"Generated response: {response}")
            yield response

    except CppExecutorError:
        # If internal executor error is raised, shutdown the server
        signal.raise_signal(signal.SIGINT)
    except Exception as e:
        logger.error(f"Error in generate: {e}")
        raise RuntimeError("Failed to generate: " + str(e))

if __name__ == "__main__":
    print(f"MAIN: {sys.argv}")
    engine = TensorrtLLMEngine(trt_llm_engine_config)