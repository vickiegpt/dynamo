import sys
import json
import time
import asyncio
import torch
import uuid
import logging
import signal
import random
import threading
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from dataclasses import asdict
from queue import Queue
from typing import Any, Optional, Any, List, Optional, TypedDict, Dict, Union, Literal, TypeAlias
from pydantic import BaseModel, ConfigDict, Field
from openai.types.chat import ChatCompletionMessageParam
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi import LLM, SamplingParams, KvCacheConfig
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionResponseStreamChoice,
    DisaggregatedParams,
    DeltaMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DynamoTRTLLMChatCompletionRequest(ChatCompletionRequest):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    nvext: Optional[dict] = Field(default=None)

class DynamoTRTLLMChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)

@dataclass
class LLMAPIConfig:
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        pytorch_backend_config: PyTorchConfig | None = None,
        kv_cache_config: KvCacheConfig | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.pytorch_backend_config = pytorch_backend_config
        self.kv_cache_config = kv_cache_config
        self.extra_args = kwargs

        # Hardcoded to skip tokenizer init for now.
        # We will handle the tokenization/detokenization
        # outside of the engine
        if "skip_tokenizer_init" in self.extra_args:
            self.extra_args.pop("skip_tokenizer_init")
        self.skip_tokenizer_init = True

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "pytorch_backend_config": self.pytorch_backend_config,
            "kv_cache_config": self.kv_cache_config,
            "skip_tokenizer_init": self.skip_tokenizer_init,
        }
        if self.extra_args:
            data.update(self.extra_args)
        return data

class Tokens(BaseModel):
    tokens: list[int]

class ConversationMessage(TypedDict):
    role: str
    content: str

class TRTLLMWorkerRequest(BaseModel):
    model: str
    id: str
    prompt: str | None = None
    sampling_params: dict
    streaming: bool = True
    conversation: Optional[List[ConversationMessage]] = Field(default=None)
    tokens: Optional[Tokens] = Field(default=None)

class TRTLLMWorkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    request_id: str
    prompt: str | None = None
    prompt_token_ids: list[int]
    outputs: list[dict]
    finished: bool

@dataclass(slots=True)
class Logprob:
    """Holds logprob and vocab rank for a token."""

    logprob: float
    rank: Optional[int] = None

TokenLogprobs: TypeAlias = list[dict[int, Logprob]]

@dataclass
class TRTLLMWorkerResponseOutput:
    index: int
    text: str = ""
    token_ids: Optional[List[int]] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[TokenLogprobs] = field(default_factory=list)
    prompt_logprobs: Optional[TokenLogprobs] = field(default_factory=list)
    finish_reason: Optional[Literal["stop", "length", "timeout", "cancelled"]] = None
    stop_reason: Optional[Union[int, str]] = None
    generation_logits: Optional[torch.Tensor] = None
    disaggregated_params: Optional[DisaggregatedParams] = None

    # hidden fields for tracking the diffs
    _last_text_len: int = field(default=0, init=True, repr=False)
    _last_token_ids_len: int = field(default=0, init=True, repr=False)
    _last_logprobs_len: int = field(default=0, init=True, repr=False)
    _incremental_states: Optional[dict] = field(default=None, init=True, repr=False)
    # the result of result_handler passed to postprocess workers
    _postprocess_result: Any = None

    @property
    def length(self) -> int:
        return 0 if self.token_ids is None else len(self.token_ids)

    @property
    def text_diff(self) -> str:
        return self.text[self._last_text_len :]

    @property
    def token_ids_diff(self) -> List[int]:
        return (
            [] if self.token_ids is None else self.token_ids[self._last_token_ids_len :]
        )

    # Ignoring the mypy error here as this is copied from TensorRT-LLM project.
    # https://github.com/NVIDIA/TensorRT-LLM/blob/19c6e68bec891b66146a09647ee7b70230ef5f67/tensorrt_llm/executor/result.py#L68
    # TODO: Work with the TensorRT-LLM team to get this fixed.
    @property
    def logprobs_diff(self) -> List[float]:  # type: ignore
        return [] if self.logprobs is None else self.logprobs[self._last_logprobs_len :]  # type: ignore



def get_sampling_params(sampling_params):
    # Removes keys starting with '_' from the sampling params which gets
    # added by the LLM API. TRTLLM does not support creating SamplingParams
    # from a dictionary with keys starting with '_'.
    cleaned_dict = {
        key: value for key, value in sampling_params.items() if not key.startswith("_")
    }
    return SamplingParams(**cleaned_dict)


# Core LLM engine
class BaseTensorrtLLMEngine:
    def __init__(self, engine_config: LLMAPIConfig = None):
        self._error_queue: Queue = Queue()
        self._engine_config = engine_config

    def init_engine(self):
        # Run the engine in a separate thread running the AsyncIO event loop.
        self._llm_engine: Optional[Any] = None
        self._llm_engine_start_cv = threading.Condition()
        self._llm_engine_shutdown_event = asyncio.Event()
        self._event_thread = threading.Thread(
            target=asyncio.run, args=(self._run_llm_engine(),)
        )

        self._event_thread.start()
        with self._llm_engine_start_cv:
            while self._llm_engine is None:
                self._llm_engine_start_cv.wait()

        # The 'threading.Thread()' will not raise the exception here should the engine
        # failed to start, so the exception is passed back via the engine variable.
        if isinstance(self._llm_engine, Exception):
            e = self._llm_engine
            logger.error(f"Failed to start engine: {e}")
            if self._event_thread is not None:
                self._event_thread.join()
                self._event_thread = None
            raise e
    
    def shutdown_engine(self):
        self._llm_engine_shutdown_event.set()

    async def _run_llm_engine(self):
        # Counter to keep track of ongoing request counts.
        self._ongoing_request_count = 0

        @asynccontextmanager
        async def async_llm_wrapper():
            # Create LLM in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            try:
                llm = await loop.run_in_executor(
                    None,
                    lambda: LLM(
                        model=self._engine_config.model_name,
                        **self._engine_config.to_dict(),
                    ),
                )
                yield llm
            finally:
                if "llm" in locals():
                    # Run shutdown in a thread to avoid blocking
                    await loop.run_in_executor(None, llm.shutdown)

        try:
            async with async_llm_wrapper() as engine:
                # Capture the engine event loop and make it visible to other threads.
                self._event_loop = asyncio.get_running_loop()

                # Signal the engine is started and make it visible to other threads.
                with self._llm_engine_start_cv:
                    self._llm_engine = engine
                    self._llm_engine_start_cv.notify_all()

                logger.info("Engine loaded and ready to serve...")

                # Wait for the engine shutdown signal.
                await self._llm_engine_shutdown_event.wait()

                # Wait for the ongoing requests to complete.
                while self._ongoing_request_count > 0:
                    logger.info(
                        "Awaiting remaining {} requests".format(
                            self._ongoing_request_count
                        )
                    )
                    await asyncio.sleep(1)

                # Cancel all tasks in the event loop.
                for task in asyncio.all_tasks(loop=self._event_loop):
                    if task is not asyncio.current_task():
                        task.cancel()

        except Exception as e:
            # Signal and pass the exception back via the engine variable if the engine
            # failed to start. If the engine has started, re-raise the exception.
            with self._llm_engine_start_cv:
                if self._llm_engine is None:
                    self._llm_engine = e
                    self._llm_engine_start_cv.notify_all()
                    return
            raise e

        self._llm_engine = None
        logger.info("Shutdown complete")

    async def generate(self, request: TRTLLMWorkerRequest):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        if not self._error_queue.empty():
            raise self._error_queue.get()

        self._ongoing_request_count += 1
        logger.info(f"ongoing request count: {self._ongoing_request_count}")

        try:
            worker_inputs = request.tokens.tokens
            sampling_params = get_sampling_params(request.sampling_params)
            async for response in self._llm_engine.generate_async(
                inputs=worker_inputs,
                sampling_params=sampling_params,
                streaming=request.streaming,
            ):
                yield TRTLLMWorkerResponse(
                    request_id=request.id,
                    prompt_token_ids=response.prompt_token_ids,
                    outputs=[asdict(response.outputs[0])],
                    finished=response.finished,
                ).model_dump_json(exclude_unset=True)

        except CppExecutorError:
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))
        
        self._ongoing_request_count -= 1


class Processor:
    def __init__(self, model_name: str, llm_engine: BaseTensorrtLLMEngine):
        self.model_name = model_name
        self.llm_engine = llm_engine
        self.tokenizer = tokenizer_factory(self.model_name)

    async def generate_chat(self, raw_request: DynamoTRTLLMChatCompletionRequest):
        # max_tokens is deprecated, however if the max_tokens is provided instead
        # of max_completion_tokens, we will use the value as max_completion_tokens.
        if raw_request.max_tokens is not None:
            if raw_request.max_completion_tokens is None:
                raw_request.max_completion_tokens = raw_request.max_tokens
            else:
                if raw_request.max_tokens != raw_request.max_completion_tokens:
                    raise ValueError(
                        "max_tokens and max_completion_tokens must be the same"
                    )

        # min_tokens isn't currently propagated through the Rust OpenAI HTTP frontend,
        # and ignore_eos is passed through the 'nvext' field, so set both when found.
        if raw_request.nvext:
            ignore_eos = raw_request.nvext.get("ignore_eos")
            raw_request.ignore_eos = ignore_eos
            # If ignore_eos is True, set min_tokens to max_tokens to guarantee
            # the full expected OSL for consistent benchmarking purposes.
            if ignore_eos:
                raw_request.min_tokens = raw_request.max_completion_tokens

        raw_request.skip_special_tokens = False
        raw_request.add_special_tokens = False
        raw_request.spaces_between_special_tokens = False
        preprocessed_request = await self.preprocess(raw_request)

        first_iteration = True
        # keep generating but ignore the response
        async for raw_response in self.llm_engine.generate(preprocessed_request):
            response = TRTLLMWorkerResponse.model_validate_json(raw_response)
            last_token_ids_len = response.outputs[0]["_last_token_ids_len"]
            response.outputs[0]["text"] = self.tokenizer.decode(
                response.outputs[0]["token_ids"][last_token_ids_len:]
            )
            response.outputs = [TRTLLMWorkerResponseOutput(**response.outputs[0])]
            response_data = self.create_chat_stream_response(
                raw_request,
                raw_request.id,
                response,
                first_iteration=first_iteration,
            )
            first_iteration = False
            #logger.debug(f"[postprocessor] Response: {response_data}")
            yield response_data
    
    def _get_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            role = "assistant"
        else:
            role = request.messages[-1]["role"]
        return role

    def yield_first_chat(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        response: RequestOutput,
        content: str | None = None,
    ):
        role = self._get_role(request)
        num_choices = 1 if request.n is None else request.n
        num_tokens = len(response.prompt_token_ids)
        content = response.outputs[0].text_diff

        for i in range(num_choices):
            choice = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role=role, content=content),
                finish_reason=None,
            )

            chunk = DynamoTRTLLMChatCompletionStreamResponse(
                id=request_id,
                choices=[choice],
                model=self.model_name,
            )
            chunk.usage = self._stream_usage_info(request, num_tokens, 0)

            return chunk.model_dump_json()

    def create_chat_stream_response(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        response: RequestOutput,
        first_iteration: bool = True,
    ) -> str:
        num_choices = 1 if request.n is None else request.n
        finish_reason_sent = [False] * num_choices
        role = self._get_role(request)

        prompt_tokens = len(response.prompt_token_ids)
        if first_iteration:
            return self.yield_first_chat(request, request_id, response)
        
        first_iteration = False
        for output in response.outputs:
            i = output.index

            if finish_reason_sent[i]:
                continue

            delta_text = output.text_diff
            if (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                delta_message = DeltaMessage(
                    tool_calls=[
                        ToolCall(
                            function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=delta_text,
                            )
                        )
                    ]
                )
            else:
                delta_message = DeltaMessage(content=delta_text, role=role)

            choice = ChatCompletionResponseStreamChoice(
                index=i, delta=delta_message, finish_reason=None
            )

            if output.finish_reason is not None:
                choice.finish_reason = output.finish_reason
                choice.stop_reason = output.stop_reason
                finish_reason_sent[i] = True

            chunk = DynamoTRTLLMChatCompletionStreamResponse(
                id=request_id,
                choices=[choice],
                model=self.model_name,
            )
            #logger.debug(f"[processor] Chunk: {chunk}")
            chunk.usage = self._stream_usage_info(request, prompt_tokens, output.length)
            return chunk.model_dump_json()
        
        return "data: [DONE]\n\n"


    def _stream_usage_info(
        self, request: ChatCompletionRequest, prompt_tokens: int, completion_tokens: int
    ):
        if (
            request.stream_options
            and request.stream_options.include_usage
            and request.stream_options.continuous_usage_stats
        ):
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        else:
            usage = None
        return usage
                
    async def preprocess(self, request):
        conversation: List[Any] = []
        for message in request.messages:
            conversation.extend(self.parse_chat_message_content(message))

        tool_dicts = (
            None
            if request.tools is None
            else [tool.model_dump() for tool in request.tools]
        )
        prompt = self.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=True,
            add_generation_prompt=request.add_generation_prompt,
            tools=tool_dicts,
            documents=request.documents,
            chat_template=request.chat_template,
            **(request.chat_template_kwargs or {}),
        )
        sampling_params = request.to_sampling_params()
        sampling_params._setup(self.tokenizer)
        sampling_params.stop = None

        return TRTLLMWorkerRequest(
            id=request.id,
            model=request.model,
            sampling_params=asdict(sampling_params),
            streaming=request.stream,
            conversation=conversation,
            tokens=Tokens(tokens=prompt),
        )

    def parse_chat_message_content(
            self,
            message: ChatCompletionMessageParam,
        ) -> Union[ConversationMessage, List[ConversationMessage], List[None]]:
            role = message["role"]
            content = message.get("content")

            if content is None:
                return []
            if isinstance(content, str):
                return [ConversationMessage(role=role, content=content)]

            texts: List[str] = []
            for part in content:
                part_type = part["type"]
                if part_type == "text":
                    text = part["text"]  # type: ignore
                    texts.append(text)
                else:
                    raise NotImplementedError(f"{part_type} is not supported")

            text_prompt = "\n".join(texts)
            return [ConversationMessage(role=role, content=text_prompt)]

# benchmark related setup
ISL = 8000
OSL = 256
NUM_THREADS = 256
NUM_REQUESTS = NUM_THREADS * 10
NUM_LOOPS = 10
INPUTS_JSON_PATH="/lustre/fsw/core_dlfw_ci/rihuo/output.json"

def content_generator(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        contents = [item["content"] for item in data]

    index = 0
    while True:
        yield contents[index]
        index = (index + 1) % len(contents)

input_gen = content_generator(INPUTS_JSON_PATH)

async def flush_generate_chat(chat_processor: Processor, raw_request: DynamoTRTLLMChatCompletionRequest):
    responses = []
    async for response_data in chat_processor.generate_chat(raw_request):
        responses.append(response_data)
        await asyncio.sleep(random.uniform(0.001, 0.1))
    return responses

async def run_thread(thread_id: int, queue: asyncio.Queue, chat_processor: Processor):
    while True:
        request_id = await queue.get()
        try:
            input_text = next(input_gen)
            request = DynamoTRTLLMChatCompletionRequest(
                messages=[{"role": "user", "content": input_text}],
                model=chat_processor.model_name,
                max_tokens=OSL,
                min_tokens=OSL,
                stream=True,
                nvext={"ignore_eos": True}
            )

            await flush_generate_chat(chat_processor, request)
            print(f"Worker {thread_id} completed request {request_id}")
        except Exception as e:
            print(f"Worker {thread_id} failed on request {request_id}: {e}")
            return
        finally:
            queue.task_done()


async def main():
    model_name = "/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4"
    # initialize the engine
    pytorch_backend_config = PyTorchConfig(
        use_cuda_graph=True,
        cuda_graph_padding_enabled=True,
        cuda_graph_batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        kv_cache_dtype="fp8",
        enable_overlap_scheduler=True,
    )
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.3
    )
    kwargs = {
        "backend": "pytorch",
        "tensor_parallel_size": 4,
        "moe_expert_parallel_size": 4,
        "enable_attention_dp": True,
        "max_batch_size": 256,
        "max_num_tokens": 8448,
        "max_seq_len": 8448,
    }

    engine_config = LLMAPIConfig(
        model_name=model_name,
        model_path=None,
        pytorch_backend_config=pytorch_backend_config,
        kv_cache_config=kv_cache_config,
        **kwargs
    )

    trtllm_engine = BaseTensorrtLLMEngine(engine_config=engine_config)
    trtllm_engine.init_engine()

    # initialize the processor
    chat_processor = Processor(model_name=model_name, llm_engine=trtllm_engine)

    try: 
        for loop_id in range(NUM_LOOPS):
            logger.info(f"Starting loop {loop_id + 1}")
            # start the benchmarks with num_threads
            queue = asyncio.Queue()
            for i in range(NUM_REQUESTS):
                await queue.put(i)
            
            threads = [
                asyncio.create_task(run_thread(i, queue, chat_processor=chat_processor))
                for i in range(NUM_THREADS)
            ]

            # Wait until all tasks are completed
            await queue.join()

            # Cancel the workers once done
            for t in threads:
                t.cancel()

            results = await asyncio.gather(*threads, return_exceptions=True)

            has_exception = False
            for result in results:
                if isinstance(result, Exception):
                    print("Task failed with: ", result)
                    has_exception = True
            
            if has_exception:
                break

            time.sleep(20)
    except KeyboardInterrupt:
        print("Interrupted by user!")
        results = await asyncio.gather(*threads, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print("Task failed with:", repr(result))
    except Exception as e:
        print("Unexpected error occurred:", repr(e))
        results = await asyncio.gather(*threads, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print("Task failed with:", repr(result))
    finally:
        input_gen.close()
        trtllm_engine.shutdown_engine()

if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
