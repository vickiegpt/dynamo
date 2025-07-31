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

import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional, Protocol

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams

from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.engine import TensorRTLLMEngine
from dynamo.trtllm.publisher import Publisher
from dynamo.trtllm.utils.disagg_utils import (
    DisaggregatedParams,
    DisaggregatedParamsCodec,
)
from dynamo.trtllm.utils.multimodal_processor import MultimodalRequestProcessor

configure_dynamo_logging()


class DisaggregationMode(Enum):
    AGGREGATED = "prefill_and_decode"
    PREFILL = "prefill"
    DECODE = "decode"


class DisaggregationStrategy(Enum):
    PREFILL_FIRST = "prefill_first"
    DECODE_FIRST = "decode_first"


class TokenizerProtocol(Protocol):
    """
    A protocol for tokenizers that defines a decode method.

    This is used for type hinting to resolve mypy errors related to
    the tokenizer's decode method not being found on a generic 'object' type.
    """

    def decode(self, token_ids: List[int]) -> str:
        ...


@dataclass
class RequestHandlerConfig:
    """
    Configuration for the request handler
    """

    component: object
    engine: TensorRTLLMEngine
    default_sampling_params: SamplingParams
    publisher: Publisher
    disaggregation_mode: DisaggregationMode
    disaggregation_strategy: DisaggregationStrategy
    next_client: object
    multimodal_processor: Optional[
        MultimodalRequestProcessor
    ] = None  # for multimodal support
    tokenizer: Optional[
        TokenizerProtocol
    ] = None  # for decoding tokens in multimodal mode


class HandlerBase:
    """
    Base class for request handlers.
    """

    def __init__(self, config: RequestHandlerConfig):
        self.engine = config.engine
        self.component = config.component
        self.default_sampling_params = config.default_sampling_params
        self.publisher = config.publisher
        self.disaggregation_mode = config.disaggregation_mode
        self.disaggregation_strategy = config.disaggregation_strategy
        self.next_client = config.next_client
        self.multimodal_processor = config.multimodal_processor
        self.tokenizer = config.tokenizer  # store tokenizer for multimodal mode
        self.first_generation = True

    def check_error(self, result: dict):
        """
        Check if there is an error in the result.
        """
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return result["finish_reason"] == "error"
        else:
            return (
                result["finish_reason"] == "stop" or result["finish_reason"] == "error"
            )

    async def generate_locally(self, request: dict):
        """
        Generate responses based on the disaggregation mode in the request.
        """
        logging.debug(f"Request: {request}")

        # Default to text-based input. This will be overwritten if multimodal
        # content is found and processed.
        processed_input = None

        # Check for multimodal request and process it
        if self.multimodal_processor:
            # Normalize the request to handle OpenAI format
            if "stop_conditions" not in request:
                request["stop_conditions"] = {}
            if (
                "max_tokens" in request
                and "max_tokens" not in request["stop_conditions"]
            ):
                request["stop_conditions"]["max_tokens"] = request.pop("max_tokens")

            if "sampling_options" not in request:
                request["sampling_options"] = {}
            if (
                "temperature" in request
                and "temperature" not in request["sampling_options"]
            ):
                request["sampling_options"]["temperature"] = request.pop("temperature")

            processed_inputs = await self.multimodal_processor.process_openai_request(
                request
            )

            if "processed_inputs" in processed_inputs:
                processed_input = processed_inputs["processed_inputs"][0]

        else:
            # text-only flow
            processed_input = request.get("token_ids")

        # Check if there is an error in the publisher error queue
        publishers_error = (
            self.publisher.check_error_queue() if self.publisher else None
        )
        if publishers_error:
            raise publishers_error

        # Decode the disaggregated params from the request
        disaggregated_params = None

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            request["stop_conditions"]["max_tokens"] = 1
            disaggregated_params = LlmDisaggregatedParams(request_type="context_only")

        if "disaggregated_params" in request:
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                raise ValueError("Cannot provide disaggregated_params in prefill mode")

            received_params = DisaggregatedParams(**request["disaggregated_params"])

            disaggregated_params = DisaggregatedParamsCodec.decode(received_params)
            disaggregated_params.request_type = "generation_only"

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and disaggregated_params is None
        ):
            raise ValueError("Disaggregated params are required for decode mode")

        num_output_tokens_so_far = 0

        sampling_params = self.default_sampling_params

        for key, value in request["sampling_options"].items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request["stop_conditions"]["max_tokens"]
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        ignore_eos = request["stop_conditions"].get("ignore_eos")
        if ignore_eos:
            sampling_params.ignore_eos = ignore_eos

        min_tokens = request["stop_conditions"].get("min_tokens")
        if min_tokens:
            sampling_params.min_tokens = min_tokens

        # TODO: Instead of True, we should use streaming from the request.
        # However, currently dynamo run does not send streaming in the request.
        streaming = (
            False if self.disaggregation_mode == DisaggregationMode.PREFILL else True
        )

        request_id = request.get("id") or request.get("request_id", "unknown-id")
        model_name = request.get("model", "unknown_model")

        # NEW: Updated engine call to include multimodal data
        async for res in self.engine.llm.generate_async(
            inputs=processed_input,  # Use the correctly extracted inputs
            sampling_params=sampling_params,
            disaggregated_params=disaggregated_params,
            streaming=streaming,
        ):
            # TRTLLM engine needs to start generating tokens first before stats
            # can be retrieved.
            if self.first_generation and self.publisher:
                self.publisher.start()
                self.first_generation = False

            # Upon completion, send a final chunk with "stop" as the finish reason.
            # This signals to the client that the stream has ended.
            if res.finished and self.disaggregation_mode != DisaggregationMode.PREFILL:
                if self.multimodal_processor:
                    final_choice = {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                    final_out = {
                        "id": request_id,
                        "model": model_name,
                        "created": int(time.time()),
                        "object": "chat.completion.chunk",
                        "choices": [final_choice],
                        "finish_reason": "stop",
                    }
                    yield final_out
                else:
                    yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            # The engine returns all tokens generated so far. We must calculate the new
            # tokens generated in this iteration to create the "delta".
            next_total_toks = len(output.token_ids)
            if self.multimodal_processor and self.tokenizer:
                new_tokens = output.token_ids[num_output_tokens_so_far:]
                # Decode the new token IDs into a string. This is the incremental piece
                # of text to be sent to the client.
                delta_text = self.tokenizer.decode(new_tokens)
                # Assemble the delta payload for the response chunk.
                delta = {"content": delta_text if delta_text else ""}
                if self.first_generation:
                    # The first chunk must include the "assistant" role.
                    delta["role"] = "assistant"
                    self.first_generation = False
                choice = {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": output.finish_reason,
                }
                # Wrap the choice in the final response chunk following the OpenAI
                # streaming format.
                out = {
                    "id": request_id,
                    "model": model_name,
                    "created": int(time.time()),
                    "object": "chat.completion.chunk",
                    "choices": [choice],
                }
            else:
                out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # Return the disaggregated params only when operating in prefill mode.
                out["disaggregated_params"] = asdict(
                    DisaggregatedParamsCodec.encode(output.disaggregated_params)
                )
            # Yield the chunk to the client and update the token count for the next iteration.
            yield out
            num_output_tokens_so_far = next_total_toks
