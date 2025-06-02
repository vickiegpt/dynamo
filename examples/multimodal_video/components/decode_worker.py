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

import json
import logging

# import signal # No longer shutting down vLLM engine
from typing import AsyncIterator, Optional  # Added AsyncIterator

import connect

# import numpy as np # Not directly used now
import torch
from components.encode_worker import (
    VllmEncodeWorker,  # Still used for dynamo_address and client
)

# from components.prefill_worker import VllmPrefillWorker # Removed
from transformers import (  # Changed import
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
)
from utils.logging import check_required_workers

# from utils.nixl import NixlMetadataStore # Removed
# from utils.prefill_queue import PrefillQueue # Removed
from utils.protocol import EncodeRequest, MyRequestOutput, vLLMMultimodalRequest
from utils.vllm import parse_vllm_args  # Still used for model_id from args

from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service

# from vllm.entrypoints.openai.api_server import ( # Removed
#     build_async_engine_client_from_engine_args,
# )
# from vllm.inputs.data import TokensPrompt # Removed
# from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest # Removed
# from vllm.sampling_params import RequestOutputKind # Removed, sampling_params still exist on request


logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},  # GPU is used by HF model
    workers=1,
)
class VllmDecodeWorker:  # Consider renaming to HfDecodeWorker if vLLM is fully removed
    # prefill_worker = depends(VllmPrefillWorker) # Removed
    encode_worker = depends(
        VllmEncodeWorker
    )  # Kept for service discovery if runtime uses it

    def __init__(self):
        self.min_workers = 1  # For encode_worker client check
        class_name = self.__class__.__name__
        # Parse engine_args primarily for model_id and device
        # Other vLLM specific args might be irrelevant now.
        # Assuming engine_args.model provides the HuggingFace model_id
        # Assuming engine_args.device (or similar) could specify device, defaulting to "cuda:0" or "cuda"
        temp_engine_args = parse_vllm_args(class_name, "")  # To get model_id
        self.model_id = (
            temp_engine_args.model
            if temp_engine_args.model
            else "llava-hf/LLaVA-NeXT-Video-7B-hf"
        )
        self.device = "cuda"  # Or determine from temp_engine_args if available, ensuring it matches resource spec

        logger.info(
            f"Loading Hugging Face model: {self.model_id} on device: {self.device}"
        )
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()  # Set to evaluation mode

        logger.info(f"Loading Hugging Face processor: {self.model_id}")
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_id)

        # The do_remote_prefill flag and related logic are now largely irrelevant
        # as they were tied to vLLM's disaggregated prefill.
        # We will primarily support the "aggregated" path where data comes from EncodeWorker.
        # Set to False to default to the RDMA path.
        self.do_remote_prefill = (
            False  # Effectively makes this worker use the RDMA path
        )

        # signal.signal(signal.SIGTERM, self.shutdown_vllm_engine) # Removed
        # signal.signal(signal.SIGINT, self.shutdown_vllm_engine) # Removed

    @async_on_start
    async def async_init(self):
        # vLLM engine client and related initializations are removed.

        # Setup for RDMA transfer from EncodeWorker (Aggregated mode)
        # This part remains similar if we expect EncodeWorker to send processed data.
        PIXEL_VALUES_VIDEOS_SHAPE = (
            1,
            8,
            3,
            336,
            336,
        )  # Batch, NumFrames, Channels, Height, Width
        PIXEL_VALUES_VIDEOS_DTYPE = torch.float16
        # Device should match the model's device for the tensor received via RDMA
        PIXEL_VALUES_VIDEOS_DEVICE = self.device

        runtime = dynamo_context["runtime"]
        enc_comp_ns, enc_comp_name = VllmEncodeWorker.dynamo_address()  # type: ignore
        self.encode_worker_client = (
            await runtime.namespace(enc_comp_ns)
            .component(enc_comp_name)
            .endpoint("encode")
            .client()
        )

        self._connector = connect.Connector(runtime=runtime, namespace=enc_comp_ns)
        await self._connector.initialize()

        pixel_values_videos_tensor = torch.empty(
            PIXEL_VALUES_VIDEOS_SHAPE,
            dtype=PIXEL_VALUES_VIDEOS_DTYPE,
            device=PIXEL_VALUES_VIDEOS_DEVICE,
        )
        descriptor = connect.Descriptor(pixel_values_videos_tensor)
        descriptor.register_memory(self._connector)
        self._pixel_values_videos_descriptor = (pixel_values_videos_tensor, descriptor)

        await check_required_workers(self.encode_worker_client, self.min_workers)

        logger.info(
            f"{self.__class__.__name__} initialization with Hugging Face model complete."
        )

    # def shutdown_vllm_engine(self, signum, frame): # Removed
    #     pass

    # def get_remote_prefill_request_callback(self): # Removed
    #     pass

    @endpoint()
    async def generate(
        self, request: vLLMMultimodalRequest
    ) -> AsyncIterator[str]:  # Yields JSON string of MyRequestOutput
        request_id = request.request_id
        image_url = request.image_url  # Still needed to pass to EncodeWorker
        logger.info(
            f"Received multimodal request {{ id: {request_id} }} for HF generation."
        )

        if self.do_remote_prefill:
            logger.error(
                f"Request {request_id}: Disaggregated prefill path is not supported with Hugging Face direct generation."
            )
            # Yield an error response or raise an exception
            error_output = MyRequestOutput(
                request_id=request_id,
                prompt="",
                prompt_token_ids=[],
                prompt_logprobs=[],
                outputs=[
                    {
                        "text": "Error: Disaggregated prefill not supported.",
                        "token_ids": [],
                        "logprobs": [],
                        "finish_reason": "error",
                    }
                ],
                finished=True,
            )
            yield error_output.model_dump_json()
            return

        # Aggregated path: Get data from EncodeWorker
        logger.debug(f"Aggregated path for HF: request {{ id: {request_id} }}.")
        pixel_values_videos_tensor, descriptor = self._pixel_values_videos_descriptor

        # Create a WritableOperation for the EncodeWorker to write to.
        # This operation is then serialized and sent to the EncodeWorker.
        writable_op = self._connector.create_writable(descriptor)
        # This serialized_request represents the DecodeWorker's WritableOperation.
        sr_for_encoder = writable_op.to_serialized()

        # The EncodeRequest needs to carry sr_for_encoder as 'serialized_request'
        # This assumes EncodeRequest in utils/protocol.py has a field 'serialized_request: Optional[SerializedRequest]'.
        encode_request = EncodeRequest(
            request_id=request_id,
            image_url=image_url,
            serialized_request=sr_for_encoder,  # Pass the SerializedRequest of the WritableOperation
        )
        logger.debug(
            f"Sending encode request to encode_worker (with serialized_request): {encode_request.model_dump_json(exclude_none=True)}"
        )

        encode_generator = await self.encode_worker_client.round_robin(
            encode_request.model_dump_json(exclude_none=True)
        )

        serialized_auxiliary_payload_str: Optional[str] = None
        async for encode_response_item in encode_generator:
            raw_item_for_json_parsing: Optional[str] = None
            if isinstance(encode_response_item, str):
                raw_item_for_json_parsing = encode_response_item
            elif hasattr(
                encode_response_item, "data"
            ):  # Check if it has 'data' attribute
                data_attr_or_method = getattr(encode_response_item, "data")
                if callable(data_attr_or_method):
                    logger.info(
                        f"Request {request_id}: encode_response_item.data is a callable method. Calling it."
                    )
                    try:
                        called_data = data_attr_or_method()
                        if isinstance(called_data, str):
                            logger.info(
                                f"Request {request_id}: Result of calling .data() is a string. Using it."
                            )
                            raw_item_for_json_parsing = called_data
                        else:
                            logger.warning(
                                f"Request {request_id}: Result of calling .data() is type {type(called_data)}. Attempting str() conversion. Value (repr): {repr(called_data)}"
                            )
                            raw_item_for_json_parsing = str(called_data)
                    except Exception as e:
                        logger.error(
                            f"Request {request_id}: Error calling encode_response_item.data(): {e}. Falling back to str(encode_response_item).",
                            exc_info=True,
                        )
                        raw_item_for_json_parsing = str(
                            encode_response_item
                        )  # Fallback
                elif isinstance(data_attr_or_method, str):
                    logger.info(
                        f"Request {request_id}: encode_response_item.data is a string attribute. Using it."
                    )
                    raw_item_for_json_parsing = data_attr_or_method
                else:
                    logger.warning(
                        f"Request {request_id}: encode_response_item.data is type {type(data_attr_or_method)} (not str/callable). Attempting str() conversion. Value (repr): {repr(data_attr_or_method)}"
                    )
                    raw_item_for_json_parsing = str(data_attr_or_method)
            else:
                logger.warning(
                    f"Request {request_id}: encode_response item is type {type(encode_response_item)} and has no 'data' attribute. Attempting direct str() conversion of item. Value (repr): {repr(encode_response_item)}"
                )
                raw_item_for_json_parsing = str(encode_response_item)

            if raw_item_for_json_parsing is None:
                # This case should ideally not be reached if the logic above is exhaustive for expected types.
                logger.error(
                    f"Request {request_id}: Could not resolve JSON string from encode_response_item: {repr(encode_response_item)}"
                )
                raise ValueError(
                    f"Request {request_id}: Could not extract JSON string from encode_response_item: {repr(encode_response_item)}"
                )

            # Log the exact string and its type before attempting to parse
            logger.debug(
                f"Request {request_id}: Attempting json.loads on string of type: {type(raw_item_for_json_parsing)}."
            )
            logger.debug(
                f"Request {request_id}: JSON string to parse (first 500 chars): '{raw_item_for_json_parsing[:500]}'"
            )
            logger.debug(
                f"Request {request_id}: JSON string to parse (repr): {repr(raw_item_for_json_parsing)}"
            )

            encode_output_data = json.loads(raw_item_for_json_parsing)
            serialized_auxiliary_payload_str = encode_output_data.get(
                "serialized_auxiliary_payload"
            )
            logger.info(
                f"Received response from encode_worker for request {{ id: {request_id} }}. Aux payload received: {serialized_auxiliary_payload_str is not None}"
            )
            break
        else:  # No response from encode_worker
            raise RuntimeError(
                f"Request {request_id}: Did not receive any response from encode_worker."
            )

        if serialized_auxiliary_payload_str is None:
            raise RuntimeError(
                f"Request {request_id}: Did not receive serialized_auxiliary_payload in response from encode_worker."
            )

        # Now wait for the RDMA transfer (triggered by EncodeWorker using the sr_for_encoder) to complete.
        await writable_op.wait_for_completion()
        logger.info(
            f"Request {request_id}: RDMA transfer for pixel_values_videos completed."
        )

        # Auxiliary data is taken from the direct JSON response.
        auxiliary_payload = json.loads(serialized_auxiliary_payload_str)
        input_ids_list = auxiliary_payload.get("input_ids")
        attention_mask_list = auxiliary_payload.get("attention_mask")

        if input_ids_list is None or attention_mask_list is None:
            raise ValueError(
                f"Request {request_id}: 'input_ids' or 'attention_mask' not found in auxiliary_payload."
            )

        input_ids_tensor = torch.tensor(
            input_ids_list, dtype=torch.long, device=self.device
        )
        attention_mask_tensor = torch.tensor(
            attention_mask_list, dtype=torch.long, device=self.device
        )

        # pixel_values_videos_tensor is already on self.device from initialization

        inputs_for_hf_model = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "pixel_values_videos": pixel_values_videos_tensor,
        }

        # Map vLLM SamplingParams to Hugging Face generate arguments
        # For simplicity, starting with max_new_tokens and do_sample
        # request.sampling_params is vllm.SamplingParams
        max_new_tokens = (
            request.sampling_params.max_tokens
            if request.sampling_params.max_tokens is not None
            else 100
        )

        # do_sample logic: vLLM's temp=0 means greedy. HF's do_sample=False means greedy.
        # If temp > 0, then do_sample=True.
        temperature = request.sampling_params.temperature
        do_sample = False
        if temperature is not None and temperature > 0:
            do_sample = True
        else:  # temperature is 0 or None
            temperature = None  # HF generate handles None temp if do_sample is True, or ignores if False

        top_p = (
            request.sampling_params.top_p
            if request.sampling_params.top_p < 1.0
            else None
        )  # HF expects None if effectively no top_p
        top_k = (
            request.sampling_params.top_k
            if request.sampling_params.top_k != -1
            else None
        )  # HF expects None if effectively no top_k

        logger.info(
            f"Generating with HF model. Params: max_new_tokens={max_new_tokens}, do_sample={do_sample}, temperature={temperature}, top_p={top_p}, top_k={top_k}"
        )

        # Perform generation
        with torch.inference_mode():  # Important for HF inference
            output_ids = self.model.generate(
                **inputs_for_hf_model,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature
                if do_sample
                else None,  # Only pass temp if sampling
                top_p=top_p if do_sample else None,  # Only pass top_p if sampling
                top_k=top_k if do_sample else None,  # Only pass top_k if sampling
                pad_token_id=self.processor.tokenizer.pad_token_id,  # Important for some models
            )

        input_token_len = inputs_for_hf_model["input_ids"].shape[1]
        generated_tokens_ids = output_ids[0][input_token_len:]
        response_text = self.processor.decode(
            generated_tokens_ids, skip_special_tokens=True
        )

        logger.debug(
            f"Request {request_id} generated response: {response_text[:200]}..."
        )

        # Construct and yield MyRequestOutput
        # For simplicity, we are not streaming token by token here, but yielding one complete response.
        # To stream, one would need to implement a custom stopping criteria and yield token by token.
        final_output = MyRequestOutput(
            request_id=request_id,
            prompt="",  # The original prompt text isn't directly available here, input_ids are
            prompt_token_ids=input_ids_tensor[
                0
            ].tolist(),  # Send back the input_ids used
            prompt_logprobs=[],  # Not available from basic HF generate
            outputs=[
                {
                    "index": 0,  # Added: Default index for the output sequence
                    "text": response_text,
                    "token_ids": generated_tokens_ids.tolist(),
                    "logprobs": [],  # Not available from basic HF generate
                    "cumulative_logprob": 0.0,  # Added: Placeholder value
                    "finish_reason": "length"
                    if len(generated_tokens_ids) >= max_new_tokens
                    else "stop",  # Heuristic
                }
            ],
            finished=True,
        )
        yield final_output.model_dump_json()
