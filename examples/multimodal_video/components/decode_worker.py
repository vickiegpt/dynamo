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
import dataclasses
import json
import logging
import os
import signal
from typing import Optional, Union

import connect
import torch
from components.disagg_router import PyDisaggregatedRouter
from components.encode_worker import VllmEncodeWorker
from components.prefill_worker import VllmPrefillWorker
from transformers import AutoProcessor
from utils.logging import check_required_workers
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import (
    EncodeRequest,
    EncodeResponse,
    MyRequestOutput,
    vLLMMultimodalRequest,
)
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest
from vllm.sampling_params import RequestOutputKind

from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)

# Constants for the shape and dtype of the INCOMING FRAMES tensor from EncodeWorker.
# IMPORTANT ASSUMPTION: EncodeWorker must provide frames of this fixed shape and dtype.
# Example: 8 frames, each 224x224 RGB.
NUM_SAMPLED_FRAMES = 8  # Should match VllmEncodeWorker's num_frames_to_sample
FRAME_HEIGHT = 336
FRAME_WIDTH = 336
FRAME_CHANNELS = 3
INCOMING_FRAMES_SHAPE = (
    NUM_SAMPLED_FRAMES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FRAME_CHANNELS,
)
INCOMING_FRAMES_DTYPE = torch.uint8
INCOMING_FRAMES_DEVICE = "cuda"


def _convert_logprobs_to_cpu_serializable(logprobs_data):
    """Converts logprobs data, which might contain CUDA scalar tensors, to a CPU-based serializable format."""
    if logprobs_data is None:
        return None

    # Handles List[Optional[Dict[int, float_or_scalar_tensor]]]
    serializable_logprobs = []
    for item in logprobs_data:
        if item is None:
            serializable_logprobs.append(None)
        else:
            cpu_item = {
                token_id: (
                    val.cpu().item() if isinstance(val, torch.Tensor) else float(val)
                )
                for token_id, val in item.items()
            }
            serializable_logprobs.append(cpu_item)
    return serializable_logprobs


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmDecodeWorker:
    # For disaggregated serving, we need to link the prefill worker to the vllm worker
    prefill_worker = depends(VllmPrefillWorker)
    # For aggregated serving, we need to link the encode worker to the vllm worker.
    encode_worker = depends(VllmEncodeWorker)

    def _expand_video_tokens_in_prompt(
        self,
        original_tokens: list[int],
        num_frames_to_expand_to: int,
        image_token_id: int,  # This should be the ID from hf_processor.tokenizer
        add_dummy_tokens: bool,
        dummy_token_id: int = 0,
        num_dummy_tokens_per_frame: int = 0,
    ) -> list[int]:
        """
        Expands the first occurrence of image_token_id in original_tokens
        to num_frames_to_expand_to occurrences. Optionally adds dummy tokens.
        """
        expanded_prompt_list = []
        token_expanded_successfully = False
        for (
            token_id_val
        ) in original_tokens:  # Renamed token to token_id_val to avoid conflict
            if token_id_val == image_token_id and not token_expanded_successfully:
                # Debug: log the expansion details
                video_tokens_to_add = num_frames_to_expand_to
                dummy_tokens_per_frame = (
                    num_dummy_tokens_per_frame if add_dummy_tokens else 0
                )
                total_dummy_tokens = video_tokens_to_add * dummy_tokens_per_frame
                total_expansion = video_tokens_to_add + total_dummy_tokens
                logger.info(
                    f"Token expansion debug: {video_tokens_to_add} video tokens + {total_dummy_tokens} dummy tokens = {total_expansion} total"
                )

                for frame_idx in range(num_frames_to_expand_to):
                    expanded_prompt_list.append(image_token_id)
                    if add_dummy_tokens:
                        dummy_tokens_to_add = [
                            dummy_token_id
                        ] * num_dummy_tokens_per_frame
                        expanded_prompt_list.extend(dummy_tokens_to_add)
                        logger.debug(
                            f"Frame {frame_idx}: added 1 video token + {len(dummy_tokens_to_add)} dummy tokens"
                        )

                # Debug: count what we actually added
                added_video_tokens = sum(
                    1 for t in expanded_prompt_list if t == image_token_id
                )
                added_dummy_tokens = sum(
                    1 for t in expanded_prompt_list if t == dummy_token_id
                )
                logger.info(
                    f"Expansion result: {added_video_tokens} video tokens (ID {image_token_id}), {added_dummy_tokens} dummy tokens (ID {dummy_token_id})"
                )

                token_expanded_successfully = True
            else:
                expanded_prompt_list.append(token_id_val)

        if not token_expanded_successfully:
            # If the specific video token ID isn't found (e.g. prompt had no video placeholder),
            # it implies the original prompt didn't intend for video.
            # This might be an issue if video data is expected.
            logger.warning(
                f"Image token ID {image_token_id} for expansion not found in prompt tokenized by hf_processor. Prompt: {original_tokens}. This might be okay if no video was intended in this specific prompt structure."
            )
            return list(original_tokens)  # Return original if no video token to expand

        # Final debug: count all placeholder tokens in the result
        video_placeholders = sum(1 for t in expanded_prompt_list if t == image_token_id)
        dummy_placeholders = sum(1 for t in expanded_prompt_list if t == dummy_token_id)
        other_zeros = sum(
            1 for t in expanded_prompt_list if t == 0 and t != dummy_token_id
        )
        total_placeholders = video_placeholders + dummy_placeholders + other_zeros
        logger.info(
            f"Final token analysis: {len(expanded_prompt_list)} total tokens, {total_placeholders} potential placeholders ({video_placeholders} video + {dummy_placeholders} dummy + {other_zeros} other zeros)"
        )

        return expanded_prompt_list

    def __init__(self):
        self.client = None
        self.min_workers = 1
        self.disaggregated_router: Optional[PyDisaggregatedRouter] = None
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.model_path = self.engine_args.model
        self.do_remote_prefill = self.engine_args.remote_prefill
        self.model_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = self.model_name
        logger.info(
            f"Prefill queue: {self._prefill_queue_nats_server}:{self._prefill_queue_stream_name}"
        )

        if self.engine_args.remote_prefill:
            if self.engine_args.enable_chunked_prefill is not False:
                logger.info("Chunked prefill is not supported yet, setting to False")
                self.engine_args.enable_chunked_prefill = False

            if self.engine_args.preemption_mode != "swap":
                logger.info("Preemption mode is not supported yet, setting to swap")
                self.engine_args.preemption_mode = "swap"

            if self.engine_args.pipeline_parallel_size != 1:
                logger.info("Pipeline parallel size is not supported yet, setting to 1")
                self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

        # Load the Hugging Face processor
        try:
            self.hf_processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            logger.info(f"Successfully loaded AutoProcessor from: {self.model_path}")
            if (
                not hasattr(self.hf_processor, "tokenizer")
                or self.hf_processor.tokenizer is None
            ):
                logger.warning(
                    f"Loaded AutoProcessor from {self.model_path} but it does not have a 'tokenizer' attribute or it is None."
                )
        except Exception as e:
            logger.error(
                f"Failed to load AutoProcessor from {self.model_path}: {e}",
                exc_info=True,
            )
            # Depending on the desired behavior, you might want to raise the error
            # or allow the worker to start without a processor if it's optional for some paths.
            # For this change, processor is critical.
            raise RuntimeError(f"Failed to initialize AutoProcessor: {e}")

        runtime = dynamo_context["runtime"]

        # Common setup for interacting with EncodeWorker (NIXL, client)
        # This is needed for aggregated mode OR for local prefill in disaggregated mode.
        enc_comp_ns, enc_comp_name = VllmEncodeWorker.dynamo_address()  # type: ignore
        self.encode_worker_client = (
            await runtime.namespace(enc_comp_ns)
            .component(enc_comp_name)
            .endpoint("encode")
            .client()
        )
        self._connector = connect.Connector(runtime=runtime, namespace=enc_comp_ns)
        await self._connector.initialize()

        # NIXL buffer for receiving raw video frames.
        # Uses INCOMING_FRAMES_SHAPE, INCOMING_FRAMES_DTYPE, INCOMING_FRAMES_DEVICE constants.
        raw_frames_tensor = torch.empty(
            INCOMING_FRAMES_SHAPE,
            dtype=INCOMING_FRAMES_DTYPE,
            device=INCOMING_FRAMES_DEVICE,
        )
        descriptor = connect.Descriptor(raw_frames_tensor)
        descriptor.register_memory(self._connector)
        self._frames_descriptor = (raw_frames_tensor, descriptor)

        await check_required_workers(self.encode_worker_client, self.min_workers)

        if self.do_remote_prefill:  # Disaggregated mode specific setup
            metadata = self.engine_client.nixl_metadata
            metadata_store = NixlMetadataStore("dynamo", runtime)
            await metadata_store.put(metadata.engine_id, metadata)

            if self.engine_args.conditional_disagg:
                self.disaggregated_router = PyDisaggregatedRouter(
                    runtime,
                    self.model_name,
                    max_local_prefill_length=self.engine_args.max_local_prefill_length,
                    max_prefill_queue_size=self.engine_args.max_prefill_queue_size,
                )
                await self.disaggregated_router.async_init()
            else:
                self.disaggregated_router = (
                    None  # Always remote prefill if not conditional_disagg
                )

            # embedding_size is used for dummy token calculation in remote prefill case
            # For LLaVA-NeXT-Video, the model expects exactly 144 tokens per frame
            # This was fixed in vLLM PR #8496 for LLaVA-NeXT feature size calculation
            # Each video frame needs 144 total tokens (143 dummy + 1 video token)
            self.embedding_size = 144
            logger.info(
                f"Disaggregated mode: Using correct LLaVA-NeXT-Video embedding size: {self.embedding_size}"
            )

        else:  # Aggregated mode specific setup
            self.disaggregated_router = (
                None  # No disaggregated router in aggregated mode
            )
            logger.info(
                "Aggregated mode: VllmDecodeWorker will handle multimodal data directly via NIXL."
            )

        logger.info("Initialization complete.")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("Shutdown complete.")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    def get_remote_prefill_request_callback(self):
        async def callback(request: RemotePrefillRequest):
            logger.info(
                f"DecodeWorker {request.request_id}: Remote prefill callback triggered. Attempting to enqueue to stream '{self._prefill_queue_stream_name}'."
            )
            logger.info(
                f"DecodeWorker {request.request_id}: Enqueueing request with multimodal_data_source: {request.multimodal_data_source}, prompt_token_ids length: {len(request.prompt_token_ids) if request.prompt_token_ids else 'None'}"
            )
            try:
                async with PrefillQueue.get_instance(
                    nats_server=self._prefill_queue_nats_server,
                    stream_name=self._prefill_queue_stream_name,
                ) as prefill_queue:
                    await prefill_queue.enqueue_prefill_request(request)
                logger.info(
                    f"DecodeWorker {request.request_id}: Successfully enqueued remote prefill request."
                )
            except Exception as e:
                logger.error(
                    f"DecodeWorker {request.request_id}: Failed to enqueue remote prefill request: {e}",
                    exc_info=True,
                )

        return callback

    @endpoint()
    async def generate(self, request: vLLMMultimodalRequest):
        request_id = request.request_id
        image_url = request.image_url  # Video path for EncodeWorker
        user_text_prompt = request.engine_prompt.get(
            "text_prompt", "Describe the video."
        )
        logger.info(
            f"Received multimodal request {{ id: {request_id} }} with user text: '{user_text_prompt}'."
        )

        # 1. Construct conversation and get text_prompt_string from hf_processor
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text_prompt},
                    {"type": "video"},
                ],
            },
        ]
        if (
            not hasattr(self, "hf_processor")
            or self.hf_processor is None
            or not hasattr(self.hf_processor, "tokenizer")
        ):
            logger.error(
                "VllmDecodeWorker: hf_processor or its tokenizer not initialized!"
            )
            raise RuntimeError("hf_processor or tokenizer not initialized.")

        text_prompt_string = self.hf_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        logger.info(
            f"Formatted text_prompt_string (request {request_id}):\\n{text_prompt_string}"
        )

        # Constants for token manipulation
        # For LLaVA-NeXT-Video models, the video token ID is 32000, not 32001
        # 32001 is for image tokens in LLaVA-NeXT-Video, 32000 is for video tokens
        VIDEO_TOKEN_ID_FOR_EXPANSION = 32000
        DUMMY_TOKEN_ID = 0

        # Variables to be set based on processing path
        prompt_argument_for_vllm: Union[str, TokensPrompt]
        current_received_multimodal_data_tensor: Optional[torch.Tensor] = None
        current_remote_prefill_params: Optional[RemotePrefillParams] = None
        multi_modal_data_for_engine: Optional[dict] = None

        if self.do_remote_prefill:
            logger.info(f"Disaggregated mode: request {{ id: {request_id} }}.")
            # Tokenize the prompt string to get base IDs for router length check and potential remote prefill manipulation
            tokenized_output_for_router = self.hf_processor.tokenizer(
                text_prompt_string, add_special_tokens=True
            )
            base_prompt_ids_for_router = tokenized_output_for_router.input_ids
            if (
                isinstance(base_prompt_ids_for_router, list)
                and len(base_prompt_ids_for_router) > 0
                and isinstance(base_prompt_ids_for_router[0], list)
                and len(base_prompt_ids_for_router) == 1
            ):
                base_prompt_ids_for_router = base_prompt_ids_for_router[0]
            logger.info(
                f"Base prompt_ids for router (request {request_id}, length {len(base_prompt_ids_for_router)}): {base_prompt_ids_for_router}"
            )

            should_prefill_remotely_decision = True
            if self.disaggregated_router:
                async with PrefillQueue.get_instance(
                    nats_server=self._prefill_queue_nats_server,
                    stream_name=self._prefill_queue_stream_name,
                ) as prefill_queue:
                    prefill_queue_size = await prefill_queue.get_queue_size()
                should_prefill_remotely_decision = (
                    await self.disaggregated_router.prefill_remote(
                        len(base_prompt_ids_for_router),
                        request.prefix_hit_rate,
                        prefill_queue_size,
                    )
                )

            if should_prefill_remotely_decision:
                logger.info(
                    f"Disaggregated: Prefilling REMOTELY for request {{ id: {request_id} }} (orig prompt len {len(base_prompt_ids_for_router)})"
                )
                current_remote_prefill_params = RemotePrefillParams(
                    is_remote_prefill=True,
                    remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
                    multimodal_data_source={"video_url": image_url},
                )
                num_dummies = self.embedding_size - 1
                # For remote prefill, expand the *single* video token from base_prompt_ids and add dummies
                expanded_and_dummied_ids = self._expand_video_tokens_in_prompt(
                    base_prompt_ids_for_router,  # Use the tokenized output of chat_template
                    NUM_SAMPLED_FRAMES,
                    VIDEO_TOKEN_ID_FOR_EXPANSION,
                    add_dummy_tokens=True,
                    dummy_token_id=DUMMY_TOKEN_ID,
                    num_dummy_tokens_per_frame=num_dummies,
                )
                prompt_argument_for_vllm = TokensPrompt(
                    prompt_token_ids=expanded_and_dummied_ids, multi_modal_data=None
                )
                logger.info(
                    f"REMOTE prefill: using TokensPrompt with {NUM_SAMPLED_FRAMES} video tokens (ID {VIDEO_TOKEN_ID_FOR_EXPANSION}), {num_dummies} dummies each. Expanded length: {len(expanded_and_dummied_ids)}."
                )
                multi_modal_data_for_engine = None  # Handled by prefill worker
            else:  # Local prefill in disaggregated mode
                logger.info(
                    f"Disaggregated: Prefilling LOCALLY for request {{ id: {request_id} }} (orig prompt len {len(base_prompt_ids_for_router)})"
                )
                raw_frames_tensor_from_nixl, desc = self._frames_descriptor
                with self._connector.create_writable(desc) as writable:
                    enc_req = EncodeRequest(
                        request_id=request_id,
                        image_url=image_url,
                        serialized_request=writable.to_serialized(),
                    )
                    logger.info(
                        f"Local prefill (disagg) - Encode request: {enc_req.model_dump_json()}"
                    )
                    async for enc_resp in await self.encode_worker_client.round_robin(
                        enc_req.model_dump_json()
                    ):
                        logger.info(
                            f"Local prefill (disagg) - Enc resp: {EncodeResponse.model_validate_json(enc_resp.data()).request_id}"
                        )
                    await writable.wait_for_completion()
                current_received_multimodal_data_tensor = raw_frames_tensor_from_nixl
                video_numpy = current_received_multimodal_data_tensor.cpu().numpy()
                multi_modal_data_for_engine = {"video": video_numpy}
                prompt_argument_for_vllm = text_prompt_string  # Pass raw string to vLLM
                logger.info(
                    f"LOCAL prefill (disagg): using raw prompt string for vLLM and {video_numpy.shape[0]} frames."
                )
                current_remote_prefill_params = None
        else:  # AGGREGATED MODE
            logger.info(
                f"Aggregated mode: request {{ id: {request_id} }}. Fetching frames directly."
            )
            raw_frames_tensor_from_nixl, desc = self._frames_descriptor
            with self._connector.create_writable(desc) as writable:
                enc_req = EncodeRequest(
                    request_id=request_id,
                    image_url=image_url,
                    serialized_request=writable.to_serialized(),
                )
                logger.info(f"Aggregated - Encode request: {enc_req.model_dump_json()}")
                async for enc_resp in await self.encode_worker_client.round_robin(
                    enc_req.model_dump_json()
                ):
                    logger.info(
                        f"Aggregated - Enc resp: {EncodeResponse.model_validate_json(enc_resp.data()).request_id}"
                    )
                await writable.wait_for_completion()
            current_received_multimodal_data_tensor = raw_frames_tensor_from_nixl
            video_numpy = current_received_multimodal_data_tensor.cpu().numpy()
            multi_modal_data_for_engine = {"video": video_numpy}
            prompt_argument_for_vllm = text_prompt_string  # Pass raw string to vLLM
            logger.info(
                f"AGGREGATED mode: using raw prompt string for vLLM and {video_numpy.shape[0]} frames."
            )
            current_remote_prefill_params = None

        request.sampling_params.output_kind = RequestOutputKind.DELTA

        logger.info(
            f"Final prompt for vLLM engine (request {request_id}): Type: {type(prompt_argument_for_vllm)}"
        )
        if isinstance(prompt_argument_for_vllm, str):
            logger.info(f"  Prompt string: {prompt_argument_for_vllm[:200]}...")
        elif isinstance(
            prompt_argument_for_vllm, dict
        ):  # Handles TokensPrompt which is a TypedDict
            if "prompt_token_ids" in prompt_argument_for_vllm:
                logger.info(
                    f"  Prompt token IDs (from dict): {str(prompt_argument_for_vllm.get('prompt_token_ids'))[:100]}..."
                )
            else:
                logger.info(
                    f"  Prompt is a dict (no prompt_token_ids key): {str(prompt_argument_for_vllm)[:200]}..."
                )
        else:
            logger.warning(
                f"  Prompt is of an unexpected type: {type(prompt_argument_for_vllm)}"
            )

        # Logging multi_modal_data_for_engine (populated for local/aggregated)
        if multi_modal_data_for_engine and "video" in multi_modal_data_for_engine:
            logger.info(
                f"  Multi_modal_data_for_engine['video'] type: {type(multi_modal_data_for_engine['video'])}, shape: {multi_modal_data_for_engine['video'].shape}"
            )
        elif not self.do_remote_prefill or (
            self.do_remote_prefill and not should_prefill_remotely_decision
        ):  # only log if it was expected
            logger.info(f"  Multi_modal_data_for_engine: {multi_modal_data_for_engine}")

        logger.info(
            f"  Remote_prefill_params active: {current_remote_prefill_params is not None}"
        )

        # Prepare the first argument for vLLM engine's generate call
        final_vllm_input: Union[str, dict]
        if isinstance(prompt_argument_for_vllm, str):
            if multi_modal_data_for_engine:
                final_vllm_input = {
                    "prompt": prompt_argument_for_vllm,
                    "multi_modal_data": multi_modal_data_for_engine,
                }
                logger.info(
                    "Constructed dict input for vLLM: prompt string + multi_modal_data"
                )
            else:
                final_vllm_input = prompt_argument_for_vllm
                logger.warning(
                    "Passing raw string to vLLM generate without multi_modal_data, though data was expected for local/aggregated path."
                )
        elif isinstance(
            prompt_argument_for_vllm, dict
        ):  # This handles the TokensPrompt case
            final_vllm_input = prompt_argument_for_vllm
            logger.info(
                "Passing dict (originally TokensPrompt) directly to vLLM generate."
            )
        else:
            logger.error(
                f"Unexpected type for prompt_argument_for_vllm: {type(prompt_argument_for_vllm)}"
            )
            raise TypeError("Invalid type for vLLM prompt argument.")

        sampling_params_dict = None
        if request.sampling_params:
            if hasattr(request.sampling_params, "model_dump"):
                sampling_params_dict = request.sampling_params.model_dump()
            elif hasattr(request.sampling_params, "dict"):
                sampling_params_dict = request.sampling_params.dict()
            else:
                try:
                    sampling_params_dict = vars(request.sampling_params)
                except (
                    TypeError
                ):  # vars() doesn't work on all objects, e.g. if __slots__ is used extensively
                    sampling_params_dict = str(
                        request.sampling_params
                    )  # Fallback to string representation

        remote_prefill_params_str = "None"
        if current_remote_prefill_params:
            if hasattr(current_remote_prefill_params, "model_dump_json"):
                remote_prefill_params_str = (
                    current_remote_prefill_params.model_dump_json()
                )
            elif hasattr(current_remote_prefill_params, "model_dump"):
                try:
                    remote_prefill_params_str = json.dumps(
                        current_remote_prefill_params.model_dump()
                    )
                except (
                    Exception
                ):  # Handle potential issues with json.dumps on complex objects
                    remote_prefill_params_str = str(
                        current_remote_prefill_params
                    )  # Fallback
            elif hasattr(current_remote_prefill_params, "dict"):
                try:
                    remote_prefill_params_str = json.dumps(
                        current_remote_prefill_params.dict()
                    )
                except Exception:
                    remote_prefill_params_str = str(
                        current_remote_prefill_params
                    )  # Fallback
            else:
                remote_prefill_params_str = str(
                    current_remote_prefill_params
                )  # Fallback to basic string representation

        logger.info(
            f"Calling VllmDecodeWorker.engine_client.generate for request {request_id} with:"
            f"  final_vllm_input type: {type(final_vllm_input)},"
            f"  sampling_params: {sampling_params_dict},"
            f"  remote_prefill_params: {remote_prefill_params_str}"
        )

        async for response in self.engine_client.generate(
            final_vllm_input,  # This is now the prompts argument (str, dict, or TokensPrompt)
            sampling_params=request.sampling_params,
            request_id=request.request_id,
            remote_prefill_params=current_remote_prefill_params,
            # multi_modal_data kwarg is removed as it's part of final_vllm_input if needed
        ):
            logger.info(
                f"Yeilding response {{ id: {response.request_id}, prompt: '{response.prompt}' }}"
            )

            processed_prompt_logprobs = _convert_logprobs_to_cpu_serializable(
                response.prompt_logprobs
            )

            serializable_outputs = []
            for vllm_out_item in response.outputs:
                new_item_logprobs = _convert_logprobs_to_cpu_serializable(
                    vllm_out_item.logprobs
                )

                # vllm.outputs.CompletionOutput is a dataclass.
                # Use dataclasses.replace to create a new instance with updated logprobs.
                new_out_item = dataclasses.replace(
                    vllm_out_item, logprobs=new_item_logprobs
                )

                serializable_outputs.append(new_out_item)

            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=processed_prompt_logprobs,
                outputs=serializable_outputs,  # Use the processed list
                finished=response.finished,
            ).model_dump_json()
