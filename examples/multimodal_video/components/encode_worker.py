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
import base64
import binascii
import json
import logging
import os
from io import BytesIO
from queue import Queue
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

import av
import connect
import httpx
import numpy as np
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from utils.protocol import EncodeRequest
from utils.vllm import parse_vllm_args

from dynamo.sdk import async_on_start, endpoint, service

logger = logging.getLogger(__name__)

CACHE_SIZE_MAXIMUM = 8


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmEncodeWorker:
    def __init__(self) -> None:
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.MODEL_ID = self.engine_args.model

        self.video_processor = LlavaNextVideoProcessor.from_pretrained(self.MODEL_ID)
        self.video_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
        logger.info(
            f"Model {self.MODEL_ID} loaded on device: {self.video_model.device}"
        )

        self._video_content_cache: dict[str, BytesIO] = {}
        self._cache_queue: Queue[str] = Queue(maxsize=CACHE_SIZE_MAXIMUM)

        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_timeout = 60.0

        self.num_frames_to_sample = 8

    async def _read_video_pyav(
        self, container: av.container.InputContainer, indices: np.ndarray
    ) -> np.ndarray:
        """
        Decode the video with PyAV decoder. Async wrapper.
        """

        def blocking_decode():
            container.seek(0)  # Reset container for decoding
            processed_indices = set(indices)

            # Determine min/max index to optimize decoding loop slightly
            min_idx = 0
            max_idx = -1
            if len(indices) > 0:
                min_idx = np.min(indices)
                max_idx = np.max(indices)

            if (
                not processed_indices
                and container.streams.video
                and container.streams.video[0].frames > 0
            ):
                logger.warning(
                    "_read_video_pyav called with empty indices for a non-empty video, attempting to read first frame."
                )
                try:
                    frame = next(container.decode(video=0))
                    return np.stack([frame.to_ndarray(format="rgb24")])
                except StopIteration:
                    logger.error(
                        "Failed to read even the first frame despite non-empty indices check."
                    )
                    return np.array([])

            decoded_frames_list = []
            for i, frame in enumerate(container.decode(video=0)):
                if i > max_idx and max_idx != -1:  # max_idx is -1 if indices is empty
                    break
                if i >= min_idx and i in processed_indices:
                    decoded_frames_list.append(frame)

            if not decoded_frames_list and len(processed_indices) > 0:
                actual_decoded_count = 0
                try:
                    container.seek(0)  # Reset for counting
                    for _ in container.decode(video=0):
                        actual_decoded_count += 1
                except Exception:  # Handle cases where re-decoding/counting fails
                    pass  # Keep original error message
                raise ValueError(
                    f"Could not decode any frames for the given indices: {indices.tolist()}. "
                    f"Video might be shorter than expected or indices out of bounds. "
                    f"Actual decodable frames in container (approx): {actual_decoded_count}."
                )

            return (
                np.stack([x.to_ndarray(format="rgb24") for x in decoded_frames_list])
                if decoded_frames_list
                else np.array([])
            )

        return await asyncio.to_thread(blocking_decode)

    async def _load_video_content(self, video_url: str) -> BytesIO:
        parsed_url = urlparse(video_url)
        video_url_lower = video_url.lower()

        if parsed_url.scheme in ("http", "https"):
            if video_url_lower in self._video_content_cache:
                logger.info(f"Video content found in cache for URL: {video_url}")
                cached_content = self._video_content_cache[video_url_lower]
                cached_content.seek(0)
                return cached_content

        try:
            video_data: BytesIO
            if parsed_url.scheme == "data":
                if not parsed_url.path.startswith(
                    ("video/", "application/octet-stream")
                ):
                    raise ValueError("Data URL must be a video type or octet-stream")

                media_type_and_data = parsed_url.path.split(",", 1)
                if len(media_type_and_data) != 2:
                    raise ValueError("Invalid Data URL format: missing comma separator")

                media_type, data_segment = media_type_and_data
                if ";base64" not in media_type:
                    raise ValueError("Video Data URL currently must be base64 encoded")

                try:
                    video_bytes = base64.b64decode(data_segment)
                    video_data = BytesIO(video_bytes)
                except binascii.Error as e:
                    raise ValueError(f"Invalid base64 encoding for video data: {e}")

            elif parsed_url.scheme in ("http", "https"):
                if not self._http_client:
                    await self._init_http_client()

                logger.info(f"Downloading video from URL: {video_url}")
                response = await self._http_client.get(
                    video_url, timeout=self._http_timeout
                )
                response.raise_for_status()

                if not response.content:
                    raise ValueError(
                        f"Empty response content from video URL: {video_url}"
                    )
                video_data = BytesIO(response.content)
                logger.info(
                    f"Video downloaded from {video_url}, size: {len(response.content)} bytes."
                )

            elif parsed_url.scheme == "file" or not parsed_url.scheme:
                file_path = parsed_url.path if parsed_url.scheme else video_url
                # Ensure path is absolute or resolve relative to a known base if necessary
                # For simplicity, assuming it's an accessible path.
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Local video file not found: {file_path}")

                logger.info(f"Reading local video file: {file_path}")
                with open(file_path, "rb") as f:
                    video_bytes = f.read()
                video_data = BytesIO(video_bytes)
                logger.info(
                    f"Local video {file_path} read, size: {len(video_bytes)} bytes."
                )
            else:
                raise ValueError(
                    f"Unsupported video source scheme: {parsed_url.scheme} for URL {video_url}"
                )

            if parsed_url.scheme in (
                "http",
                "https",
            ):  # Cache successfully downloaded content
                if self._cache_queue.full():
                    oldest_url = self._cache_queue.get_nowait()
                    if oldest_url in self._video_content_cache:
                        del self._video_content_cache[oldest_url]

                # Store the BytesIO object directly; it will be seek(0)'d when retrieved
                self._video_content_cache[video_url_lower] = video_data
                self._cache_queue.put(video_url_lower)
                video_data.seek(0)  # Ensure it's ready for the first consumer (av.open)

            return video_data

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error {e.response.status_code} loading video {video_url}: {e.response.text[:200]}"
            )
            raise ValueError(
                f"Failed to download video {video_url}: HTTP {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Request error loading video {video_url}: {e}")
            raise ValueError(f"Network request failed for video {video_url}") from e
        except FileNotFoundError as e:
            logger.error(f"File error loading video {video_url}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Error loading video content from {video_url}: {type(e).__name__} - {e}"
            )
            raise ValueError(f"Failed to load video content: {e}")

    @endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[str]:
        request_id = request.request_id
        video_url = getattr(request, "video_url", getattr(request, "image_url", None))
        if not video_url:
            logger.error(
                f"Request {request_id}: 'video_url' or 'image_url' not provided."
            )
            raise ValueError("'video_url' or 'image_url' is required for encoding.")

        prompt_text = getattr(request, "prompt_text", "Describe this video in detail.")

        logger.info(
            f"Received encode request: {{ id: {request_id}, video_url: '{video_url[:100]}...' }}"
        )

        container: Optional[av.container.InputContainer] = None
        try:
            video_content_stream = await self._load_video_content(video_url)

            def open_video_container_sync():
                try:
                    return av.open(video_content_stream, mode="r")
                except av.AVError as ave:
                    logger.error(
                        f"PyAV error opening video stream from {video_url}: {ave}"
                    )
                    raise ValueError(
                        f"Invalid video format or corrupted data from {video_url}."
                    ) from ave
                except Exception as e:
                    logger.error(
                        f"Unexpected error opening video stream from {video_url} with PyAV: {e}"
                    )
                    raise ValueError(
                        f"Unexpected error opening video from {video_url}."
                    ) from e

            container = await asyncio.to_thread(open_video_container_sync)

            if not container.streams.video:
                logger.error(f"No video stream found in {video_url}.")
                raise ValueError(f"No video stream in {video_url}.")

            stream_info = container.streams.video[0]
            total_frames = stream_info.frames
            # Duration can be useful for streams where total_frames is 0
            duration_sec = (
                stream_info.duration * float(stream_info.time_base)
                if stream_info.duration
                else 0
            )

            if total_frames == 0 and duration_sec == 0:
                logger.error(f"Video file '{video_url}' has 0 frames and 0 duration.")
                raise ValueError(f"Video {video_url} has 0 frames and 0 duration.")
            if total_frames == 0 and duration_sec > 0:
                logger.warning(
                    f"Video {video_url} reports 0 frames but has duration {duration_sec:.2f}s. Frame sampling may be based on requested count directly."
                )

            logger.info(
                f"Video {video_url} has {total_frames} frames (duration: {duration_sec:.2f}s). Sampling {self.num_frames_to_sample} frames."
            )
            indices: np.ndarray
            if total_frames > 0:
                if total_frames < self.num_frames_to_sample:
                    logger.warning(
                        f"Video frames ({total_frames}) < samples ({self.num_frames_to_sample}). Using all available."
                    )
                    indices = np.arange(0, total_frames).astype(int)
                else:
                    indices = np.linspace(
                        0, total_frames - 1, self.num_frames_to_sample, dtype=int
                    )
                indices = np.unique(indices)
                if (
                    len(indices) == 0 and total_frames > 0
                ):  # Safety for linspace oddities with few frames
                    indices = (
                        np.array([0])
                        if total_frames == 1
                        else np.arange(
                            0, min(self.num_frames_to_sample, total_frames)
                        ).astype(int)
                    )

            else:  # total_frames is 0 (likely a stream), sample by count
                logger.warning(
                    f"Video {video_url} frame count is 0. Sampling {self.num_frames_to_sample} frames by index."
                )
                indices = np.arange(0, self.num_frames_to_sample).astype(int)

            logger.info(f"Selected frame indices for {video_url}: {indices.tolist()}")

            clip_np: np.ndarray = await self._read_video_pyav(container, indices)

            if clip_np.size == 0:
                raise ValueError(
                    f"Failed to extract any video frames from {video_url} for indices {indices.tolist()}. Clip is empty."
                )

            logger.info(
                f"Successfully extracted {len(clip_np) if clip_np.ndim > 1 and clip_np.shape[0] > 0 else 0} frames for {video_url}."
            )

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "video"},
                    ],
                }
            ]
            logger.info(
                f"Applying chat template for request {request_id} with prompt: '{prompt_text}'"
            )
            prompt_for_processor = self.video_processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            processed_inputs = None
            if clip_np.size > 0:  # We have video frames
                logger.info(
                    f"Req {request_id}: Processing text and {clip_np.shape[0]} video frames."
                )
                processed_inputs = self.video_processor(
                    text=prompt_for_processor,
                    videos=[clip_np],
                    padding=True,
                    return_tensors="pt",
                )
            else:
                raise ValueError(
                    f"Request {request_id}: Video clip is unexpectedly empty despite requesting {self.num_frames_to_sample} frames."
                )

            for key, value in processed_inputs.items():
                if isinstance(value, torch.Tensor):
                    processed_inputs[key] = value.to(self.video_model.device)
            logger.info(
                f"Req {request_id}: Processor output tensor devices: {{ {', '.join([f'{k}: {v.device}' for k, v in processed_inputs.items() if isinstance(v, torch.Tensor)])} }}"
            )

            pixel_values_for_rdma = processed_inputs.pop(
                "pixel_values_videos", None
            )  # Changed key from 'pixel_values'

            if pixel_values_for_rdma is None:
                # This is unexpected if num_frames_to_sample=8 and clip_np was valid.
                logger.error(
                    f"Req {request_id}: 'pixel_values_videos' key not found in processor output or was None. This is unexpected when video frames are processed."
                )
                raise ValueError(
                    f"Request {request_id}: Failed to obtain 'pixel_values_videos' from video processor output. Expected with {self.num_frames_to_sample} frames."
                )

            auxiliary_payload_dict = {}
            for (
                key,
                value,
            ) in (
                processed_inputs.items()
            ):  # Iterate over remaining items (input_ids, attention_mask)
                if isinstance(value, torch.Tensor):
                    # Convert tensors to lists for JSON serialization, move to CPU first.
                    auxiliary_payload_dict[key] = value.cpu().tolist()
                else:
                    # For any other non-tensor data (shouldn't be much from processor normally)
                    auxiliary_payload_dict[key] = value

            serialized_auxiliary_payload = json.dumps(auxiliary_payload_dict)

            # Since an error is raised if pixel_values_for_rdma is None,
            # tensor_for_descriptor will always be the actual pixel_values.
            # Ensure the tensor is contiguous and cast to the model's expected dtype.
            tensor_for_descriptor: torch.Tensor = pixel_values_for_rdma.to(
                dtype=self.video_model.dtype
            ).contiguous()
            logger.info(
                f"Req {request_id}: Preparing pixel_values (shape: {tensor_for_descriptor.shape}, "
                f"dtype: {tensor_for_descriptor.dtype}, device: {tensor_for_descriptor.device}, "
                f"contiguous: {tensor_for_descriptor.is_contiguous()}) for RDMA."
            )

            logger.info(
                f"Req {request_id}: Auxiliary payload for next stage: {serialized_auxiliary_payload[:250]}..."
            )

            descriptor = connect.Descriptor(tensor_for_descriptor)
            logger.info(f"Req {request_id}: Beginning connector write operation.")
            if request.serialized_request is None:
                logger.error(
                    f"Request serialized_request is None for request: {{ id: {request_id} }}."
                )
            # Pass the DecodeWorker's SerializedRequest (representing its WritableOperation) to begin_write.
            write_op = await self._connector.begin_write(
                descriptor, request.serialized_request
            )
            await write_op.wait_for_completion()
            logger.info(f"Req {request_id}: Connector write operation completed.")

            # Yield a dict containing request_id and the auxiliary payload, as DecodeWorker expects this.
            final_response_data = {
                "request_id": request.request_id,
                "serialized_auxiliary_payload": serialized_auxiliary_payload,
            }
            yield json.dumps(final_response_data)
            logger.info(f"Encode request {request_id} processed successfully.")

        except (
            FileNotFoundError,
            av.error.AVError,
            ValueError,
        ) as e:  # av.AVError might need to be av.error.AVError
            logger.error(
                f"Error processing request {request_id} ({video_url[:100]}...): {type(e).__name__} - {e}"
            )
            raise  # Re-raise to be handled by the service framework
        except Exception as e:
            logger.exception(
                f"Unexpected error processing request {request_id} ({video_url[:100]}...): {e}"
            )
            raise
        finally:
            if container:
                await asyncio.to_thread(container.close)

    async def _init_http_client(self):
        if (
            not self._http_client or self._http_client.is_closed
        ):  # Check if closed as well
            self._http_client = httpx.AsyncClient(timeout=self._http_timeout)
            logger.info("HTTP client (re)initialized.")

    @async_on_start
    async def async_init(self):
        logger.info(f"{self.__class__.__name__} async_init started.")
        self._connector = connect.Connector()
        await self._connector.initialize()
        logger.info("Dynamo connector initialized.")
        await self._init_http_client()
        logger.info(
            f"{self.__class__.__name__} async_init completed. Model: {self.MODEL_ID} on {self.video_model.device}."
        )
