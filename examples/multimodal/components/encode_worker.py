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
from io import BytesIO
import base64
from queue import Queue
from typing import AsyncIterator

import connect
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, LlavaForConditionalGeneration
from utils.protocol import EncodeRequest, EncodeResponse
from utils.vllm import parse_vllm_args

from dynamo.sdk import async_on_start, endpoint, service

logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"

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

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )

        self.vision_model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID, device_map="auto", torch_dtype=torch.float16
        ).eval()

        self._image_cache: dict[str, Image.Image] = {}
        self._cache_queue: Queue[str] = Queue(maxsize=CACHE_SIZE_MAXIMUM)

    @endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
        logger.debug(
            f"Received encode request: {{ id: {request.request_id} }}."
        )

        request_id = request.request_id

        # The following steps encode the requested image and provided useful embeddings.
        # 1. Open the image from the provided URL.
        # 2. Process the image using the image processor.
        # 3. Run the image through the vision model's vision tower.
        # 4. Run the results of the vision tower through the multi-modal projector.
        # 5. Create a descriptor for the embeddings.
        # 6. Create a write operation using the serialized request and the descriptor.
        # 7. Await for the write operation to complete.
        # 8. Yield the encode response.

        # Either retrieve the image from the cache or download it and then cache it.

        # Only cache for url images.
        if request.image_url.startswith("data:image/"):
            image = self.open_image(request.image_url)
        elif request.image_url.startswith("http") or request.image_url.startswith("https"):
            if request.image_url in self._image_cache:
                image_url = request.image_url.lower()
                image = self._image_cache[image_url]
                logger.debug(
                    f"Image found in cache for request: {{ id: {request_id} }}."
                )
            else:
                image = self.open_image(request.image_url)
                logger.debug(
                    f"Downloading/opening image for request: {{ id: {request_id} }}."
                )
                # Cache the image for future use, and evict the oldest image if the cache is full.
                if self._cache_queue.full():
                    oldest_image_url = self._cache_queue.get()
                    del self._image_cache[oldest_image_url]

                self._image_cache[request.image_url] = image
                self._cache_queue.put(request.image_url)
        else:
            raise ValueError(f"Invalid image source: {request.image_url}")

        logger.debug(
            f"Processing image for request: {{ id: {request_id} }}"
        )
        image_embeds = self.image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            logger.debug(f"Vision model device: {self.vision_model.device}")
            vision_outputs = self.vision_model.vision_tower(
                image_embeds["pixel_values"].to(self.vision_model.device)
            )
            logger.debug("Vision model completed.")

            embeddings = vision_outputs.last_hidden_state
            embeddings = self.vision_model.multi_modal_projector(embeddings)

            logger.debug(
                f"Embeddings: {{ shape: {embeddings.shape}, dtype: {embeddings.dtype}, device: {embeddings.device}, ptr: {embeddings.data_ptr()}, elements: {{ count: {embeddings.numel()}, size: {embeddings.element_size()} }} }}."
            )

            if request.serialized_request is None:
                logger.error(
                    f"Request serialized_request is None for request: {{ id: {request_id} }}."
                )

            # Create a descriptor for the embeddings, this will register the memory with the connector (and the NIXL runtime).
            descriptor = connect.Descriptor(embeddings)
            # Create a write operation using the serialized request and the descriptor.
            # This will begin the RDMA transfer of the embeddings to the remote worker.
            write_op = await self._connector.begin_write(
                descriptor,
                request.serialized_request,
            )
            # Await for the write operation to complete.
            # This will block until the data has been written to the remote worker or an error occurs.
            await write_op.wait_for_completion()

            yield EncodeResponse(
                request_id=request.request_id,
            ).model_dump_json()

    @async_on_start
    async def async_init(self):
        logger.info("Startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        await self._connector.initialize()
        logger.info("Startup completed.")

    def open_image(self, image_url: str) -> Image.Image:
        # TODO: Have a seperate field for url and non url - and avoid auto detection
        try:
            if image_url.startswith("data:image/"):
                # Remove the data URL prefix to get just the base64 string
                base64_data = image_url.split(",", 1)[1]
                try:
                    image_bytes = base64.standard_b64decode(base64_data)
                    image_data = BytesIO(image_bytes)
                except base64.binascii.Error as e:
                    raise ValueError(f"Invalid base64 encoding: {e}")
            elif image_url.startswith("http") or image_url.startswith("https"):
                response = requests.get(image_url)
                response.raise_for_status()  # Raise an exception for bad status codes

                if not response.content:
                    raise ValueError("Empty response content from image URL")
                image_data = BytesIO(response.content)

            image_data = Image.open(image_data).convert("RGB")

            return image_data
        except Exception as e:
            logger.error(f"Error opening image: {e}")
            raise e
