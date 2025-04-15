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

from functools import lru_cache
import logging
from typing import AsyncIterator, Dict, Any
from functools import lru_cache

from io import BytesIO
from PIL import Image
from transformers import AutoImageProcessor
import requests
import torch
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.worker import Worker
from vllm.config import VllmConfig

from utils.protocol import EncodeRequest, EncodeResponse
from vllm import AsyncEngineArgs
from dynamo.sdk import depends, dynamo_endpoint, service
from transformers import LlavaProcessor, LlavaForConditionalGeneration

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class EncodeWorker:
    # worker = depends(VllmWorker)

    def __init__(self) -> None:
        self.MODEL_ID = "llava-hf/llava-1.5-7b-hf"
        # self.MAX_TOKENS = 4096

        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.mm_projector_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # torch.cuda.set_device(self.device)
        # torch.cuda.empty_cache()
        # self.engine_args = AsyncEngineArgs(model=self.MODEL_ID, trust_remote_code=True,
        #                                    max_num_seqs=15, max_model_len=self.MAX_TOKENS,
        #                                    device=self.device, limit_mm_per_prompt={"image": 10})

        self.image_processor = AutoImageProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
    
        # self.processor = LlavaProcessor.from_pretrained(
        #     self.MODEL_ID,
        #     use_fast=True  # Suppresses the slow processor warning
        # )
        self.vision_model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            device_map="auto",  # Automatically uses GPU if available
            torch_dtype=torch.float16
        ).eval()
    
    # @lru_cache(maxsize=5)
    # def load_multi_modal_projector(self) -> Any:
    #     self.engine_args.device = self.mm_projector_device
    #     engine_config = self.engine_args.create_engine_config()
    #     distributed_init_method = get_distributed_init_method(
    #         get_ip(), get_open_port())
    #     vllm_config = VllmConfig(
    #         model_config=engine_config.model_config,
    #         parallel_config=engine_config.parallel_config,
    #         scheduler_config=engine_config.scheduler_config,
    #         device_config=engine_config.device_config,
    #         cache_config=engine_config.cache_config,
    #         load_config=engine_config.load_config,
    #     )
    #     worker = Worker(
    #         vllm_config=vllm_config,
    #         local_rank=0,
    #         rank=0,
    #         distributed_init_method=distributed_init_method,
    #         is_driver_worker=True,
    #     )
    #     # Initialize the worker.
    #     worker.init_device()
    #     worker.load_model()

    #     print("Multimodal projector loaded.")
    #     return worker.model_runner.model.vision_embed_tokens



    # def multi_modal_project(self,
    #                         image_embeds: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     mm_projector = self.load_multi_modal_projector()
        
    #     pixel_values = image_embeds['pixel_values'].to(self.mm_projector_device)
    #     image_sizes = image_embeds['image_sizes'].to(self.mm_projector_device)
    #     with torch.no_grad():
    #         image_features_proj = mm_projector(pixel_values, image_sizes)[0]
    #     print("image_features_proj", image_features_proj)
    #     print(image_features_proj.shape)
    #     # tensors don't play nice when going from GPU to GPU; they can only be preserved by moving to CPU in the midst
    #     # https://discuss.pytorch.org/t/tensor-totally-changes-when-allocating-moving-from-gpu-to-gpu/73930/15
    #     return image_features_proj.to('cpu').to(self.device)


    @dynamo_endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
        image_embeds = self.encode_image(request.image_url)
        print("type of image_embeds: ", type(image_embeds))

        # image_input = [self.multi_modal_project(image_embeds)]

        # Convert tensor to list
        yield EncodeResponse(image_features=image_embeds.tolist()).model_dump_json()


    def open_image(self, image: str) -> Image.Image:
        if image.startswith('http') or image.startswith('https'):
            print("Image is a URL, downloading...")
            response = requests.get(image)
            print(f"Downloaded image, response status: {response.status_code}")
            image_data = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            print(f"Loading image from file path: {image}")
            image_data = Image.open(image).convert('RGB')
        return image_data


    def encode_image(self, image: str) -> Dict[str, torch.Tensor]:
        image = self.open_image(image)
        # image_embeds = self.image_processor(images=image, return_tensors="pt")
        inputs = self.image_processor(images=image, return_tensors="pt")

        # Generate proper image embeddings
        with torch.no_grad():
            vision_outputs = self.vision_model.vision_tower(
                inputs.pixel_values.to(self.vision_model.device)
            )
            image_features = vision_outputs.last_hidden_state
            image_embeds = self.vision_model.multi_modal_projector(image_features)
        print(f"Processed image, input tensor shape: {image_embeds.shape}")

        return image_embeds



    

