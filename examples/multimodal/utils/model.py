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

import torch
from transformers import AutoConfig, AutoImageProcessor
from vllm import AsyncEngineArgs
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.worker import Worker


def load_vision_model(model_id: str) -> torch.nn.Module:
    """
    Load a vision model from a HuggingFace model ID.
    """
    engine_args = AsyncEngineArgs(model=model_id, trust_remote_code=True)

    engine_config = engine_args.create_engine_config()
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    worker = Worker(
        vllm_config=engine_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=True,
    )
    # Initialize the worker.
    worker.init_device()
    worker.load_model()
    return worker.model_runner.model


def get_vision_embedding_size(model_id: str) -> int:
    """Calculate vision embedding size using model config and image processor"""
    # 1. Get image dimensions from processor
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    # Handle different processor formats (CLIP, Align, etc.)
    if hasattr(image_processor, "size"):
        size_info = image_processor.size
        if isinstance(size_info, dict):
            image_size = size_info.get("height", size_info["shortest_edge"])
        else:
            image_size = size_info  # Single integer value
    else:
        raise ValueError(f"Image size not found in processor for {model_id}")

    # 2. Get patch dimensions from model config
    config = AutoConfig.from_pretrained(model_id)

    # Handle different config structures (LLaVA, Qwen-VL, Phi-3V)
    vision_config = getattr(config, "vision_config", config)
    patch_size = getattr(vision_config, "patch_size", None)

    if not patch_size:
        # Fallback for models using spatial/temporal patches (e.g., video)
        patch_size = getattr(vision_config, "spatial_patch_size", 14)

    # 3. Calculate grid dimensions
    if isinstance(image_size, (list, tuple)):
        h, w = image_size[:2]
    else:
        h = w = image_size

    if isinstance(patch_size, (list, tuple)):
        ph, pw = patch_size[:2]
    else:
        ph = pw = patch_size

    num_patches = (h // ph) * (w // pw)

    return num_patches
