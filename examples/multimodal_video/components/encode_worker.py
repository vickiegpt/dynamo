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

import cv2
import logging
from io import BytesIO
from typing import AsyncIterator, List
import numpy as np

import requests
import torch
from PIL import Image
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

from utils.protocol import EncodeRequest, EncodeResponse
from utils.vllm import parse_vllm_args

from dynamo.sdk import dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class EncodeWorker:
    def __init__(self) -> None:
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        # self.MODEL_ID = self.engine_args.model # Not strictly needed if only sampling frames

        # The model and processor are not needed if EncodeWorker only samples frames
        # and vLLM handles the actual processing.
        # self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        #     self.MODEL_ID, 
        #     torch_dtype=torch.float16, 
        #     device_map="auto"
        # )
        # self.processor = LlavaNextVideoProcessor.from_pretrained(self.MODEL_ID)
        logger.info(f"{class_name} initialized to sample frames.")

    def sample_frames(self, video_path: str, num_frames: int = 8) -> List[np.ndarray]:
        """Sample frames from a video file or URL.
        
        Args:
            video_path: Path to video file or URL
            num_frames: Number of frames to sample
            
        Returns:
            List of RGB frames as numpy arrays
        """
        try:
            if video_path.startswith(("http://", "https://")):
                # Download video from URL
                response = requests.get(video_path)
                video_data = BytesIO(response.content)
                cap = cv2.VideoCapture()
                cap.open(video_data)
            else:
                # Open local video file
                cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"Video has no frames: {video_path}")
            
            # Calculate frame indices to sample
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    logger.warning(f"Failed to read frame at index {idx}")
            
            cap.release()
            
            if len(frames) != num_frames:
                logger.warning(f"Could only sample {len(frames)} frames out of {num_frames} requested")
            
            return frames
            
        except Exception as e:
            logger.error(f"Error sampling frames from video: {e}")
            raise

    # def encode_frames(self, frames: List[np.ndarray]) -> torch.Tensor: # This method is no longer used as we pass raw frames
    #     """Encode video frames using the vision encoder.
        
    #     Args:
    #         frames: List of RGB frames as numpy arrays
            
    #     Returns:
    #         Tensor of frame embeddings
    #     """
    #     try:
    #         # Convert frames to PIL Images
    #         images = [Image.fromarray(f) for f in frames]
            
    #         # Process frames through the processor
    #         inputs = self.processor(
    #             text="",  # Empty text as we only need video embeddings
    #             videos=[images],  # Pass frames as video input
    #             return_tensors="pt"
    #         )
            
    #         # Move inputs to the same device as the model
    #         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
    #         return inputs["pixel_values_videos"] # Return the direct output of the processor
            
    #     except Exception as e:
    #         logger.error(f"Error encoding frames: {e}")
    #         raise

    # def aggregate_embeddings(self, frame_embeddings: torch.Tensor, method: str = 'mean') -> torch.Tensor: # No longer used
    #     """Aggregate frame embeddings into a single video embedding.
        
    #     Args:
    #         frame_embeddings: Tensor of frame embeddings
    #         method: Aggregation method ('mean' or 'max')
            
    #     Returns:
    #         Single video embedding tensor
    #     """
    #     if method == 'mean':
    #         return frame_embeddings.mean(dim=0)
    #     elif method == 'max':
    #         return frame_embeddings.max(dim=0).values
    #     else:
    #         raise ValueError(f"Unknown aggregation method: {method}")

    @dynamo_endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
        """Sample frames from a video.
        
        Args:
            request: EncodeRequest containing video URL or path
            
        Yields:
            EncodeResponse containing raw video frames
        """
        try:
            # Sample frames from video
            # sample_frames returns a list of np.ndarray (H, W, C) uint8
            sampled_frames_np_list = self.sample_frames(request.image_url, num_frames=request.num_frames or 8) # Use num_frames from request
            
            # Convert List[np.ndarray] to List[List[List[List[int]]]] for JSON serialization
            # This creates a list of frames, where each frame is a list of rows,
            # each row is a list of pixels, and each pixel is a list of [R,G,B] values.
            raw_frames_for_json = [frame.tolist() for frame in sampled_frames_np_list]
            
            logger.info(f"Sampled {len(raw_frames_for_json)} frames. Shape of first frame (if any): {sampled_frames_np_list[0].shape if sampled_frames_np_list else 'N/A'}")
            
            yield EncodeResponse(
                raw_frames=raw_frames_for_json
            ).model_dump_json()
            
        except Exception as e:
            logger.error(f"Error in EncodeWorker encode (frame sampling): {e}")
            raise

