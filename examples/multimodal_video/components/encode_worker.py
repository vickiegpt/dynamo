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
        self.MODEL_ID = self.engine_args.model

        # Load the model in half-precision
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.MODEL_ID, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.MODEL_ID)

        # self.image_processor = AutoImageProcessor.from_pretrained(
        #     self.MODEL_ID, trust_remote_code=True
        # )

        # self.vision_model = LlavaForConditionalGeneration.from_pretrained(
        #     self.MODEL_ID, device_map="auto", torch_dtype=torch.float16
        # ).eval()

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

    def encode_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Encode video frames using the vision encoder.
        
        Args:
            frames: List of RGB frames as numpy arrays
            
        Returns:
            Tensor of frame embeddings
        """
        try:
            # Convert frames to PIL Images
            images = [Image.fromarray(f) for f in frames]
            
            # Process frames through the processor
            inputs = self.processor(
                text="",  # Empty text as we only need video embeddings
                videos=[images],  # Pass frames as video input
                return_tensors="pt"
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Get frame embeddings using the vision tower
            with torch.no_grad():
                # Get the vision tower outputs
                logger.info(f"Inputs['pixel_values_videos'].shape: {inputs['pixel_values_videos'].shape}")
                
                # Ensure correct input shape for vision tower
                pixel_values = inputs["pixel_values_videos"]
                if len(pixel_values.shape) == 5:  # (batch, frames, channels, height, width)
                    pixel_values = pixel_values.squeeze(0)  # Remove batch dimension
                
                vision_outputs = self.model.vision_tower(pixel_values)
                
                # Extract the frame embeddings from the last hidden state
                frame_embeddings = vision_outputs.last_hidden_state
                logger.info(f"Frame embeddings shape: {frame_embeddings.shape}")
                
                # Project embeddings through multi-modal projector
                frame_embeddings = self.model.multi_modal_projector(frame_embeddings)
                logger.info(f"Frame embeddings shape after projector: {frame_embeddings.shape}")
                
            return frame_embeddings
            
        except Exception as e:
            logger.error(f"Error encoding frames: {e}")
            raise

    def aggregate_embeddings(self, frame_embeddings: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """Aggregate frame embeddings into a single video embedding.
        
        Args:
            frame_embeddings: Tensor of frame embeddings
            method: Aggregation method ('mean' or 'max')
            
        Returns:
            Single video embedding tensor
        """
        if method == 'mean':
            return frame_embeddings.mean(dim=0)
        elif method == 'max':
            return frame_embeddings.max(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    @dynamo_endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
        """Encode a video into embeddings.
        
        Args:
            request: EncodeRequest containing video URL or path
            
        Yields:
            EncodeResponse containing video embeddings
        """
        try:
            # Sample frames from video
            frames = self.sample_frames(request.image_url)
            
            # Encode frames
            frame_embeddings = self.encode_frames(frames)
            
            # Aggregate frame embeddings into video embedding
            # Mean pool across frames
            video_embedding = frame_embeddings.mean(dim=0)  # Average across frames
            video_embedding = video_embedding.unsqueeze(0)  # Add batch dimension
            
            logger.info(f"Final video embedding shape: {video_embedding.shape}")
            
            yield EncodeResponse(
                video_features=video_embedding.tolist()
            ).model_dump_json()
            
        except Exception as e:
            logger.error(f"Error in encode: {e}")
            raise

