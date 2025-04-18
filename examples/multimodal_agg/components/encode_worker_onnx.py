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
from typing import AsyncIterator
import os

import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from transformers import LlavaProcessor # Needed only during init for preprocessing params

from dynamo.sdk import depends, dynamo_endpoint, service
# Assuming protocol definitions are in utils relative to components
from utils.protocol import EncodeRequest, EncodeResponse

logger = logging.getLogger(__name__)

# --- Configuration --- (Adjust paths as needed for deployment)
ONNX_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "llava_onnx_encoder") # Relative path
VISION_TOWER_ONNX_PATH = os.path.join(ONNX_MODEL_DIR, "llava_vision_tower.onnx")
PROJECTOR_ONNX_PATH = os.path.join(ONNX_MODEL_DIR, "llava_projector.onnx")

# Path to original model files (needed *only* for init to get processor params)
# In a real deployment, you might hardcode these params or load from a config file
# instead of loading the original processor here.
ORIGINAL_MODEL_PATH = "/tmp/llava-1.5-7b-hf" # Or path where original files are accessible

# Use "cuda" if GPU is available and configured, otherwise "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" # Assume GPU is always present

# --- Helper Functions (Copied/Adapted from onnx_infer.py) ---

def load_image(image_path_or_url: str) -> Image.Image:
    """Loads an image from a URL or local file path."""
    try:
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            logger.debug(f"Loaded image from URL: {image_path_or_url}")
        else:
            if not os.path.exists(image_path_or_url):
                 logger.error(f"Local image not found: {image_path_or_url}")
                 raise FileNotFoundError(f"Image not found at {image_path_or_url}")
            image = Image.open(image_path_or_url).convert("RGB")
            logger.debug(f"Loaded image from path: {image_path_or_url}")
        return image
    except Exception as e:
        logger.error(f"Error loading image '{image_path_or_url}': {e}")
        raise

def get_preprocessing_params(model_path: str):
    """Loads the processor ONCE during init to extract image preprocessing parameters."""
    try:
        logger.info(f"Loading processor from {model_path} to get preprocessing parameters (init only)...")
        processor = LlavaProcessor.from_pretrained(model_path)
        img_processor_config = processor.image_processor
        target_dtype = np.float16 if DEVICE == 'cuda' else np.float32
        logger.info(f"Using target dtype for preprocessing arrays: {target_dtype}")
        params = {
            "size": img_processor_config.size["shortest_edge"],
            "crop_size": (img_processor_config.crop_size["height"], img_processor_config.crop_size["width"]),
            "rescale_factor": img_processor_config.rescale_factor,
            "image_mean": np.array(img_processor_config.image_mean, dtype=target_dtype),
            "image_std": np.array(img_processor_config.image_std, dtype=target_dtype)
        }
        logger.info(f"Extracted preprocessing parameters: {params}")
        return params
    except Exception as e:
        logger.error(f"Failed to load processor or extract params from {model_path}: {e}")
        raise RuntimeError(f"Could not initialize preprocessing parameters: {e}")

def preprocess_image(image: Image.Image, params: dict) -> np.ndarray:
    """Replicates the image preprocessing using PIL and NumPy based on extracted params."""
    logger.debug("Starting image preprocessing...")
    target_size = params["size"]
    crop_h, crop_w = params["crop_size"]
    rescale_factor = params["rescale_factor"]
    mean = params["image_mean"]
    std = params["image_std"]
    target_dtype = mean.dtype # Get dtype from loaded params

    # Resize
    img_w, img_h = image.size
    if img_w < img_h:
        new_w = target_size
        new_h = int(target_size * img_h / img_w)
    else:
        new_h = target_size
        new_w = int(target_size * img_w / img_h)
    image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    # Center Crop
    left = (new_w - crop_w) / 2
    top = (new_h - crop_h) / 2
    right = (new_w + crop_w) / 2
    bottom = (new_h + crop_h) / 2
    image = image.crop((left, top, right, bottom))

    # Convert, rescale, normalize
    img_array = (np.array(image).astype(np.float32) * rescale_factor)
    img_array = (img_array - mean.astype(np.float32)) / std.astype(np.float32) # Use float32 for stability

    # Transpose & Add batch dim
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Ensure final dtype
    img_array = img_array.astype(target_dtype)
    logger.debug(f"Preprocessing complete. Output shape: {img_array.shape}, dtype: {img_array.dtype}")
    return img_array

# --- Dynamo Service Definition ---
@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    # Adjust resources based on ONNX model needs & execution provider
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class EncodeWorker:

    def __init__(self) -> None:
        logger.info("Initializing EncodeWorkerONNX...")
        logger.info(f"Using device for provider selection: {DEVICE}")

        # --- Get Preprocessing Params (once during init) ---
        # IMPORTANT: In production, avoid loading the full processor here.
        #            Hardcode params or load from a separate config.
        self.preprocessing_params = get_preprocessing_params(ORIGINAL_MODEL_PATH)

        # --- Load ONNX Sessions ---
        logger.info("Loading ONNX inference sessions...")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
            logger.info(f"Attempting to load ONNX models with providers: {providers}")

            if not os.path.exists(VISION_TOWER_ONNX_PATH):
                raise FileNotFoundError(f"Vision tower ONNX model not found at {VISION_TOWER_ONNX_PATH}")
            if not os.path.exists(PROJECTOR_ONNX_PATH):
                raise FileNotFoundError(f"Projector ONNX model not found at {PROJECTOR_ONNX_PATH}")

            self.vision_sess = ort.InferenceSession(VISION_TOWER_ONNX_PATH, providers=providers)
            logger.info(f"Loaded Vision Tower. Effective providers: {self.vision_sess.get_providers()}")
            self.proj_sess = ort.InferenceSession(PROJECTOR_ONNX_PATH, providers=providers)
            logger.info(f"Loaded Projector. Effective providers: {self.proj_sess.get_providers()}")

            # Get input/output names
            self.vision_input_name = self.vision_sess.get_inputs()[0].name
            self.vision_output_name = self.vision_sess.get_outputs()[0].name
            self.proj_input_name = self.proj_sess.get_inputs()[0].name
            self.proj_output_name = self.proj_sess.get_outputs()[0].name
            logger.info("ONNX sessions and I/O names initialized.")

        except Exception as e:
            logger.error(f"Fatal error loading ONNX models during initialization: {e}", exc_info=True)
            # Reraise to prevent the worker from starting in a bad state
            raise RuntimeError(f"Failed to initialize ONNX sessions: {e}")

    def encode_image_onnx(self, image_path_or_url: str) -> np.ndarray:
        """Loads image, preprocesses, and runs ONNX inference."""
        try:
            # 1. Load Image
            image = load_image(image_path_or_url)

            # 2. Preprocess Image (using params stored during init)
            # The steps required to transform the raw image into the model-ready format (resizing, cropping, scaling, normalizing) were originally part of the Hugging Face LlavaProcessor. 
            # This logic is not automatically included within the exported .onnx files.
            preprocessed_image = preprocess_image(image, self.preprocessing_params)

            # 3. Run ONNX Vision Tower Inference
            logger.debug("Running ONNX Vision Tower inference...")
            vision_inputs = {self.vision_input_name: preprocessed_image}
            vision_outputs = self.vision_sess.run([self.vision_output_name], vision_inputs)
            # we take [0] because session.run() returns a list, 
            # and we configured the run to return a list containing only the single output tensor we are interested in (the vision features needed for the next step)
            vision_features = vision_outputs[0]
            logger.debug(f"ONNX Vision Tower output shape: {vision_features.shape}")

            # 4. Run ONNX Projector Inference
            logger.debug("Running ONNX Projector inference...")
            proj_inputs = {self.proj_input_name: vision_features}
            proj_outputs = self.proj_sess.run([self.proj_output_name], proj_inputs)
            # we use [0] to get the actual NumPy array containing the final embeddings from the list returned by the projector's ONNX Runtime session
            final_embeddings = proj_outputs[0]
            logger.debug(f"ONNX Projector output shape (final embeddings): {final_embeddings.shape}")

            return final_embeddings
        except Exception as e:
            logger.error(f"Error during ONNX image encoding for '{image_path_or_url}': {e}", exc_info=True)
            # Decide how to handle errors during a request - perhaps return empty or raise specific exception
            raise # Reraise for now

    @dynamo_endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
        """Dynamo endpoint to handle encoding requests using ONNX."""
        logger.info(f"Received encode request for image: {request.image_url}")
        try:
            # Perform encoding using the ONNX method
            image_embeds_np = self.encode_image_onnx(request.image_url)
            logger.info(f"ONNX encoding successful, embedding shape: {image_embeds_np.shape}")

            # Convert NumPy array to list for JSON serialization
            yield EncodeResponse(image_features=image_embeds_np.tolist()).model_dump_json()

        except Exception as e:
            # Log error and potentially yield an error response if protocol supports it
            logger.error(f"Failed to process encode request for {request.image_url}: {e}", exc_info=True)
            # Handle error appropriately, maybe yield an error indicator if possible
            # For now, the exception will likely terminate the stream/response
            pass # Or re-raise e
