import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import logging
import torch
from transformers import LlavaProcessor # Needed to get preprocessing details
import json

# --- Configuration ---
# Path to the directory containing the exported ONNX models
ONNX_MODEL_DIR = "./llava_onnx_encoder"
VISION_TOWER_ONNX_PATH = os.path.join(ONNX_MODEL_DIR, "llava_vision_tower.onnx")
PROJECTOR_ONNX_PATH = os.path.join(ONNX_MODEL_DIR, "llava_projector.onnx")

# Path to the original model files (needed to load processor for params)
# You might need to adjust this if it's different from convert.py's path
ORIGINAL_MODEL_PATH = "/tmp/host/llava-1.5-7b-hf"

# Image to run inference on
# IMAGE_SOURCE = "https://llava-vl.github.io/static/images/view.jpg"
IMAGE_SOURCE = "view.jpg" # Use local image

# Use "cuda" if GPU is available and configured, otherwise "cpu"
DEVICE = "cuda" # Assume GPU is always present

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def load_image(image_path_or_url: str) -> Image.Image:
    """Loads an image from a URL or local file path."""
    try:
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            logger.info(f"Loaded image from URL: {image_path_or_url}")
        else:
            if not os.path.exists(image_path_or_url):
                 logger.error(f"Local image not found: {image_path_or_url}")
                 raise FileNotFoundError(f"Image not found at {image_path_or_url}")
            image = Image.open(image_path_or_url).convert("RGB")
            logger.info(f"Loaded image from path: {image_path_or_url}")
        return image
    except Exception as e:
        logger.error(f"Error loading image '{image_path_or_url}': {e}")
        raise

def get_preprocessing_params(model_path: str):
    """Loads the processor to extract image preprocessing parameters."""
    try:
        logger.info(f"Loading processor from {model_path} to get preprocessing parameters...")
        # Load processor only to access its image_processor's config
        processor = LlavaProcessor.from_pretrained(model_path)
        img_processor_config = processor.image_processor # Access the CLIPImageProcessor part

        # Determine the target dtype based on the device used for export
        target_dtype = np.float16 if DEVICE == 'cuda' else np.float32
        logger.info(f"Using target dtype for preprocessing arrays: {target_dtype}")

        params = {
            "size": img_processor_config.size["shortest_edge"], # e.g., 336
            "crop_size": (img_processor_config.crop_size["height"], img_processor_config.crop_size["width"]), # e.g., (336, 336)
            "rescale_factor": img_processor_config.rescale_factor, # e.g., 1/255.0
            # Load mean and std with the target dtype
            "image_mean": np.array(img_processor_config.image_mean, dtype=target_dtype),
            "image_std": np.array(img_processor_config.image_std, dtype=target_dtype)
        }
        logger.info(f"Extracted preprocessing parameters: {params}")
        return params
    except Exception as e:
        logger.error(f"Failed to load processor or extract params from {model_path}: {e}")
        raise

def preprocess_image(image: Image.Image, params: dict) -> np.ndarray:
    """
    Replicates the image preprocessing using PIL and NumPy based on extracted params.
    NOTE: This is a simplified replication. CLIPImageProcessor might have subtle
          details (e.g., exact interpolation modes). Verify if precision is critical.
    """
    logger.info("Starting image preprocessing...")
    target_size = params["size"]          # Shortest edge size
    crop_h, crop_w = params["crop_size"]
    rescale_factor = params["rescale_factor"]
    mean = params["image_mean"]
    std = params["image_std"]

    # Determine the target dtype based on the device used for export
    target_dtype = np.float16 if DEVICE == 'cuda' else np.float32

    # 1. Resize (shortest edge) - PIL default is BICUBIC (resample=3)
    img_w, img_h = image.size
    if img_w < img_h:
        new_w = target_size
        new_h = int(target_size * img_h / img_w)
    else:
        new_h = target_size
        new_w = int(target_size * img_w / img_h)
    image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    logger.debug(f"Resized to: ({new_w}, {new_h})")

    # 2. Center Crop
    left = (new_w - crop_w) / 2
    top = (new_h - crop_h) / 2
    right = (new_w + crop_w) / 2
    bottom = (new_h + crop_h) / 2
    image = image.crop((left, top, right, bottom))
    logger.debug(f"Center cropped to: ({crop_w}, {crop_h})")

    # 3. Convert to NumPy array (H, W, C), rescale, and set target dtype
    # Perform calculations potentially in float32 for intermediate precision
    img_array = (np.array(image).astype(np.float32) * rescale_factor)

    # 4. Normalize (using potentially float16 mean/std now)
    img_array = (img_array - mean) / std

    # 5. Transpose to (C, H, W) format expected by many models
    img_array = img_array.transpose(2, 0, 1)

    # 6. Add batch dimension (1, C, H, W)
    img_array = np.expand_dims(img_array, axis=0)

    # 7. Ensure final array has the target dtype
    img_array = img_array.astype(target_dtype)

    logger.info(f"Preprocessing complete. Output shape: {img_array.shape}, dtype: {img_array.dtype}")
    return img_array

def run_inference(image_source: str, params: dict):
    """Loads ONNX models, preprocesses image, and runs inference."""

    # --- 1. Load ONNX Sessions ---
    logger.info("Loading ONNX inference sessions...")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
        logger.info(f"Using providers: {providers}")

        vision_sess = ort.InferenceSession(VISION_TOWER_ONNX_PATH, providers=providers)
        logger.info(f"Loaded Vision Tower from {VISION_TOWER_ONNX_PATH}. Providers: {vision_sess.get_providers()}")
        proj_sess = ort.InferenceSession(PROJECTOR_ONNX_PATH, providers=providers)
        logger.info(f"Loaded Projector from {PROJECTOR_ONNX_PATH}. Providers: {proj_sess.get_providers()}")

        # Get input/output names (essential for feeding data)
        vision_input_name = vision_sess.get_inputs()[0].name
        vision_output_name = vision_sess.get_outputs()[0].name # Assumes first output is needed
        proj_input_name = proj_sess.get_inputs()[0].name
        proj_output_name = proj_sess.get_outputs()[0].name

        logger.info(f"Vision Tower I/O names: Input='{vision_input_name}', Output='{vision_output_name}'")
        logger.info(f"Projector I/O names: Input='{proj_input_name}', Output='{proj_output_name}'")

    except Exception as e:
        logger.error(f"Failed to load ONNX models: {e}")
        return None

    # --- 2. Load and Preprocess Image ---
    try:
        dummy_image_source = "https://llava-vl.github.io/static/images/view.jpg"
        image = load_image(dummy_image_source)
        preprocessed_image = preprocess_image(image, params)
    except Exception as e:
        logger.error(f"Failed during image loading or preprocessing: {e}")
        return None

    # --- 3. Run Vision Tower Inference ---
    logger.info("Running ONNX Vision Tower inference...")
    try:
        vision_inputs = {vision_input_name: preprocessed_image}
        vision_outputs = vision_sess.run([vision_output_name], vision_inputs)
        vision_features = vision_outputs[0] # Output corresponding to 'hidden_states'
        logger.info(f"Vision Tower ONNX output shape: {vision_features.shape}")
    except Exception as e:
        logger.error(f"ONNX Vision Tower inference failed: {e}")
        return None

    # --- 4. Run Projector Inference ---
    logger.info("Running ONNX Projector inference...")
    try:
        proj_inputs = {proj_input_name: vision_features} # Use vision tower output
        proj_outputs = proj_sess.run([proj_output_name], proj_inputs)
        final_embeddings = proj_outputs[0]
        logger.info(f"Projector ONNX output shape (final embeddings): {final_embeddings.shape}")
    except Exception as e:
        logger.error(f"ONNX Projector inference failed: {e}")
        return None

    return final_embeddings


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting ONNX inference process...")

    if not os.path.exists(VISION_TOWER_ONNX_PATH) or not os.path.exists(PROJECTOR_ONNX_PATH):
        logger.error(f"ONNX model files not found in {ONNX_MODEL_DIR}. Please run convert.py first.")
    else:
        try:
            # Get necessary preprocessing info from the original processor
            preprocessing_params = get_preprocessing_params(ORIGINAL_MODEL_PATH)

            # Run the full inference pipeline
            embeddings = run_inference(IMAGE_SOURCE, preprocessing_params)

            if embeddings is not None:
                logger.info("Inference finished successfully.")
                # You can now use the 'embeddings' NumPy array
                # Example: print shape and first few values
                print("\n--- Final Embeddings ---")
                print(f"Shape: {embeddings.shape}")
                print(f"Data type: {embeddings.dtype}")
                print("First few values of the first embedding vector:")
                print(embeddings[0, 0, :5]) # Print first 5 values of the first token embedding

                # --- Add JSON Output ---
                print("\n--- Final Embeddings (JSON) ---")
                try:
                    embeddings_list = embeddings.tolist() # Convert NumPy array to Python list
                    embeddings_json = json.dumps(embeddings_list, indent=2) # Serialize list to JSON string with indentation
                    print(embeddings_json)
                except Exception as e:
                    logger.error(f"Failed to convert embeddings to JSON: {e}")
                # --- End JSON Output ---

        except Exception as e:
            logger.error(f"An error occurred during the inference process: {e}")
