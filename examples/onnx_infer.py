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
from onnxruntime import OrtValue, IOBinding # Import necessary classes
import torch.utils.dlpack
import time # Import the time module
# Import numpy for averaging - Added
import numpy as np

print(f"ONNX Runtime version in use: {ort.__version__}") # Add version print

# --- Configuration ---
# Path to the directory containing the exported ONNX models
ONNX_MODEL_DIR = "./llava_onnx_encoder"
VISION_TOWER_ONNX_PATH = os.path.join(ONNX_MODEL_DIR, "llava_vision_tower.onnx")
PROJECTOR_ONNX_PATH = os.path.join(ONNX_MODEL_DIR, "llava_projector.onnx")

# Path to the original model files (needed to load processor for params)
ORIGINAL_MODEL_PATH = "/tmp/host/llava-1.5-7b-hf"

# Image to run inference on
IMAGE_SOURCE = "view.jpg" # Use local image

# Use "cuda" if GPU is available and configured, otherwise "cpu"
DEVICE = "cuda" # Assume GPU is always present
ORT_DEVICE = "cuda" if DEVICE == "cuda" else "cpu" # Device string for OrtValue should just be 'cuda'

NUM_RUNS = 10 # Number of runs: 1 warmup + 9 measured - Changed

# --- Setup Logging ---
# Lower logging level for repeated runs to avoid excessive output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Changed level back to INFO
logger = logging.getLogger(__name__)
# Set logger level explicitly if needed
# logger.setLevel(logging.INFO) # Keep INFO level for main execution steps

# --- Helper Functions ---

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
    """Loads the processor to extract image preprocessing parameters."""
    try:
        logger.info(f"Loading processor from {model_path} to get preprocessing parameters...")
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
        raise

def preprocess_image(image: Image.Image, params: dict) -> np.ndarray:
    """ Replicates image preprocessing. """
    logger.debug("Starting image preprocessing...")
    target_size = params["size"]
    crop_h, crop_w = params["crop_size"]
    rescale_factor = params["rescale_factor"]
    mean = params["image_mean"]
    std = params["image_std"]
    target_dtype = np.float16 if DEVICE == 'cuda' else np.float32

    img_w, img_h = image.size
    if img_w < img_h:
        new_w = target_size
        new_h = int(target_size * img_h / img_w)
    else:
        new_h = target_size
        new_w = int(target_size * img_w / img_h)
    image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    logger.debug(f"Resized to: ({new_w}, {new_h})")

    left = (new_w - crop_w) / 2
    top = (new_h - crop_h) / 2
    right = (new_w + crop_w) / 2
    bottom = (new_h + crop_h) / 2
    image = image.crop((left, top, right, bottom))
    logger.debug(f"Center cropped to: ({crop_w}, {crop_h})")

    img_array = (np.array(image).astype(np.float32) * rescale_factor)
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(target_dtype)

    logger.debug(f"Preprocessing complete. Output shape: {img_array.shape}, dtype: {img_array.dtype}")
    return img_array

# --- Global variables for sessions ---
vision_sess = None
proj_sess = None
vision_input_name = None
vision_output_name = None
proj_input_name = None
proj_output_name = None
vision_output_type = None
proj_output_type = None
vision_output_meta = None
proj_output_meta = None


def initialize_sessions():
    """Initializes ONNX sessions globally."""
    global vision_sess, proj_sess, vision_input_name, vision_output_name, proj_input_name, proj_output_name
    global vision_output_type, proj_output_type, vision_output_meta, proj_output_meta

    if vision_sess is not None and proj_sess is not None:
        logger.info("ONNX sessions already initialized.")
        return

    logger.info("Initializing ONNX inference sessions globally...")
    try:
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0, 'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
                'trt_engine_cache_path': os.path.join(ONNX_MODEL_DIR, "trt_cache"),
            }),
            ('CUDAExecutionProvider', {'device_id': 0}),
            'CPUExecutionProvider'
        ]
        trt_cache_path = os.path.join(ONNX_MODEL_DIR, "trt_cache")
        if not os.path.exists(trt_cache_path):
            os.makedirs(trt_cache_path)
            logger.info(f"Created TensorRT engine cache directory: {trt_cache_path}")

        logger.info(f"Attempting to load sessions with providers: {providers}")
        vision_sess = ort.InferenceSession(VISION_TOWER_ONNX_PATH, providers=providers)
        logger.info(f"Loaded Vision Tower. Effective providers: {vision_sess.get_providers()}")
        proj_sess = ort.InferenceSession(PROJECTOR_ONNX_PATH, providers=providers)
        logger.info(f"Loaded Projector. Effective providers: {proj_sess.get_providers()}")

        vision_input_meta = vision_sess.get_inputs()[0]
        vision_output_meta = vision_sess.get_outputs()[0]
        proj_input_meta = proj_sess.get_inputs()[0]
        proj_output_meta = proj_sess.get_outputs()[0]

        vision_input_name = vision_input_meta.name
        vision_output_name = vision_output_meta.name
        proj_input_name = proj_input_meta.name
        proj_output_name = proj_output_meta.name

        type_map = {"tensor(float16)": np.float16, "tensor(float)": np.float32}
        vision_output_type = type_map.get(vision_output_meta.type, np.float32)
        proj_output_type = type_map.get(proj_output_meta.type, np.float32)

        logger.info(f"Vision Tower I/O: Input='{vision_input_name}', Output='{vision_output_name}'")
        logger.info(f"Projector I/O: Input='{proj_input_name}', Output='{proj_output_name}'")

    except Exception as e:
        logger.error(f"Failed to initialize ONNX sessions: {e}")
        raise

def run_single_inference(preprocessed_image_ortvalue: OrtValue):
    """ Runs single inference pass, returns OrtValue and time. """
    global vision_sess, proj_sess, vision_input_name, vision_output_name, proj_input_name, proj_output_name
    global vision_output_type, proj_output_type, vision_output_meta, proj_output_meta

    if vision_sess is None or proj_sess is None:
        raise RuntimeError("ONNX sessions not initialized.")

    start_time = time.perf_counter()

    # --- Vision Tower ---
    try:
        vision_output_shape = [1 if isinstance(d, str) else d for d in vision_output_meta.shape]
        vision_output_ortvalue = OrtValue.ortvalue_from_shape_and_type(vision_output_shape, vision_output_type, ORT_DEVICE)

        io_binding_vision = vision_sess.io_binding()
        io_binding_vision.bind_ortvalue_input(vision_input_name, preprocessed_image_ortvalue)
        io_binding_vision.bind_ortvalue_output(vision_output_name, vision_output_ortvalue)

        vision_sess.run_with_iobinding(io_binding_vision)
    except Exception as e:
        logger.error(f"ONNX Vision Tower inference failed: {e}")
        raise

    # --- Projector ---
    try:
        proj_output_shape = [1 if isinstance(d, str) else d for d in proj_output_meta.shape]
        proj_output_shape[1] = vision_output_ortvalue.shape()[1] # Dynamic seq len
        final_embeddings_ortvalue = OrtValue.ortvalue_from_shape_and_type(proj_output_shape, proj_output_type, ORT_DEVICE)

        io_binding_proj = proj_sess.io_binding()
        io_binding_proj.bind_ortvalue_input(proj_input_name, vision_output_ortvalue)
        io_binding_proj.bind_ortvalue_output(proj_output_name, final_embeddings_ortvalue)

        proj_sess.run_with_iobinding(io_binding_proj)
    except Exception as e:
        logger.error(f"ONNX Projector inference failed: {e}")
        raise

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    logger.debug(f"Single ONNX inference time: {inference_time:.4f} seconds")

    # Clean up intermediate tensor explicitly if needed (though IOBinding reuses buffers)
    del vision_output_ortvalue

    return final_embeddings_ortvalue, inference_time


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting ONNX inference process...")

    if not os.path.exists(VISION_TOWER_ONNX_PATH) or not os.path.exists(PROJECTOR_ONNX_PATH):
        logger.error(f"ONNX model files not found in {ONNX_MODEL_DIR}. Please run convert.py first.")
    else:
        try:
            # --- Initialize models and preprocessing once ---
            initialize_sessions()
            preprocessing_params = get_preprocessing_params(ORIGINAL_MODEL_PATH)

            # --- Preprocess image once ---
            logger.info(f"Loading and preprocessing image: {IMAGE_SOURCE}")
            image = load_image(IMAGE_SOURCE)
            preprocessed_image_np = preprocess_image(image, preprocessing_params)
            preprocessed_image_ortvalue = OrtValue.ortvalue_from_numpy(preprocessed_image_np, ORT_DEVICE)
            logger.info(f"Input image OrtValue created on device: {preprocessed_image_ortvalue.device_name()}")

            # --- Run inference multiple times ---
            inference_times = [] # List to store times of runs 2-10 - Added
            final_embeddings_ortvalue = None # To store the result of the last run

            logger.info(f"Running ONNX inference {NUM_RUNS} times (1 warmup + {NUM_RUNS-1} measured)...") # Updated log
            for i in range(NUM_RUNS):
                logger.info(f"Starting ONNX inference run {i+1}/{NUM_RUNS}...")
                current_embeddings_ortvalue, current_inference_time = run_single_inference(preprocessed_image_ortvalue)

                # --- Logic for warmup and measured runs - Added ---
                if i == 0: # First run is warmup
                    logger.info(f"Completed ONNX warmup run {i+1}. Time: {current_inference_time:.4f} seconds (discarded)")
                    # Discard result explicitly
                    del current_embeddings_ortvalue
                else: # Runs 2 through NUM_RUNS (indices 1 to 9)
                    inference_times.append(current_inference_time)
                    logger.info(f"Completed ONNX measured run {i+1}. Time: {current_inference_time:.4f} seconds")
                    if i == NUM_RUNS - 1: # If it's the very last run (index 9)
                        final_embeddings_ortvalue = current_embeddings_ortvalue # Keep the result OrtValue
                    else:
                         # Discard intermediate results if necessary (though less critical with OrtValue reuse)
                         del current_embeddings_ortvalue
                # --- End logic ---


            # --- Calculate and report average time - Added ---
            if len(inference_times) > 0:
                 average_inference_time = np.mean(inference_times)
                 print(f"\n--- Performance (Average of Last {len(inference_times)} ONNX Runs) ---")
                 print(f"Average inference time: {average_inference_time:.4f} seconds")
                 # Optionally print std deviation
                 std_dev_inference_time = np.std(inference_times)
                 print(f"Standard deviation: {std_dev_inference_time:.4f} seconds")
            else:
                 print("\n--- Performance ---")
                 logger.warning("No inference times recorded (Requires NUM_RUNS > 1).")
            # --- End average calculation ---


            # --- Report results from the very last run ---
            # This section now uses final_embeddings_ortvalue from the 10th run
            if final_embeddings_ortvalue is not None:
                logger.info("Processing results from the last ONNX inference run.")
                if final_embeddings_ortvalue.device_name() == 'cuda':
                    logger.info("Copying final embeddings from GPU to CPU...")
                    embeddings_np = final_embeddings_ortvalue.numpy()
                else:
                     embeddings_np = final_embeddings_ortvalue.numpy()

                print("\n--- Final Embeddings (from Last ONNX Run OrtValue) ---")
                print(f"Device: {final_embeddings_ortvalue.device_name()}")
                print(f"GPU Address (data_ptr): {final_embeddings_ortvalue.data_ptr()}")
                print(f"Shape: {embeddings_np.shape}")
                print(f"Data type: {embeddings_np.dtype}")
                num_elements = np.prod(embeddings_np.shape)
                element_size_bytes = embeddings_np.itemsize
                total_size_bytes = num_elements * element_size_bytes
                print(f"Total Size (bytes): {total_size_bytes}")
                print("First few values of the first embedding vector:")
                print(embeddings_np[0, 0, :5])

                # JSON output (optional)
                # ... (kept same)

                # Verification (optional)
                # ... (kept same)

                 # Clean up final OrtValue if desired
                del final_embeddings_ortvalue

        except Exception as e:
            logger.error(f"An error occurred during the overall ONNX process: {e}", exc_info=True) # Add traceback