import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import requests
from io import BytesIO
import os
import logging
import time
import numpy as np # Import numpy for averaging

# --- Configuration ---
# Path to the original model files
ORIGINAL_MODEL_PATH = "/tmp/host/llava-1.5-7b-hf"

# Image to run inference on
IMAGE_SOURCE = "view.jpg" # Use local image (same as ONNX script)
# IMAGE_SOURCE = "https://llava-vl.github.io/static/images/view.jpg"

# Use "cuda" if GPU is available, otherwise "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_RUNS = 10 # Number of times to run inference (run 1 for warmup)

# --- Setup Logging ---
# Keep logging level concise for multiple runs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables for model and processor ---
model = None
processor = None

def open_image(image_path_or_url: str) -> Image.Image:
    """Loads an image from a URL or local file path."""
    try:
        if image_path_or_url.startswith("http"):
            logger.debug(f"Fetching image from URL: {image_path_or_url}")
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image_data = Image.open(BytesIO(response.content)).convert("RGB")
            logger.debug(f"Loaded image from URL.")
        else:
            if not os.path.exists(image_path_or_url):
                 logger.error(f"Local image not found: {image_path_or_url}")
                 raise FileNotFoundError(f"Image not found at {image_path_or_url}")
            logger.debug(f"Opening local image: {image_path_or_url}")
            image_data = Image.open(image_path_or_url).convert("RGB")
            logger.debug(f"Loaded image from path.")
        return image_data
    except Exception as e:
        logger.error(f"Error opening image '{image_path_or_url}': {e}")
        raise

def initialize_model_and_processor():
    """Initializes the PyTorch model and processor globally."""
    global model, processor
    if model is not None and processor is not None:
        logger.info("PyTorch model and processor already initialized.")
        return

    logger.info(f"Initializing PyTorch model and processor from: {ORIGINAL_MODEL_PATH}")
    try:
        # Load the full LLaVA model
        model = LlavaForConditionalGeneration.from_pretrained(
            ORIGINAL_MODEL_PATH,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32 # Use float16 on GPU
        )
        model.to(DEVICE) # Move model to the target device
        model.eval() # Set model to evaluation mode

        # Load the processor
        processor = LlavaProcessor.from_pretrained(ORIGINAL_MODEL_PATH)

        logger.info(f"PyTorch model and processor loaded successfully on device: {DEVICE}")

    except Exception as e:
        logger.error(f"Failed to initialize PyTorch model or processor: {e}")
        raise

def run_single_pytorch_encoding(image: Image.Image):
    """
    Runs a single pass of PyTorch-based image encoding.
    Times only the model execution part.
    """
    global model, processor

    if model is None or processor is None:
        raise RuntimeError("Model and processor not initialized. Call initialize_model_and_processor() first.")

    # logger.debug("Starting PyTorch encoding run...") # Removed

    try:
        # Preprocess image - provide dummy text input
        dummy_text = " "
        inputs = processor(text=dummy_text, images=image, return_tensors="pt")
        start_time = time.perf_counter() # <<<--- TIMER START MOVED HERE

        # Move input tensors to the target device BEFORE starting the timer
        pixel_values = inputs['pixel_values'].to(DEVICE, dtype=model.dtype)
        # logger.debug(f"Input pixel values moved to device: {pixel_values.device}") # Removed

        # Synchronize GPU before starting timer for accurate measurement (optional but recommended for GPU)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()


        # Run inference without calculating gradients
        with torch.no_grad():
            # Pass pixel values through the vision tower
            vision_outputs = model.vision_tower(
                pixel_values
            )

            # Check output structure
            if vision_outputs is None or not hasattr(vision_outputs, 'last_hidden_state') or vision_outputs.last_hidden_state is None:
                 # Keep error logs
                 logger.error("Could not retrieve 'last_hidden_state' from vision_outputs.")
                 raise AttributeError("Vision tower output structure mismatch or missing last_hidden_state.")

            image_features = vision_outputs.last_hidden_state
            # logger.debug(f"Vision tower output shape: {image_features.shape}") # Removed


            # Pass features through the multi-modal projector
            image_features = model.multi_modal_projector(image_features)
            # logger.debug(f"Projector output shape: {image_features.shape}") # Removed


        # Synchronize GPU before stopping timer for accurate measurement (optional but recommended for GPU)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter() # <<<--- TIMER END REMAINS HERE

    except Exception as e:
        logger.exception(f"Error during PyTorch inference: {e}") # Keep exception logs
        raise

    inference_time = end_time - start_time
    logger.debug(f"Single PyTorch encoding time: {inference_time:.4f} seconds")

    # Return the features tensor (still on the device) and the time
    return image_features, inference_time

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting PyTorch inference process...")

    if not os.path.exists(ORIGINAL_MODEL_PATH):
        logger.error(f"Original model directory not found: {ORIGINAL_MODEL_PATH}")
    else:
        try:
            # --- Initialize model and processor once ---
            initialize_model_and_processor()

            # --- Load image once ---
            logger.info(f"Loading image: {IMAGE_SOURCE}")
            image = open_image(IMAGE_SOURCE)

            # --- Run inference multiple times ---
            inference_times = [] # List to store times of runs 2-10
            final_embeddings_tensor = None # To store the result of the last run

            logger.info(f"Running PyTorch inference {NUM_RUNS} times (1 warmup + {NUM_RUNS-1} measured)...")
            for i in range(NUM_RUNS):
                logger.info(f"Starting PyTorch inference run {i+1}/{NUM_RUNS}...")
                current_embeddings_tensor, current_inference_time = run_single_pytorch_encoding(image)

                if i == 0: # First run is warmup
                    logger.info(f"Completed PyTorch warmup run {i+1}. Time: {current_inference_time:.4f} seconds (discarded)")
                    # Discard tensor from warmup run
                    del current_embeddings_tensor
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
                else: # Runs 2 through NUM_RUNS
                    inference_times.append(current_inference_time)
                    logger.info(f"Completed PyTorch measured run {i+1}. Time: {current_inference_time:.4f} seconds")
                    if i == NUM_RUNS - 1: # If it's the very last run
                        final_embeddings_tensor = current_embeddings_tensor # Keep the result tensor
                    else:
                        # Discard tensor from intermediate runs if memory is a concern
                        del current_embeddings_tensor
                        if DEVICE == 'cuda':
                            torch.cuda.empty_cache()


            # --- Calculate and report average time ---
            if len(inference_times) > 0:
                 average_inference_time = np.mean(inference_times)
                 print(f"\n--- Performance (Average of Last {len(inference_times)} PyTorch Runs) ---")
                 print(f"Average inference time: {average_inference_time:.4f} seconds")
                 # Optionally print std deviation
                 std_dev_inference_time = np.std(inference_times)
                 print(f"Standard deviation: {std_dev_inference_time:.4f} seconds")
            else:
                 print("\n--- Performance ---")
                 logger.warning("No inference times recorded (Requires NUM_RUNS > 1).")


            # --- Report results from the very last run (for inspection) ---
            if final_embeddings_tensor is not None:
                logger.info("Processing results from the very last PyTorch inference run.")

                # Move tensor to CPU for inspection/saving if it's on GPU
                final_embeddings_tensor_cpu = final_embeddings_tensor.cpu()
                embeddings_np = final_embeddings_tensor_cpu.numpy() # Convert to NumPy

                print("\n--- Final Embeddings (from Last PyTorch Run) ---")
                print(f"Device (original tensor): {final_embeddings_tensor.device}")
                print(f"Shape: {embeddings_np.shape}")
                print(f"Data type: {embeddings_np.dtype}")
                num_elements = np.prod(embeddings_np.shape)
                element_size_bytes = embeddings_np.itemsize
                total_size_bytes = num_elements * element_size_bytes
                print(f"Total Size (bytes): {total_size_bytes}")
                print("First few values of the first embedding vector:")
                print(embeddings_np[0, 0, :5])

                # Clean up tensor explicitly
                del final_embeddings_tensor
                del final_embeddings_tensor_cpu
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"An error occurred during the PyTorch inference process: {e}")