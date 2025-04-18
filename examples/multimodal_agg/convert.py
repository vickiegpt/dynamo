import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import requests
from io import BytesIO
import os
import onnxruntime as ort
import numpy as np
import logging

# --- Configuration ---
# MODEL_ID = "llava-hf/llava-1.5-7b-hf" # Original Hub ID
MODEL_PATH = "/tmp/host/llava-1.5-7b-hf" # Path to the locally stored model files
# Use "cuda" if GPU is available and configured, otherwise "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" # Assume GPU is always present
# Use float16 for faster inference and lower memory if supported, else float32
# MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_DTYPE = torch.float16 # Since DEVICE is always cuda
# Directory to save the exported ONNX models
OUTPUT_DIR = "./llava_onnx_encoder"
# ONNX Opser version
OPSET_VERSION = 17 # Recommended is 16 or 17

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image(image_path_or_url: str) -> Image.Image:
    """Loads an image from a URL or local file path."""
    try:
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            logger.info(f"Loaded image from URL: {image_path_or_url}")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
            logger.info(f"Loaded image from path: {image_path_or_url}")
        return image
    except Exception as e:
        logger.error(f"Error loading image '{image_path_or_url}': {e}")
        raise

def convert_llava_vision_to_onnx():
    """Loads LLaVA, exports vision tower and projector to ONNX, and verifies."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Using dtype: {MODEL_DTYPE}")
    logger.info(f"ONNX output directory: {OUTPUT_DIR}")

    # --- 1. Load Model & Processor ---
    logger.info(f"Loading processor from local path '{MODEL_PATH}'...")
    try:
        # Load from the local directory instead of the Hub ID
        processor = LlavaProcessor.from_pretrained(MODEL_PATH)
        # --- Add logging --- 
        logger.info(f"Processor object loaded: {processor}")
        if processor is None:
             logger.error("Processor failed to load (returned None).")
             return
        # --- End logging ---
    except Exception as e:
        logger.error(f"Failed to load processor from {MODEL_PATH}: {e}")
        return

    logger.info(f"Loading model from local path '{MODEL_PATH}'...")
    try:
        # Load from the local directory instead of the Hub ID
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=MODEL_DTYPE,
            low_cpu_mem_usage=True, # Helps manage memory for large models
        ).to(DEVICE).eval() # Set to evaluation mode
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
        return
    logger.info("Model and processor loaded.")

    # --- 2. Access Vision Components ---
    try:
        vision_tower = model.vision_tower
        projector = model.multi_modal_projector
        logger.info("Identified vision_tower and multi_modal_projector.")
    except AttributeError as e:
        logger.error(f"Could not access vision components in the model: {e}")
        return

    # --- 3. Prepare Dummy Input ---
    # Use a standard image URL or a local path
    # dummy_image_source = "https://llava-vl.github.io/static/images/view.jpg"
    dummy_image_source = "coffee_2.JPG" # Use the local image if available
    if not os.path.exists(dummy_image_source):
         logger.warning(f"Local dummy image '{dummy_image_source}' not found. Trying URL.")
         dummy_image_source = "https://llava-vl.github.io/static/images/view.jpg"

    try:
        dummy_image = load_image(dummy_image_source)
        # --- Add logging ---
        if not isinstance(dummy_image, Image.Image):
             logger.error(f"load_image did not return a PIL Image object. Got: {type(dummy_image)}")
             return
        logger.info(f"Dummy image loaded successfully. Type: {type(dummy_image)}, Size: {dummy_image.size}")
        # --- End logging ---

        # --- Provide dummy text as well --- 
        dummy_text = "USER: <image>\nDescribe the image."
        logger.info(f"Using dummy text for processor: '{dummy_text}'")
        # --- End --- 

        # dummy_inputs_processed = processor(images=dummy_image, return_tensors="pt")
        dummy_inputs_processed = processor(text=dummy_text, images=dummy_image, return_tensors="pt")
        
        # --- Check --- 
        if dummy_inputs_processed is None:
            logger.error("The processor call returned None. Check processor loading and input image.")
            return
        if 'pixel_values' not in dummy_inputs_processed:
             logger.error(f"'pixel_values' not found in processor output. Output keys: {dummy_inputs_processed.keys()}")
             return
        # --- End check --- 

        # Ensure dummy input matches model's device and dtype
        dummy_pixel_values = dummy_inputs_processed['pixel_values'].to(DEVICE, dtype=MODEL_DTYPE)
        logger.info(f"Prepared dummy pixel_values with shape: {dummy_pixel_values.shape} and dtype: {dummy_pixel_values.dtype}")
    except Exception as e:
        logger.error(f"Failed to prepare dummy input: {e}")
        return

    # --- 4. Generate Intermediate Features for Projector ---
    logger.info("Generating intermediate features for projector input...")
    try:
        with torch.no_grad():
            # We need the hidden states output from the vision tower
            vision_outputs = vision_tower(dummy_pixel_values, output_hidden_states=True)
            # LLaVA uses the second-to-last hidden state
            # Shape: (batch_size, sequence_length, hidden_size)
            image_features = vision_outputs.hidden_states[-2].to(MODEL_DTYPE) # Ensure dtype matches
        logger.info(f"Intermediate image features shape: {image_features.shape}, dtype: {image_features.dtype}")
    except Exception as e:
        logger.error(f"Failed to run vision tower for intermediate features: {e}")
        return

    # --- 5. Export Vision Tower ---
    vision_tower_onnx_path = os.path.join(OUTPUT_DIR, "llava_vision_tower.onnx")
    logger.info(f"Exporting Vision Tower to {vision_tower_onnx_path}...")
    try:
        # Define dynamic axes for flexibility (batch size can vary)
        dynamic_axes_vision = {
            'pixel_values': {0: 'batch_size'},
            # Name outputs based on VisionTowerOutput structure or inspection
            'hidden_states': {0: 'batch_size'}, # Assuming this is the primary feature output needed
            # Add other outputs if needed, e.g., 'last_hidden_state', 'pooler_output'
        }
        # Check the actual outputs of vision_tower if necessary
        # For LLaVA, we primarily care about the hidden_states output for the projector

        torch.onnx.export(
            vision_tower,
            (dummy_pixel_values,), # Input needs to be a tuple
            vision_tower_onnx_path,
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=['pixel_values'],
             # IMPORTANT: Specify the output name corresponding to hidden_states[-2]
             # Check `vision_outputs` keys/structure. If it returns a custom object,
             # you might need a wrapper model for export or adjust output names.
             # Common outputs might include 'last_hidden_state', 'pooler_output', 'hidden_states'
             # Let's assume we need the sequence output (features)
            output_names=['hidden_states'], # Adjust if model returns dict or object
            dynamic_axes=dynamic_axes_vision
        )
        logger.info("Vision Tower exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export Vision Tower: {e}")
        # Continue to projector export if desired, or return here
        # return

    # --- 6. Export Multi-Modal Projector ---
    projector_onnx_path = os.path.join(OUTPUT_DIR, "llava_projector.onnx")
    logger.info(f"Exporting Multi-Modal Projector to {projector_onnx_path}...")
    try:
        dummy_projector_input = image_features # Use the intermediate features generated earlier
        dynamic_axes_projector = {
            'image_features': {0: 'batch_size', 1: 'sequence_length'}, # Batch and sequence can vary
            'projected_image_features': {0: 'batch_size'}
        }

        torch.onnx.export(
            projector,
            dummy_projector_input,
            projector_onnx_path,
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=['image_features'],
            output_names=['projected_image_features'],
            dynamic_axes=dynamic_axes_projector
        )
        logger.info("Multi-Modal Projector exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export Multi-Modal Projector: {e}")
        return # Stop if projector export fails

    # --- 7. Verification (Optional but Recommended) ---
    logger.info("--- Verifying ONNX Models ---")
    vision_tower_output_np = None
    try:
        logger.info(f"Loading Vision Tower ONNX: {vision_tower_onnx_path}")
        # Specify providers - prioritize GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
        vision_sess = ort.InferenceSession(vision_tower_onnx_path, providers=providers)
        logger.info(f"Vision Tower ONNX loaded. Providers: {vision_sess.get_providers()}")

        # Prepare input for ONNX Runtime (NumPy)
        dummy_pixel_values_np = dummy_pixel_values.cpu().numpy()
        vision_input_name = vision_sess.get_inputs()[0].name
        vision_output_name = vision_sess.get_outputs()[0].name # Assuming first output is needed
        vision_inputs = {vision_input_name: dummy_pixel_values_np}

        logger.info("Running Vision Tower ONNX inference...")
        vision_onnx_outputs = vision_sess.run([vision_output_name], vision_inputs)
        vision_tower_output_np = vision_onnx_outputs[0] # Output corresponding to 'hidden_states'
        logger.info(f"Vision Tower ONNX output shape: {vision_tower_output_np.shape}")

    except Exception as e:
        logger.error(f"Error verifying Vision Tower ONNX: {e}")
        # Cannot proceed to projector verification without vision output
        return

    try:
        logger.info(f"Loading Projector ONNX: {projector_onnx_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
        proj_sess = ort.InferenceSession(projector_onnx_path, providers=providers)
        logger.info(f"Projector ONNX loaded. Providers: {proj_sess.get_providers()}")

        # Input is the output from the ONNX vision tower run
        proj_input_name = proj_sess.get_inputs()[0].name
        proj_output_name = proj_sess.get_outputs()[0].name
        proj_inputs = {proj_input_name: vision_tower_output_np} # Use the NumPy output

        logger.info("Running Projector ONNX inference...")
        proj_onnx_outputs = proj_sess.run([proj_output_name], proj_inputs)
        final_embeddings_np = proj_onnx_outputs[0]
        logger.info(f"Projector ONNX output shape (final embeddings): {final_embeddings_np.shape}")
        logger.info("--- ONNX Verification Complete ---")

    except Exception as e:
        logger.error(f"Error verifying Projector ONNX: {e}")


if __name__ == "__main__":
    logger.info("Starting LLaVA vision components to ONNX conversion...")
    convert_llava_vision_to_onnx()
    logger.info("Conversion process finished.") 