import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import logging
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# --- Reuse functions and config from onnx_infer.py ---
from onnx_infer import (
    ONNX_MODEL_DIR, VISION_TOWER_ONNX_PATH, PROJECTOR_ONNX_PATH,
    ORIGINAL_MODEL_PATH, IMAGE_SOURCE, DEVICE,
    load_image, get_preprocessing_params, preprocess_image # Your ONNX preprocessing
)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_reference_embeddings(image_source: str, model_path: str) -> np.ndarray:
    """Generates embeddings using the original PyTorch model.
    Returns a tuple: (reference_embeddings_np, reference_pixel_values_np)
    """
    logger.info("--- Generating Reference Embeddings (PyTorch) ---")
    try:
        # Determine dtype based on device availability for consistency
        pt_dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
        logger.info(f"Loading PyTorch model from {model_path} with dtype {pt_dtype}")

        # Load original processor and model
        processor = LlavaProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=pt_dtype,
            low_cpu_mem_usage=True,
        ).to(DEVICE).eval()

        vision_tower = model.vision_tower
        projector = model.multi_modal_projector
        logger.info("PyTorch model and components loaded.")

        # Load image
        image = load_image(image_source)

        # Preprocess using the *original* processor
        # Provide dummy text if processor requires it
        dummy_text = "USER: <image>\nDescribe the image."
        inputs = processor(text=dummy_text, images=image, return_tensors="pt").to(DEVICE, dtype=pt_dtype)
        pixel_values = inputs['pixel_values']
        logger.info(f"PyTorch preprocessing done. Pixel values shape: {pixel_values.shape}")

        # Run PyTorch inference
        with torch.no_grad():
            logger.info("Running PyTorch vision_tower...")
            vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2].to(pt_dtype) # Use second-to-last hidden state
            logger.info(f"PyTorch vision_tower output shape: {image_features.shape}")

            logger.info("Running PyTorch multi_modal_projector...")
            ref_embeddings_pt = projector(image_features)
            logger.info(f"PyTorch projector output shape: {ref_embeddings_pt.shape}")

        # Convert to NumPy array on CPU
        ref_embeddings_np = ref_embeddings_pt.cpu().numpy()
        ref_pixel_values_np = pixel_values.cpu().numpy() # Also convert pixel values
        logger.info("--- Reference Embeddings Generated ---")
        return ref_embeddings_np, ref_pixel_values_np

    except Exception as e:
        logger.error(f"Failed to generate reference embeddings: {e}")
        raise

def get_onnx_embeddings(preprocessed_image: np.ndarray, params: dict) -> np.ndarray | None:
    """Generates embeddings using the ONNX pipeline from preprocessed data."""
    logger.info("--- Generating Test Embeddings (ONNX) ---")
    # --- 1. Load ONNX Sessions --- (Copied/adapted from run_inference)
    logger.info("Loading ONNX inference sessions...")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
        vision_sess = ort.InferenceSession(VISION_TOWER_ONNX_PATH, providers=providers)
        proj_sess = ort.InferenceSession(PROJECTOR_ONNX_PATH, providers=providers)
        vision_input_name = vision_sess.get_inputs()[0].name
        vision_output_name = vision_sess.get_outputs()[0].name
        proj_input_name = proj_sess.get_inputs()[0].name
        proj_output_name = proj_sess.get_outputs()[0].name
        logger.info("ONNX sessions loaded.")
    except Exception as e:
        logger.error(f"Failed to load ONNX models: {e}")
        return None

    # --- 2. Load and Preprocess Image (using custom function) ---
    # Skip image loading and preprocessing, as it's done outside now
    # try:
    #     image = load_image(image_source)
    #     preprocessed_image = preprocess_image(image, params) # Use the custom preprocess_image
    # except Exception as e:
    #     logger.error(f"Failed during image loading or ONNX preprocessing: {e}")
    #     return None
    logger.info(f"Using preprocessed image with shape: {preprocessed_image.shape} and dtype: {preprocessed_image.dtype}")

    # --- 3. Run ONNX Vision Tower Inference ---
    try:
        vision_inputs = {vision_input_name: preprocessed_image}
        vision_outputs = vision_sess.run([vision_output_name], vision_inputs)
        vision_features = vision_outputs[0]
        logger.info(f"ONNX Vision Tower output shape: {vision_features.shape}")
    except Exception as e:
        logger.error(f"ONNX Vision Tower inference failed: {e}")
        return None

    # --- 4. Run ONNX Projector Inference ---
    try:
        proj_inputs = {proj_input_name: vision_features}
        proj_outputs = proj_sess.run([proj_output_name], proj_inputs)
        onnx_embeddings = proj_outputs[0]
        logger.info(f"ONNX Projector output shape (final embeddings): {onnx_embeddings.shape}")
        logger.info("--- Test Embeddings Generated ---")
        return onnx_embeddings
    except Exception as e:
        logger.error(f"ONNX Projector inference failed: {e}")
        return None

# --- Main Verification Logic ---
if __name__ == "__main__":
    logger.info("Starting ONNX verification process...")

    if not os.path.exists(VISION_TOWER_ONNX_PATH) or not os.path.exists(PROJECTOR_ONNX_PATH):
        logger.error(f"ONNX model files not found in {ONNX_MODEL_DIR}. Please run convert.py first.")
    else:
        try:
            # 1. Get Reference Embeddings & Preprocessing Output (PyTorch)
            reference_embeddings, reference_pixel_values = get_reference_embeddings(IMAGE_SOURCE, ORIGINAL_MODEL_PATH)

            # 2. Get Preprocessing Params & Manually Preprocess Image
            preprocessing_params = get_preprocessing_params(ORIGINAL_MODEL_PATH)
            logger.info("--- Performing Manual Preprocessing (for ONNX input) ---")
            image_for_onnx = load_image(IMAGE_SOURCE)
            manual_preprocessed_image = preprocess_image(image_for_onnx, preprocessing_params)

            # --- Preprocessing Verification Step ---
            logger.info("\n--- Comparing Preprocessing Outputs --- ")
            if reference_pixel_values.shape != manual_preprocessed_image.shape:
                 logger.error(f"Preprocessing shape mismatch! Official: {reference_pixel_values.shape}, Manual: {manual_preprocessed_image.shape}")
            else:
                logger.info(f"Preprocessing shapes match: {reference_pixel_values.shape}")
                # Use stricter tolerance for preprocessing check
                prep_atol = 1e-5
                prep_rtol = 1e-4
                prep_close = np.allclose(reference_pixel_values, manual_preprocessed_image, atol=prep_atol, rtol=prep_rtol)
                if prep_close:
                    logger.info(f"Preprocessing SUCCESS: Outputs are close within tolerance (atol={prep_atol}, rtol={prep_rtol}).")
                else:
                    logger.warning(f"Preprocessing FAILED: Outputs differ significantly beyond tolerance (atol={prep_atol}, rtol={prep_rtol}).")
                    prep_abs_diff = np.abs(reference_pixel_values - manual_preprocessed_image)
                    prep_max_diff = np.max(prep_abs_diff)
                    logger.warning(f"  Max absolute difference in preprocessing: {prep_max_diff:.6f}")
            # --- End Preprocessing Verification ---

            # 3. Get Test Embeddings (ONNX) using the manually preprocessed image
            # Modify get_onnx_embeddings to accept the preprocessed image directly
            onnx_embeddings = get_onnx_embeddings(manual_preprocessed_image, preprocessing_params) # Pass preprocessed data

            # 4. Compare Final Embeddings
            if reference_embeddings is not None and onnx_embeddings is not None:
                logger.info("\n--- Comparing Embeddings ---")
                if reference_embeddings.shape != onnx_embeddings.shape:
                    logger.error(f"Shape mismatch! PyTorch: {reference_embeddings.shape}, ONNX: {onnx_embeddings.shape}")
                else:
                    logger.info(f"Shapes match: {reference_embeddings.shape}")

                    # Use numpy.allclose for numerical comparison
                    # Adjust tolerances (atol, rtol) as needed. Start stricter.
                    # For float16, tolerances might need to be slightly higher than for float32.
                    tolerance_atol = 1e-3 # Absolute tolerance
                    tolerance_rtol = 1e-3 # Relative tolerance

                    are_close = np.allclose(
                        reference_embeddings,
                        onnx_embeddings,
                        atol=tolerance_atol,
                        rtol=tolerance_rtol
                    )

                    if are_close:
                        logger.info(f"Verification SUCCESS: Embeddings are close within tolerance (atol={tolerance_atol}, rtol={tolerance_rtol}).")
                    else:
                        logger.warning(f"Verification FAILED: Embeddings differ significantly beyond tolerance (atol={tolerance_atol}, rtol={tolerance_rtol}).")

                        # Calculate and show difference metrics
                        abs_diff = np.abs(reference_embeddings - onnx_embeddings)
                        max_diff = np.max(abs_diff)
                        mean_diff = np.mean(abs_diff)
                        median_diff = np.median(abs_diff)
                        logger.warning(f"  Max absolute difference: {max_diff:.6f}")
                        logger.warning(f"  Mean absolute difference: {mean_diff:.6f}")
                        logger.warning(f"  Median absolute difference: {median_diff:.6f}")
            else:
                logger.error("Could not generate both reference and ONNX embeddings for comparison.")

        except Exception as e:
            logger.error(f"An error occurred during the verification process: {e}", exc_info=True) # Log traceback
