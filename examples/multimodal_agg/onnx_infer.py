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


print(f"ONNX Runtime version in use: {ort.__version__}") # Add version print

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
ORT_DEVICE = "cuda" if DEVICE == "cuda" else "cpu" # Device string for OrtValue should just be 'cuda'

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
    """
    Loads ONNX models, preprocesses image, and runs inference using I/O Binding
    to keep outputs on the GPU. Returns the final embeddings as an OrtValue on the GPU.
    """

    # --- 1. Load ONNX Sessions ---
    logger.info("Loading ONNX inference sessions...")
    try:
        # Configure Execution Providers (TensorRT, CUDA, CPU)
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0, # Or appropriate GPU ID
                'trt_fp16_enable': True, # Enable FP16 precision
                'trt_engine_cache_enable': True, # Enable engine caching
                'trt_engine_cache_path': os.path.join(ONNX_MODEL_DIR, "trt_cache"), # Cache directory
                # Add other TRT options if needed (e.g., int8, workspace size)
            }),
            ('CUDAExecutionProvider', {
                'device_id': 0, # Or appropriate GPU ID
                # Add other CUDA options if needed
            }),
            'CPUExecutionProvider' # Fallback
        ]

        # Ensure the cache directory exists
        trt_cache_path = os.path.join(ONNX_MODEL_DIR, "trt_cache")
        if not os.path.exists(trt_cache_path):
            os.makedirs(trt_cache_path)
            logger.info(f"Created TensorRT engine cache directory: {trt_cache_path}")

        # Remove provider options here as they are now inline with providers
        # provider_options = [{'device_id': 0}] if DEVICE == 'cuda' else []
        # logger.info(f"Using providers: {providers} with options: {provider_options}")
        logger.info(f"Attempting to load sessions with providers: {providers}")

        vision_sess = ort.InferenceSession(VISION_TOWER_ONNX_PATH, providers=providers)
        logger.info(f"Loaded Vision Tower from {VISION_TOWER_ONNX_PATH}. Effective providers: {vision_sess.get_providers()}")
        proj_sess = ort.InferenceSession(PROJECTOR_ONNX_PATH, providers=providers)
        logger.info(f"Loaded Projector from {PROJECTOR_ONNX_PATH}. Effective providers: {proj_sess.get_providers()}")

        # Get input/output names and shapes/types
        vision_input_meta = vision_sess.get_inputs()[0]
        vision_output_meta = vision_sess.get_outputs()[0]
        proj_input_meta = proj_sess.get_inputs()[0]
        proj_output_meta = proj_sess.get_outputs()[0]

        vision_input_name = vision_input_meta.name
        vision_output_name = vision_output_meta.name
        proj_input_name = proj_input_meta.name
        proj_output_name = proj_output_meta.name

        # Note: Shapes might contain symbolic dimensions (e.g., 'batch_size', 'sequence_length')
        # We'll need to provide concrete shapes when allocating buffers. Assume batch size 1 for now.
        # Type mapping might be needed (e.g., 'tensor(float16)' -> np.float16)
        type_map = {"tensor(float16)": np.float16, "tensor(float)": np.float32} # Add more as needed
        vision_output_type = type_map.get(vision_output_meta.type, np.float32)
        proj_output_type = type_map.get(proj_output_meta.type, np.float32)

        logger.info(f"Vision Tower I/O: Input='{vision_input_name}' ({vision_input_meta.type}, {vision_input_meta.shape}), Output='{vision_output_name}' ({vision_output_meta.type}, {vision_output_meta.shape})")
        logger.info(f"Projector I/O: Input='{proj_input_name}' ({proj_input_meta.type}, {proj_input_meta.shape}), Output='{proj_output_name}' ({proj_output_meta.type}, {proj_output_meta.shape})")

    except Exception as e:
        logger.error(f"Failed to load ONNX models or get metadata: {e}")
        raise # Re-raise after logging

    # --- 2. Load and Preprocess Image ---
    try:
        # Using the actual image source provided
        # dummy_image_source = "https://llava-vl.github.io/static/images/view.jpg" # Keep original dummy for testing if needed
        image = load_image(image_source)
        preprocessed_image_np = preprocess_image(image, params) # NumPy array on CPU

        # Wrap NumPy input with OrtValue. This transfers data to the specified device if needed.
        # For GPU, the copy happens here (CPU -> GPU).
        preprocessed_image_ortvalue = OrtValue.ortvalue_from_numpy(preprocessed_image_np, ORT_DEVICE)
        logger.info(f"Input image OrtValue created on device: {preprocessed_image_ortvalue.device_name()}")

    except Exception as e:
        logger.error(f"Failed during image loading or preprocessing: {e}")
        raise # Re-raise after logging

    # --- 3. Allocate Output Buffers & Run Vision Tower Inference with IOBinding ---
    logger.info("Preparing IOBinding for ONNX Vision Tower inference...")
    try:
        # Determine concrete output shape (replace symbolic dims if necessary)
        # Assuming vision tower output shape is like [batch_size, sequence_length, hidden_size]
        # Example: using preprocessed_image shape for batch_size=1, assuming fixed seq_len/hidden_size from model
        # This might need adjustment based on the actual model's output shapes
        # For Llava-1.5 vision tower, output is typically (batch_size, num_patches+1, hidden_size), e.g. (1, 577, 1024)
        # Let's try getting it dynamically, assuming batch size is 1
        vision_output_shape = [1 if isinstance(d, str) else d for d in vision_output_meta.shape]
        logger.info(f"Attempting to allocate vision output with shape: {vision_output_shape} and type: {vision_output_type}")

        # Create an OrtValue on the target device (GPU or CPU)
        vision_output_ortvalue = OrtValue.ortvalue_from_shape_and_type(vision_output_shape, vision_output_type, ORT_DEVICE)
        logger.info(f"Vision Tower output buffer allocated on device: {vision_output_ortvalue.device_name()}")

        # Create IOBinding
        io_binding_vision = vision_sess.io_binding()

        # Bind input OrtValue
        io_binding_vision.bind_ortvalue_input(vision_input_name, preprocessed_image_ortvalue)

        # Bind output OrtValue
        io_binding_vision.bind_ortvalue_output(vision_output_name, vision_output_ortvalue)

        logger.info("Running Vision Tower inference with IOBinding...")
        vision_sess.run_with_iobinding(io_binding_vision)
        logger.info("Vision Tower inference complete. Output is in pre-allocated buffer.")
        # The result is now in vision_output_ortvalue (on GPU/CPU as specified)

    except Exception as e:
        logger.error(f"ONNX Vision Tower inference with IOBinding failed: {e}")
        raise # Re-raise after logging

    # --- 4. Allocate Output Buffers & Run Projector Inference with IOBinding ---
    logger.info("Preparing IOBinding for ONNX Projector inference...")
    try:
        # Determine projector output shape (e.g., [1, 577, 4096] for Llava-1.5)
        proj_output_shape = [1 if isinstance(d, str) else d for d in proj_output_meta.shape]
        # We need the sequence length from the *actual* vision output
        proj_output_shape[1] = vision_output_ortvalue.shape()[1] # Update sequence length dynamically
        logger.info(f"Attempting to allocate projector output with shape: {proj_output_shape} and type: {proj_output_type}")


        # Create the output OrtValue on the target device
        final_embeddings_ortvalue = OrtValue.ortvalue_from_shape_and_type(proj_output_shape, proj_output_type, ORT_DEVICE)
        logger.info(f"Projector output buffer allocated on device: {final_embeddings_ortvalue.device_name()}")

        # Create IOBinding
        io_binding_proj = proj_sess.io_binding()

        # Bind input (output of vision tower, already an OrtValue on the correct device)
        io_binding_proj.bind_ortvalue_input(proj_input_name, vision_output_ortvalue)

        # Bind output
        io_binding_proj.bind_ortvalue_output(proj_output_name, final_embeddings_ortvalue)

        logger.info("Running Projector inference with IOBinding...")
        proj_sess.run_with_iobinding(io_binding_proj)
        logger.info("Projector inference complete. Final embeddings are in pre-allocated buffer.")

    except Exception as e:
        logger.error(f"ONNX Projector inference with IOBinding failed: {e}")
        raise # Re-raise after logging

    # Return the final embeddings as an OrtValue on the GPU/CPU
    return final_embeddings_ortvalue


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting ONNX inference process...")

    if not os.path.exists(VISION_TOWER_ONNX_PATH) or not os.path.exists(PROJECTOR_ONNX_PATH):
        logger.error(f"ONNX model files not found in {ONNX_MODEL_DIR}. Please run convert.py first.")
    else:
        try:
            # Get necessary preprocessing info from the original processor
            preprocessing_params = get_preprocessing_params(ORIGINAL_MODEL_PATH)

            # Run the full inference pipeline using I/O binding
            # Returns an OrtValue, potentially on GPU
            final_embeddings_ortvalue = run_inference(IMAGE_SOURCE, preprocessing_params)

            if final_embeddings_ortvalue is not None:
                logger.info("Inference finished successfully. Result is an OrtValue.")
                # Accessing data requires copying to CPU if it's on GPU
                # We do the copy here for printing/saving, but it was avoided during inference pipeline
                if final_embeddings_ortvalue.device_name() == 'cuda':
                    logger.info("Copying final embeddings from GPU to CPU for inspection/saving...")
                    embeddings_np = final_embeddings_ortvalue.numpy()
                else:
                     embeddings_np = final_embeddings_ortvalue.numpy() # Already on CPU or just get numpy


                print("\n--- Final Embeddings (from OrtValue) ---")
                print(f"Device: {final_embeddings_ortvalue.device_name()}")
                print(f"GPU Address (data_ptr): {final_embeddings_ortvalue.data_ptr()}") # Print the data pointer
                print(f"Shape: {embeddings_np.shape}") # Use numpy shape
                print(f"Data type: {embeddings_np.dtype}") # Use numpy dtype
                # Calculate size in bytes
                num_elements = np.prod(embeddings_np.shape)
                element_size_bytes = embeddings_np.itemsize
                total_size_bytes = num_elements * element_size_bytes
                print(f"Total Size (bytes): {total_size_bytes}") # Print the total size in bytes
                print("First few values of the first embedding vector:")
                print(embeddings_np[0, 0, :5])

                # --- Add JSON Output ---
                print("\n--- Final Embeddings (JSON) ---")
                try:
                    # Convert the NumPy array (copied from OrtValue if needed) to list
                    embeddings_list = embeddings_np.tolist()
                    embeddings_json = json.dumps(embeddings_list, indent=2)
                except Exception as e:
                    logger.error(f"Failed to convert embeddings to JSON: {e}")
                # --- End JSON Output ---

                # --- Verification using Manual CUDA Array Interface (UNSAFE) ---
                if final_embeddings_ortvalue.device_name() == 'cuda':
                    logger.warning("Attempting UNSAFE tensor reconstruction from raw GPU pointer...")
                    try:
                        import torch

                        # Define a wrapper class with the __cuda_array_interface__ attribute
                        class CudaArrayInterfaceWrapper:
                            def __init__(self, interface_dict):
                                self.__cuda_array_interface__ = interface_dict

                        # 1. Get necessary metadata
                        pointer = final_embeddings_ortvalue.data_ptr()
                        shape = tuple(embeddings_np.shape) # Use shape from numpy array
                        np_dtype = embeddings_np.dtype

                        # 2. Map NumPy dtype to typestr
                        typestr_map = {np.dtype(np.float16): '<f2', np.dtype(np.float32): '<f4', np.dtype(np.float64): '<f8'}
                        if np_dtype not in typestr_map:
                            raise TypeError(f"Unsupported NumPy dtype for CUDA Array Interface: {np_dtype}")
                        typestr = typestr_map[np_dtype]

                        # 3. Calculate strides (assuming C-contiguous)
                        element_size = np_dtype.itemsize
                        strides = None # None implies C-contiguous

                        # 4. Construct the interface dictionary
                        cuda_array_interface_dict = {
                            'shape': shape,
                            'typestr': typestr,
                            'data': (pointer, False), # (pointer, read-only flag)
                            'version': 3,
                            'strides': strides
                        }
                        logger.info(f"Constructed __cuda_array_interface__ dict: {cuda_array_interface_dict}")

                        # 5. Create a wrapper object instance
                        interface_wrapper = CudaArrayInterfaceWrapper(cuda_array_interface_dict)

                        # 6. Create PyTorch tensor from the wrapper object (zero-copy view)
                        reconstructed_tensor_pt = torch.as_tensor(interface_wrapper)
                        # Explicitly set device, as_tensor might not infer it correctly
                        # reconstructed_tensor_pt = reconstructed_tensor_pt.cuda() # Might not be needed if as_tensor infers from interface
                        logger.info(f"Reconstructed PyTorch tensor on device: {reconstructed_tensor_pt.device}")
                        # Check if it's actually on CUDA
                        if not reconstructed_tensor_pt.is_cuda:
                             logger.warning("Warning: Reconstructed tensor is NOT on CUDA device despite using interface. Trying manual move...")
                             reconstructed_tensor_pt = reconstructed_tensor_pt.cuda()
                             logger.info(f"Tensor moved to device: {reconstructed_tensor_pt.device}")


                        # 7. Copy the reconstructed tensor to CPU for comparison
                        reconstructed_tensor_cpu = reconstructed_tensor_pt.cpu().numpy()

                        # 8. Compare with the original NumPy array
                        if np.allclose(embeddings_np, reconstructed_tensor_cpu):
                            logger.info("Verification SUCCESS (unsafe method): Reconstructed tensor matches original.")

                    except ImportError:
                        logger.warning("PyTorch not installed. Skipping unsafe verification.")
                    except Exception as e:
                        logger.error(f"An error occurred during UNSAFE verification: {e}")
                else:
                    logger.info("Skipping verification as the tensor is not on CUDA GPU.")
                # --- End Verification --- 

        except Exception as e:
            # Catch exceptions raised from run_inference or get_preprocessing_params
            logger.error(f"An error occurred during the IOBinding inference process: {e}")
