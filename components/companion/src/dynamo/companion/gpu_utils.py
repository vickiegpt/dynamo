"""Utilities for handling GPU device mapping in IPC scenarios."""

import os
import subprocess
from typing import Optional, Tuple
import torch


def get_physical_device_index(logical_device: Optional[int] = None) -> int:
    """
    Get the physical GPU device index for a given logical device index.
    
    This correctly handles CUDA_VISIBLE_DEVICES remapping.
    
    Args:
        logical_device: The logical device index (default: current device)
        
    Returns:
        The physical device index
    """
    if logical_device is None:
        logical_device = torch.cuda.current_device()
    
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    
    if cuda_visible_devices:
        # Parse the visible devices list
        visible_devices = [int(d.strip()) for d in cuda_visible_devices.split(",") if d.strip()]
        
        if logical_device < len(visible_devices):
            return visible_devices[logical_device]
        else:
            # Fallback if index is out of range
            return logical_device
    else:
        # No remapping, logical = physical
        return logical_device


def get_gpu_uuid(device_index: int) -> str:
    """
    Get the UUID of a GPU by its physical device index.
    
    Args:
        device_index: Physical GPU device index
        
    Returns:
        GPU UUID string
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output to find the UUID for the device
        for line in result.stdout.splitlines():
            if f"GPU {device_index}:" in line:
                # Extract UUID from line like "GPU 0: NVIDIA A100 (UUID: GPU-xxxxx)"
                if "UUID: " in line:
                    uuid = line.split("UUID: ")[1].rstrip(")")
                    return uuid
                    
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback to device index
    return f"gpu-{device_index}"


def ensure_same_gpu(server_device: str, client_cuda_visible_devices: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if the server device and client environment will use the same physical GPU.
    
    Args:
        server_device: Device string from server (e.g., "cuda:1")
        client_cuda_visible_devices: CUDA_VISIBLE_DEVICES value from client
        
    Returns:
        Tuple of (is_same_gpu, explanation_message)
    """
    # Parse server device
    if server_device.startswith("cuda:"):
        server_physical_device = int(server_device.split(":")[1])
    else:
        return False, f"Invalid server device: {server_device}"
    
    # Get client's view
    if client_cuda_visible_devices is None:
        client_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    
    client_physical_device = get_physical_device_index(0)  # Assuming we use device 0
    
    is_same = server_physical_device == client_physical_device
    
    if is_same:
        msg = f"Both using physical GPU {server_physical_device}"
    else:
        msg = (f"Device mismatch: server using physical GPU {server_physical_device}, "
               f"client using physical GPU {client_physical_device} "
               f"(CUDA_VISIBLE_DEVICES={client_cuda_visible_devices})")
    
    return is_same, msg 