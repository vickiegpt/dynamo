# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message types for Dynamo companion server/client communication."""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass
import pickle
import base64
import torch


class GetModelParametersRequest(BaseModel):
    """Request to get model parameters from the server."""
    vllm_config_pickled: str = Field(description="Base64-encoded pickled VllmConfig")
    config_hash: str = Field(description="Hash of the VllmConfig for quick comparison")
    device_id: int
    local_rank: int
    global_rank: int
    world_size: int
    
    @classmethod
    def from_vllm_config(cls, vllm_config: Any, device_id: int, local_rank: int, 
                         global_rank: int, world_size: int) -> "GetModelParametersRequest":
        """Create request from VllmConfig object."""
        # Pickle and base64 encode the VllmConfig
        vllm_config_bytes = pickle.dumps(vllm_config)
        vllm_config_pickled = base64.b64encode(vllm_config_bytes).decode('utf-8')
        
        # Compute the config hash
        config_hash = vllm_config.compute_hash()
        
        return cls(
            vllm_config_pickled=vllm_config_pickled,
            config_hash=config_hash,
            device_id=device_id,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
        )
    
    def get_vllm_config(self) -> Any:
        """Deserialize and return the VllmConfig object."""
        vllm_config_bytes = base64.b64decode(self.vllm_config_pickled)
        return pickle.loads(vllm_config_bytes)


@dataclass
class CUDATensorRebuildInfo:
    """Information needed to rebuild a CUDA tensor via IPC."""
    tensor_type: type[torch.Tensor]
    tensor_size: torch.Size
    tensor_stride: tuple[int, ...]
    tensor_offset: int
    storage_type: type
    tensor_dtype: torch.dtype
    device: int  # This is the CUDA Device ID
    ipc_handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    tensor_requires_grad: bool
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool

    @classmethod
    def from_rebuild_args(cls, rebuild_args: tuple) -> "CUDATensorRebuildInfo":
        return cls(*rebuild_args)

    def to_rebuild_args(self) -> tuple:
        return (
            self.tensor_type,
            self.tensor_size,
            self.tensor_stride,
            self.tensor_offset,
            self.storage_type,
            self.tensor_dtype,
            self.device,
            self.ipc_handle,
            self.storage_size_bytes,
            self.storage_offset_bytes,
            self.tensor_requires_grad,
            self.ref_counter_handle,
            self.ref_counter_offset,
            self.event_handle,
            self.event_sync_required,
        )


class ModelParametersResponse(BaseModel):
    """Response containing model parameters for IPC sharing."""
    model_parameters_pickled: Optional[str] = Field(
        default=None,
        description="Base64-encoded pickled dict of parameter name -> CUDATensorRebuildInfo"
    )
    model_name: str
    device_id: int
    local_rank: int
    global_rank: int
    world_size: int
    error: Optional[str] = None
    
    @classmethod
    def from_model_parameters(cls, model_parameters: Optional[Dict[str, CUDATensorRebuildInfo]],
                              model_name: str, device_id: int, local_rank: int,
                              global_rank: int, world_size: int, error: Optional[str] = None):
        """Create response from model parameters dict."""
        if model_parameters:
            # Pickle and base64 encode the model parameters dict
            params_bytes = pickle.dumps(model_parameters)
            model_parameters_pickled = base64.b64encode(params_bytes).decode('utf-8')
        else:
            model_parameters_pickled = None
            
        return cls(
            model_parameters_pickled=model_parameters_pickled,
            model_name=model_name,
            device_id=device_id,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            error=error,
        )
    
    def get_model_parameters(self) -> Optional[Dict[str, CUDATensorRebuildInfo]]:
        """Deserialize and return the model parameters dict."""
        if self.model_parameters_pickled:
            params_bytes = base64.b64decode(self.model_parameters_pickled)
            return pickle.loads(params_bytes)
        return None


class StatusUpdateMessage(BaseModel):
    """Status update message from server."""
    model_name: str
    status: str  # "loading", "loaded", "error"
    message: str
    device_id: int
    progress: Optional[float] = None  # Optional loading progress 0.0-1.0


class ErrorMessage(BaseModel):
    """Error response message."""
    error: str
    details: Optional[str] = None