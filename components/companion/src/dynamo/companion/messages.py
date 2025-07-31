"""Message types for model server-client communication."""

from dataclasses import dataclass
from typing import Literal, Optional, TypedDict

import torch

from vllm.config import VllmConfig

# Status types
ModelStatusType = Literal["loading", "loaded", "error"]


class StatusUpdate(TypedDict):
    """Status update message from server."""

    model_name: str
    status: ModelStatusType
    message: str
    device_id: int


class GetModelParametersRequest(TypedDict):
    """Request to get model parameters for IPC sharing."""

    type: Literal["get_model_parameters"]
    vllm_config: VllmConfig  # Direct VllmConfig object
    device_id: int  # Physical GPU device ID of the client
    local_rank: int
    global_rank: int
    world_size: int


class ModelParametersResponse(TypedDict):
    """Response with model parameters rebuild info for IPC."""

    model_parameters: Optional[dict]  # Dict of parameter name -> CUDATensorRebuildInfo
    model_name: str
    device_id: int
    local_rank: int
    global_rank: int
    world_size: int
    error: Optional[str]


class ErrorResponse(TypedDict):
    """Error response."""

    error: str


@dataclass
class CUDATensorRebuildInfo:
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
