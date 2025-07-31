# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message types for Dynamo companion server/client communication."""

from typing import Dict, Optional
from pydantic import BaseModel


class GetModelParametersRequest(BaseModel):
    """Request to get model parameters from the server."""
    model_name: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    device_id: int
    local_rank: int
    global_rank: int
    world_size: int


class ModelParametersResponse(BaseModel):
    """Response containing model parameters for IPC sharing."""
    model_parameters: Optional[Dict[str, Dict]] = None  # Parameter name -> CUDA IPC info
    model_name: str
    device_id: int
    local_rank: int
    global_rank: int
    world_size: int
    error: Optional[str] = None


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