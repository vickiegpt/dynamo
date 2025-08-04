#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple configuration for vLLM modular components."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class VLLMConfig:
    """Configuration for vLLM modular deployment."""

    # Model configuration
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9

    # Component ports
    frontend_port: int = 8000
    scheduler_port: int = 8001
    kv_cache_port: int = 8002
    worker_base_port: int = 8010
    sampler_port: int = 8003

    # System
    num_workers: int = 2
    log_level: str = "INFO"

    # vLLM V1 specific
    max_num_batched_tokens: int = 512
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True

    @classmethod
    def from_env(cls) -> "VLLMConfig":
        """Create config from environment variables."""
        return cls(
            model=os.getenv("VLLM_MODEL", cls.model),
            tensor_parallel_size=int(
                os.getenv("VLLM_TENSOR_PARALLEL_SIZE", cls.tensor_parallel_size)
            ),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            num_workers=int(os.getenv("VLLM_NUM_WORKERS", cls.num_workers)),
        )

    def get_worker_port(self, worker_id: int) -> int:
        """Get port for specific worker."""
        return self.worker_base_port + worker_id
