#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
KV Cache Manager - Memory Management Component
==============================================

Manages the key-value cache for attention mechanism storage efficiently
by grouping tokens into fixed-size chunks and allocating/retrieving KV blocks.
"""

import argparse
import asyncio
import time
from typing import Any, Dict, Optional

import uvicorn
from config import VLLMConfig
from fastapi import FastAPI
from pydantic import BaseModel
from transport import ComponentTransport
from vllm.config import CacheConfig
from vllm.core.block_manager import SelfAttnBlockSpaceManager


# Request/Response models
class AllocationRequest(BaseModel):
    prompt_tokens: int
    sequence_id: str = None


class KVCacheManagerComponent:
    """KV Cache Manager component for memory allocation and management."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.allocations = 0
        self.deallocations = 0
        self.block_manager: Optional[SelfAttnBlockSpaceManager] = None

        # Initialize transport
        self.transport = ComponentTransport("kv_cache", config)

        # Initialize FastAPI app
        self.app = FastAPI(title="vLLM KV Cache Manager", version="1.0.0")
        self._setup_routes()

        print("[KV Cache] Memory management component")
        print("[KV Cache] Initializing block manager...")

        # Initialize block manager
        self._initialize_block_manager(config)

        print("[KV Cache] Cache management ready")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "component": "kv_cache"}

        @self.app.post("/allocate")
        async def handle_allocation(request: AllocationRequest):
            """Handle memory allocation request."""
            return await self._allocate_memory(request.dict())

    def _initialize_block_manager(self, config: VLLMConfig):
        """Initialize the vLLM block manager."""
        try:
            # Create cache config
            cache_config = CacheConfig(
                block_size=16,  # Standard block size
                gpu_memory_utilization=config.gpu_memory_utilization,
                swap_space=0,
                cache_dtype="auto",
                num_gpu_blocks=1000,  # Demo value
                num_cpu_blocks=1000,  # Demo value
                sliding_window=None,
                enable_prefix_caching=config.enable_prefix_caching,
                cpu_offload_gb=0,
            )

            # Initialize block manager
            self.block_manager = SelfAttnBlockSpaceManager(
                block_size=cache_config.block_size,
                num_gpu_blocks=cache_config.num_gpu_blocks,
                num_cpu_blocks=cache_config.num_cpu_blocks,
                watermark=0.01,
                sliding_window=cache_config.sliding_window,
                enable_caching=cache_config.enable_prefix_caching,
            )

            print("[KV Cache] Block manager initialized successfully")
            print(f"[KV Cache] GPU blocks: {cache_config.num_gpu_blocks}")
            print(f"[KV Cache] CPU blocks: {cache_config.num_cpu_blocks}")
            print(f"[KV Cache] Block size: {cache_config.block_size}")
            print(f"[KV Cache] Prefix caching: {cache_config.enable_prefix_caching}")

        except Exception as e:
            print(f"[KV Cache] Error initializing block manager: {e}")
            self.block_manager = None
            print("[KV Cache] Using mock cache management for demo")

    async def _allocate_memory(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate KV cache memory."""
        self.allocations += 1

        print(
            "\n[KV Cache] ==================== ALLOCATING MEMORY ===================="
        )
        print(f"[KV Cache] Allocation #{self.allocations}")

        try:
            prompt_tokens = request_data.get("prompt_tokens", 0)
            sequence_id = request_data.get("sequence_id", f"seq_{self.allocations}")

            print(f"[KV Cache] Prompt tokens: {prompt_tokens}")
            print(f"[KV Cache] Sequence ID: {sequence_id}")

            if self.block_manager is not None:
                # Real block allocation
                blocks_needed = (prompt_tokens + 15) // 16  # Round up to block size

                print(f"[KV Cache] Blocks needed: {blocks_needed}")
                print("[KV Cache] Allocating memory blocks...")

                # Simulate allocation
                allocated_blocks = list(
                    range(self.allocations, self.allocations + blocks_needed)
                )

                result = {
                    "status": "allocated",
                    "sequence_id": sequence_id,
                    "blocks": allocated_blocks,
                    "blocks_count": len(allocated_blocks),
                    "prompt_tokens": prompt_tokens,
                }

            else:
                # Mock allocation
                blocks_needed = (prompt_tokens + 15) // 16

                print(f"[KV Cache] Mock allocation for {blocks_needed} blocks")

                result = {
                    "status": "allocated",
                    "sequence_id": sequence_id,
                    "blocks": list(range(blocks_needed)),
                    "blocks_count": blocks_needed,
                    "prompt_tokens": prompt_tokens,
                }

            print("[KV Cache] Memory allocated successfully")
            return {"result": result}

        except Exception as e:
            print(f"[KV Cache] Allocation error: {e}")
            return {"error": f"Memory allocation failed: {str(e)}"}

    async def _deallocate_memory(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deallocate KV cache memory."""
        self.deallocations += 1

        print(
            "\n[KV Cache] ==================== DEALLOCATING MEMORY ===================="
        )
        print(f"[KV Cache] Deallocation #{self.deallocations}")

        try:
            sequence_id = request_data.get("sequence_id", "unknown")
            blocks = request_data.get("blocks", [])

            print(f"[KV Cache] Sequence ID: {sequence_id}")
            print(f"[KV Cache] Deallocating {len(blocks)} blocks")

            # Simulate deallocation
            result = {
                "status": "deallocated",
                "sequence_id": sequence_id,
                "deallocated_blocks": len(blocks),
            }

            print("[KV Cache] Memory deallocated successfully")
            return {"result": result}

        except Exception as e:
            print(f"[KV Cache] Deallocation error: {e}")
            return {"error": f"Memory deallocation failed: {str(e)}"}

    async def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.block_manager is not None:
                # Real statistics
                stats = {
                    "total_allocations": self.allocations,
                    "total_deallocations": self.deallocations,
                    "active_sequences": self.allocations - self.deallocations,
                    "block_manager_available": True,
                    "timestamp": time.time(),
                }
            else:
                # Mock statistics
                stats = {
                    "total_allocations": self.allocations,
                    "total_deallocations": self.deallocations,
                    "active_sequences": self.allocations - self.deallocations,
                    "block_manager_available": False,
                    "timestamp": time.time(),
                }

            return {"result": stats}

        except Exception as e:
            return {"error": f"Failed to get cache stats: {str(e)}"}

    async def get_cache_manager_stats(self) -> Dict[str, Any]:
        """Get cache manager statistics."""
        return {
            "component": "kv_cache",
            "rank": 0,
            "allocations": self.allocations,
            "deallocations": self.deallocations,
            "active_sequences": self.allocations - self.deallocations,
            "block_manager_loaded": self.block_manager is not None,
            "timestamp": time.time(),
        }


async def run_kv_cache_manager(config: VLLMConfig):
    """Run the KV cache manager component."""
    print("Starting vLLM KV Cache Manager Component")
    print(f"Port: {config.kv_cache_port}")
    print("Communication: HTTP transport with other components")

    # Create KV cache manager component
    kv_cache = KVCacheManagerComponent(config)

    try:
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=kv_cache.app,
            host="0.0.0.0",
            port=config.kv_cache_port,
            log_level="info",
        )

        # Start server
        server = uvicorn.Server(uvicorn_config)
        print(f"[KV Cache] Component running on port {config.kv_cache_port}")

        await server.serve()

    except KeyboardInterrupt:
        print("[KV Cache] Shutting down...")
    finally:
        await kv_cache.transport.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="vLLM KV Cache Manager Component")
    parser.add_argument("--model", type=str, help="Model name or path")

    args = parser.parse_args()

    # Load configuration
    config = VLLMConfig.from_env()
    if args.model:
        config.model = args.model

    # Run the KV cache manager microservice
    asyncio.run(run_kv_cache_manager(config))


if __name__ == "__main__":
    main()
