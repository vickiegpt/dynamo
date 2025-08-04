#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sampler - Token Sampling Component
==================================

Handles token sampling logic during generation.
Receives sampling parameters from workers.
"""

import argparse
import asyncio
import time
from typing import Any, Dict

import uvicorn
from config import VLLMConfig
from fastapi import FastAPI
from pydantic import BaseModel
from transport import ComponentTransport
from vllm.v1.sample.sampler import Sampler as VLLMSampler


# Request/Response models
class SamplerNotification(BaseModel):
    temperature: float
    max_tokens: int
    worker_id: str


class SamplerComponent:
    """Sampler component for token sampling and generation logic."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.sampling_requests = 0
        self.last_params = {}

        # Initialize transport
        self.transport = ComponentTransport("sampler", config)

        # Initialize FastAPI app
        self.app = FastAPI(title="vLLM Sampler", version="1.0.0")
        self._setup_routes()

        print("[Sampler] Token sampling component")
        print("[Sampler] Initializing vLLM sampler...")

        # Initialize vLLM sampler
        try:
            self.sampler = VLLMSampler()
            print("[Sampler] vLLM sampler initialized successfully")
        except Exception as e:
            print(f"[Sampler] Warning: Could not initialize vLLM sampler: {e}")
            self.sampler = None
            print("[Sampler] Using mock sampling for demo")

        print("[Sampler] Supports various sampling strategies")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "component": "sampler"}

        @self.app.post("/notify")
        async def handle_notification(request: SamplerNotification):
            """Handle sampling parameter notification."""
            return await self._sample_notification(request.dict())

    async def _sample_notification(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sampling parameter notification from worker."""
        self.sampling_requests += 1

        print(
            "\n[Sampler] ==================== SAMPLING NOTIFICATION ===================="
        )
        print(f"[Sampler] Notification #{self.sampling_requests}")

        try:
            temperature = params.get("temperature", 1.0)
            max_tokens = params.get("max_tokens", 100)
            worker_id = params.get("worker_id", "unknown")

            print(f"[Sampler] Worker ID: {worker_id}")
            print(f"[Sampler] Temperature: {temperature}")
            print(f"[Sampler] Max tokens: {max_tokens}")

            # Store latest parameters for demo purposes
            self.last_params = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "worker_id": worker_id,
                "timestamp": time.time(),
            }

            # Log sampling strategy
            if temperature == 0.0:
                print("[Sampler] Using greedy sampling (deterministic)")
            elif temperature < 0.8:
                print("[Sampler] Using low-temperature sampling (focused)")
            else:
                print("[Sampler] Using high-temperature sampling (creative)")

            # Demonstrate different sampling strategies
            if self.sampler is not None:
                print("[Sampler] vLLM sampler ready for token generation")
            else:
                print("[Sampler] Mock sampling logic applied")

            result = {
                "status": "acknowledged",
                "sampling_strategy": self._get_sampling_strategy(temperature),
                "parameters_received": self.last_params,
            }

            print("[Sampler] Sampling parameters processed successfully")
            return {"result": result}

        except Exception as e:
            print(f"[Sampler] Sampling error: {e}")
            return {"error": f"Sampling failed: {str(e)}"}

    def _get_sampling_strategy(self, temperature: float) -> str:
        """Determine sampling strategy based on temperature."""
        if temperature == 0.0:
            return "greedy"
        elif temperature < 0.5:
            return "low_temperature"
        elif temperature < 1.0:
            return "medium_temperature"
        else:
            return "high_temperature"

    async def _get_sampler_info(self) -> Dict[str, Any]:
        """Get sampler information."""
        result = {
            "component": "sampler",
            "status": "healthy",
            "sampler_loaded": self.sampler is not None,
            "sampling_requests": self.sampling_requests,
            "last_params": self.last_params,
            "supported_strategies": [
                "greedy",
                "low_temperature",
                "medium_temperature",
                "high_temperature",
            ],
            "timestamp": time.time(),
        }

        return {"result": result}

    async def get_sampler_stats(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            "component": "sampler",
            "sampling_requests": self.sampling_requests,
            "last_params": self.last_params,
            "sampler_loaded": self.sampler is not None,
            "timestamp": time.time(),
        }


async def run_sampler(config: VLLMConfig):
    """Run the sampler component."""
    print("Starting vLLM Sampler Component")
    print(f"Port: {config.sampler_port}")
    print("Communication: HTTP transport with other components")

    # Create sampler component
    sampler = SamplerComponent(config)

    try:
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=sampler.app, host="0.0.0.0", port=config.sampler_port, log_level="info"
        )

        # Start server
        server = uvicorn.Server(uvicorn_config)
        print(f"[Sampler] Component running on port {config.sampler_port}")

        await server.serve()

    except KeyboardInterrupt:
        print("[Sampler] Shutting down...")
    finally:
        await sampler.transport.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="vLLM Sampler Component")
    parser.add_argument("--model", type=str, help="Model name or path")

    args = parser.parse_args()

    # Load configuration
    config = VLLMConfig.from_env()
    if args.model:
        config.model = args.model

    # Run the sampler microservice
    asyncio.run(run_sampler(config))


if __name__ == "__main__":
    main()
