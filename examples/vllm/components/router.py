#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Router - Traffic Management Component
====================================

Centralizes traffic management, load-balances among workers,
routes based on model name or traffic shaping.
"""

import argparse
import asyncio
import time
from typing import Any, Dict, List

import uvicorn
from config import VLLMConfig
from fastapi import FastAPI
from pydantic import BaseModel
from transport import ComponentTransport


# Request/Response models
class CompletionRequest(BaseModel):
    request_id: str
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class RouterComponent:
    """Router component for traffic management and load balancing."""

    def __init__(self, config: VLLMConfig):
        self.config = config

        # Worker pool tracking
        self.available_workers = []
        self.worker_stats = {}
        self.processed_requests = 0

        # Initialize transport
        self.transport = ComponentTransport("router", config)

        # Initialize FastAPI app
        self.app = FastAPI(title="vLLM Router", version="1.0.0")
        self._setup_routes()

        print("[Router] Traffic management component")
        print("[Router] Load balancing and routing logic")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "component": "router"}

        @self.app.post("/completion")
        async def handle_completion(request: CompletionRequest):
            """Handle completion request routing."""
            return await self._route_completion_request(request.dict())

    async def _route_completion_request(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route completion request through the pipeline."""
        self.processed_requests += 1
        print("\n[Router] ==================== ROUTING REQUEST ====================")
        print(f"[Router] Request ID: {request_data.get('request_id', 'unknown')}")
        print(f"[Router] Model: {request_data.get('model', 'unknown')}")
        print(f"[Router] Messages: {len(request_data.get('messages', []))}")

        try:
            # Step 1: Forward to scheduler for coordination
            print("[Router] Forwarding to scheduler via HTTP transport")
            scheduler_response = await self.transport.send_message(
                "scheduler", "process", request_data
            )

            if "error" in scheduler_response:
                return {"error": f"Scheduler error: {scheduler_response['error']}"}

            result = scheduler_response.get("result", {})
            print("[Router] Request routed successfully")

            return {"result": result}

        except Exception as e:
            print(f"[Router] Routing error: {e}")
            return {"error": f"Routing failed: {str(e)}"}

    async def register_worker(self, worker_id: str) -> Dict[str, Any]:
        """Register a worker with the router."""
        self.available_workers.append(worker_id)
        self.worker_stats[worker_id] = {
            "registered_at": time.time(),
            "last_seen": time.time(),
            "status": "available",
        }

        print(f"[Router] Registered worker {worker_id}")

        return {
            "status": "registered",
            "worker_id": worker_id,
            "total_workers": len(self.available_workers),
        }

    def select_best_worker(self) -> str:
        """Select best available worker (simple round-robin for demo)."""
        if not self.available_workers:
            return "worker"  # Default worker

        # Simple round-robin selection
        selected = self.available_workers[
            self.processed_requests % len(self.available_workers)
        ]
        print(f"[Router] Selected worker: {selected}")
        return selected

    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "component": "router",
            "rank": 0,
            "processed_requests": self.processed_requests,
            "available_workers": len(self.available_workers),
            "worker_stats": self.worker_stats,
            "timestamp": time.time(),
        }


async def run_router(config: VLLMConfig):
    """Run the router component."""
    print("Starting vLLM Router Component")
    print(f"Port: {8004}")  # Router internal port
    print("Communication: HTTP transport with other components")

    # Create router component
    router = RouterComponent(config)

    try:
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=router.app, host="0.0.0.0", port=8004, log_level="info"
        )

        # Start server
        server = uvicorn.Server(uvicorn_config)
        print("[Router] Component running on port 8004")

        await server.serve()

    except KeyboardInterrupt:
        print("[Router] Shutting down...")
    finally:
        await router.transport.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="vLLM Router Component")
    parser.add_argument("--model", type=str, help="Model name or path")

    args = parser.parse_args()

    # Load configuration
    config = VLLMConfig.from_env()
    if args.model:
        config.model = args.model

    # Run the router microservice
    asyncio.run(run_router(config))


if __name__ == "__main__":
    main()
