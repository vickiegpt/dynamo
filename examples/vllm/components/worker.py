#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Worker - Model Inference Component
==================================

Executes model forward passes, processes batches of tokens through transformer layers,
computes Query, Key, and Value tensors and caches them.
"""

import argparse
import asyncio
import time
from typing import Any, Dict, List, Optional

import uvicorn
from config import VLLMConfig
from fastapi import FastAPI
from pydantic import BaseModel
from transport import ComponentTransport
from vllm import LLM, SamplingParams


# Request/Response models
class InferenceRequest(BaseModel):
    prompt: str
    prompt_tokens: List[int]
    max_tokens: int = 100
    temperature: float = 0.7
    request_id: str
    model: str


class WorkerComponent:
    """Worker component for model inference and execution."""

    def __init__(self, config: VLLMConfig, worker_id: int = 0):
        self.config = config
        self.worker_id = worker_id
        self.processed_inferences = 0
        self.llm: Optional[LLM] = None

        # Initialize transport
        self.transport = ComponentTransport(
            f"worker_{worker_id}" if worker_id > 0 else "worker", config
        )

        # Initialize FastAPI app
        self.app = FastAPI(title=f"vLLM Worker {worker_id}", version="1.0.0")
        self._setup_routes()

        print(f"[Worker-{worker_id}] Model inference component")
        print(f"[Worker-{worker_id}] Initializing vLLM engine...")

        # Initialize vLLM engine
        self._initialize_llm(config)

        print(f"[Worker-{worker_id}] Model loaded: {config.model}")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "component": f"worker_{self.worker_id}"}

        @self.app.post("/inference")
        async def handle_inference(request: InferenceRequest):
            """Handle inference request."""
            return await self._execute_inference(request.dict())

    def _initialize_llm(self, config: VLLMConfig):
        """Initialize the vLLM LLM engine."""
        try:
            self.llm = LLM(
                model=config.model,
                trust_remote_code=False,
                max_model_len=config.max_model_len,
                gpu_memory_utilization=config.gpu_memory_utilization,
                enable_chunked_prefill=config.enable_chunked_prefill,
                enable_prefix_caching=config.enable_prefix_caching,
                disable_log_stats=False,
                tensor_parallel_size=1,  # Single GPU per worker
                pipeline_parallel_size=1,
            )
            print(f"[Worker-{self.worker_id}] vLLM engine initialized successfully")

        except Exception as e:
            print(f"[Worker-{self.worker_id}] Error initializing vLLM: {e}")
            # Create a mock LLM for demo purposes if model loading fails
            self.llm = None
            print(f"[Worker-{self.worker_id}] Using mock inference for demo")

    async def _execute_inference(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model inference."""
        self.processed_inferences += 1

        print(
            f"\n[Worker-{self.worker_id}] ==================== EXECUTING INFERENCE ===================="
        )
        print(f"[Worker-{self.worker_id}] Inference #{self.processed_inferences}")
        print(
            f"[Worker-{self.worker_id}] Request ID: {request_data.get('request_id', 'unknown')}"
        )

        try:
            prompt = request_data.get("prompt", "")
            max_tokens = request_data.get("max_tokens", 100)
            temperature = request_data.get("temperature", 0.7)
            model = request_data.get("model", "unknown")

            print(f"[Worker-{self.worker_id}] Prompt length: {len(prompt)} chars")
            print(f"[Worker-{self.worker_id}] Max tokens: {max_tokens}")
            print(f"[Worker-{self.worker_id}] Temperature: {temperature}")

            # Notify sampler about parameters
            await self._notify_sampler(
                {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "worker_id": self.worker_id,
                }
            )

            # Execute inference
            if self.llm is not None:
                # Real vLLM inference
                sampling_params = SamplingParams(
                    temperature=temperature, max_tokens=max_tokens, top_p=0.9, seed=42
                )

                print(f"[Worker-{self.worker_id}] Running vLLM generation...")
                outputs = self.llm.generate([prompt], sampling_params)

                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].outputs[0].text
                    finish_reason = outputs[0].outputs[0].finish_reason

                    print(
                        f"[Worker-{self.worker_id}] Generated {len(generated_text)} chars"
                    )
                    print(f"[Worker-{self.worker_id}] Finish reason: {finish_reason}")

                    result = {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": generated_text,
                                },
                                "finish_reason": finish_reason,
                                "index": 0,
                            }
                        ],
                        "model": model,
                        "usage": {
                            "prompt_tokens": len(request_data.get("prompt_tokens", [])),
                            "completion_tokens": max_tokens,
                            "total_tokens": len(request_data.get("prompt_tokens", []))
                            + max_tokens,
                        },
                    }
                else:
                    result = {"error": "No output generated"}

            else:
                # Mock inference for demo
                print(f"[Worker-{self.worker_id}] Running mock inference...")
                await asyncio.sleep(0.5)  # Simulate processing time

                mock_response = f"This is a mock response from Worker-{self.worker_id} for the prompt: {prompt[:50]}..."

                result = {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": mock_response},
                            "finish_reason": "stop",
                            "index": 0,
                        }
                    ],
                    "model": model,
                    "usage": {
                        "prompt_tokens": len(request_data.get("prompt_tokens", [])),
                        "completion_tokens": max_tokens,
                        "total_tokens": len(request_data.get("prompt_tokens", []))
                        + max_tokens,
                    },
                }

            print(f"[Worker-{self.worker_id}] Inference completed successfully")
            return {"result": result}

        except Exception as e:
            print(f"[Worker-{self.worker_id}] Inference error: {e}")
            return {"error": f"Inference failed: {str(e)}"}

    async def _notify_sampler(self, params: Dict[str, Any]):
        """Notify sampler about sampling parameters."""
        print(f"[Worker-{self.worker_id}] Notifying sampler via HTTP transport")

        try:
            response = await self.transport.send_message("sampler", "notify", params)

            if "error" not in response:
                print(
                    f"[Worker-{self.worker_id}] Sampler notification: {response.get('status', 'acknowledged')}"
                )
            else:
                print(
                    f"[Worker-{self.worker_id}] Sampler notification warning: {response['error']}"
                )

        except Exception as e:
            print(f"[Worker-{self.worker_id}] Sampler notification warning: {e}")

    async def _health_check(self) -> Dict[str, Any]:
        """Worker health check."""
        return {
            "status": "healthy",
            "worker_id": self.worker_id,
            "model_loaded": self.llm is not None,
            "processed_inferences": self.processed_inferences,
            "timestamp": time.time(),
        }

    async def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "component": f"worker_{self.worker_id}",
            "rank": 0,
            "processed_inferences": self.processed_inferences,
            "model_loaded": self.llm is not None,
            "timestamp": time.time(),
        }


async def run_worker(config: VLLMConfig, worker_id: int = 0):
    """Run the worker component."""
    print(f"Starting vLLM Worker-{worker_id} Component")
    print(f"Port: {config.get_worker_port(worker_id)}")
    print("Communication: HTTP transport with other components")

    # Create worker component
    worker = WorkerComponent(config, worker_id)

    try:
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=worker.app,
            host="0.0.0.0",
            port=config.get_worker_port(worker_id),
            log_level="info",
        )

        # Start server
        server = uvicorn.Server(uvicorn_config)
        print(
            f"[Worker-{worker_id}] Component running on port {config.get_worker_port(worker_id)}"
        )

        await server.serve()

    except KeyboardInterrupt:
        print(f"[Worker-{worker_id}] Shutting down...")
    finally:
        await worker.transport.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="vLLM Worker Component")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID")

    args = parser.parse_args()

    # Load configuration
    config = VLLMConfig.from_env()
    if args.model:
        config.model = args.model

    # Run the worker microservice
    asyncio.run(run_worker(config, args.worker_id))


if __name__ == "__main__":
    main()
