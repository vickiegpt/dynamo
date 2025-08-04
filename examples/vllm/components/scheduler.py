#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Scheduler - Request Coordination Component
=========================================

Responsible for queuing, batching, and orchestrating allocation of model inference tasks.
Tracks request priorities, batching for prefill and decode, chunked processing.
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
from vllm.config import SchedulerConfig
from vllm.transformers_utils.tokenizer import get_tokenizer


# Request/Response models
class ScheduleRequest(BaseModel):
    request_id: str
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 100
    temperature: float = 0.7


class ScheduleResponse(BaseModel):
    result: Dict[str, Any]


class SchedulerComponent:
    """Scheduler component for request coordination and task orchestration."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.processed_requests = 0
        self.current_batch = []

        # Initialize transport
        self.transport = ComponentTransport("scheduler", config)

        # Initialize FastAPI app
        self.app = FastAPI(title="vLLM Scheduler", version="1.0.0")
        self._setup_routes()

        # Initialize tokenizer
        print(f"[Scheduler] Initializing tokenizer for {config.model}")
        self.tokenizer = get_tokenizer(
            tokenizer_name=config.model,
            trust_remote_code=False,
            tokenizer_mode="auto",
            revision=None,
        )

        # Create scheduler config
        self.scheduler_config = SchedulerConfig(
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=256,
            max_model_len=config.max_model_len or 8192,
            use_v2_block_manager=False,
            num_lookahead_slots=0,
            delay_factor=0.0,
            enable_chunked_prefill=config.enable_chunked_prefill,
            embedding_mode=False,
            preemption_mode="swap",
        )

        print("[Scheduler] Request coordination component")
        print(f"[Scheduler] Max batched tokens: {config.max_num_batched_tokens}")
        print(f"[Scheduler] Chunked prefill: {config.enable_chunked_prefill}")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "component": "scheduler"}

        @self.app.post("/process")
        async def process_request(request: ScheduleRequest):
            """Process scheduling request."""
            return await self._schedule_completion_request(request.dict())

    async def _schedule_completion_request(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Schedule and coordinate completion request."""
        self.processed_requests += 1

        print(
            "\n[Scheduler] ==================== SCHEDULING REQUEST ===================="
        )
        print(f"[Scheduler] Request #{self.processed_requests}")
        print(f"[Scheduler] Model: {request_data.get('model', 'unknown')}")

        try:
            # Step 1: Prepare and tokenize prompt
            prompt = self._prepare_prompt(request_data.get("messages", []))
            prompt_tokens = self._tokenize_prompt(prompt)

            # Step 2: Check KV cache allocation
            await self._allocate_kv_cache(len(prompt_tokens))

            # Step 3: Forward to worker for execution
            worker_response = await self._forward_to_worker(
                {
                    "prompt": prompt,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": request_data.get("max_tokens", 100),
                    "temperature": request_data.get("temperature", 0.7),
                    "stream": request_data.get("stream", False),
                    "model": request_data.get("model"),
                    "request_id": request_data.get("request_id"),
                }
            )

            if "error" in worker_response:
                return {"error": f"Worker error: {worker_response['error']}"}

            result = worker_response.get("result", {})
            print("[Scheduler] Request scheduled and completed")

            return {"result": result}

        except Exception as e:
            print(f"[Scheduler] Scheduling error: {e}")
            return {"error": f"Scheduling failed: {str(e)}"}

    def _prepare_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt string."""
        print(f"[Scheduler] Processing {len(messages)} message(s)")

        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        print(f"[Scheduler] Generated prompt ({len(prompt)} chars)")
        return prompt

    def _tokenize_prompt(self, prompt: str) -> List[int]:
        """Tokenize prompt for scheduling."""
        print("[Scheduler] Tokenizing prompt...")
        tokens = self.tokenizer.encode(prompt)
        print(f"[Scheduler] Tokenized to {len(tokens)} tokens")
        return tokens

    async def _allocate_kv_cache(self, prompt_tokens: int):
        """Request KV cache allocation."""
        print("[Scheduler] Requesting KV cache allocation via transport")

        try:
            response = await self.transport.send_message(
                "kv_cache", "allocate", {"prompt_tokens": prompt_tokens}
            )

            if "error" not in response:
                print(
                    f"[Scheduler] KV cache allocation: {response.get('status', 'allocated')}"
                )
            else:
                print(f"[Scheduler] KV cache allocation warning: {response['error']}")

        except Exception as e:
            print(f"[Scheduler] KV cache allocation warning: {e}")

    async def _forward_to_worker(
        self, worker_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Forward request to worker component."""
        print("[Scheduler] Forwarding to worker via transport")

        try:
            response = await self.transport.send_message(
                "worker", "inference", worker_request
            )

            print("[Scheduler] Received response from worker")
            return response

        except Exception as e:
            print(f"[Scheduler] Worker communication error: {e}")
            return {"error": f"Worker communication failed: {str(e)}"}

    async def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "component": "scheduler",
            "rank": 0,
            "processed_requests": self.processed_requests,
            "current_batch_size": len(self.current_batch),
            "scheduler_config": {
                "max_num_batched_tokens": self.scheduler_config.max_num_batched_tokens,
                "max_num_seqs": self.scheduler_config.max_num_seqs,
                "enable_chunked_prefill": self.scheduler_config.enable_chunked_prefill,
            },
            "timestamp": time.time(),
        }


async def run_scheduler(config: VLLMConfig):
    """Run the scheduler component."""
    print("Starting vLLM Scheduler Component")
    print(f"Port: {config.scheduler_port}")
    print("Communication: HTTP transport with other components")

    # Create scheduler component
    scheduler = SchedulerComponent(config)

    try:
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=scheduler.app,
            host="0.0.0.0",
            port=config.scheduler_port,
            log_level="info",
        )

        # Start server
        server = uvicorn.Server(uvicorn_config)
        print(f"[Scheduler] Component running on port {config.scheduler_port}")

        await server.serve()

    except KeyboardInterrupt:
        print("[Scheduler] Shutting down...")
    finally:
        await scheduler.transport.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="vLLM Scheduler Component")
    parser.add_argument("--model", type=str, help="Model name or path")

    args = parser.parse_args()

    # Load configuration
    config = VLLMConfig.from_env()
    if args.model:
        config.model = args.model

    # Run the scheduler component
    asyncio.run(run_scheduler(config))


if __name__ == "__main__":
    main()
