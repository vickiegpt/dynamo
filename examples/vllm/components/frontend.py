# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Frontend - OpenAI API Server
============================

Provides the OpenAI-compatible API interface for inference requests.
Acts as the main entry point, communicating with router via dynamo transport.
"""

import argparse
import asyncio
import time
import uuid
from typing import Any, Dict, List

import uvicorn
from config import VLLMConfig
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transport import ComponentTransport


# Pydantic models for OpenAI API
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class FrontendComponent:
    """Frontend component providing OpenAI API interface."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.app = FastAPI(title="vLLM Modular Frontend", version="1.0.0")
        self.transport = ComponentTransport("frontend", config)
        self.processed_requests = 0

        # Setup routes
        self._setup_routes()

        print("[Frontend] OpenAI API server")
        print("[Frontend] Communication: HTTP transport to router")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "component": "frontend"}

        @self.app.get("/v1/models")
        async def list_models():
            """List available models."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.config.model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "vllm-modular",
                    }
                ],
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Handle chat completion requests."""
            return await self._handle_chat_completion(request)

    async def _handle_chat_completion(
        self, request: ChatCompletionRequest
    ) -> JSONResponse:
        """Handle chat completion request."""
        self.processed_requests += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        print("\n[Frontend] ==================== NEW REQUEST ====================")
        print(f"[Frontend] Request #{self.processed_requests}")
        print(f"[Frontend] Request ID: {request_id}")
        print(f"[Frontend] Model: {request.model}")
        print(f"[Frontend] Messages: {len(request.messages)}")
        print(f"[Frontend] Max tokens: {request.max_tokens}")
        print(f"[Frontend] Temperature: {request.temperature}")

        try:
            # Prepare request data
            request_data = {
                "request_id": request_id,
                "model": request.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": request.stream,
            }

            # Forward to router via HTTP transport
            print("[Frontend] Forwarding to router via HTTP transport")

            response = await self.transport.send_message(
                "router", "completion", request_data
            )

            if "error" in response:
                print(f"[Frontend] Router error: {response['error']}")
                raise HTTPException(status_code=500, detail=response["error"])

            # Extract result from router response
            result = response.get("result", {})

            if "error" in result:
                print(f"[Frontend] Processing error: {result['error']}")
                raise HTTPException(status_code=500, detail=result["error"])

            # Format as OpenAI response
            choices = result.get("choices", [])
            usage = result.get("usage", {})

            openai_response = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": choices,
                "usage": usage,
            }

            print("[Frontend] Request completed successfully")
            return JSONResponse(content=openai_response)

        except HTTPException:
            raise
        except Exception as e:
            print(f"[Frontend] Unexpected error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    async def start_transport(self):
        """Start HTTP transport."""
        # Transport is ready immediately for HTTP
        print("[Frontend] HTTP transport ready")

    async def stop_transport(self):
        """Stop HTTP transport."""
        await self.transport.close()
        print("[Frontend] HTTP transport closed")


async def run_frontend(config: VLLMConfig):
    """Run the frontend component."""
    print("Starting vLLM Frontend Component")
    print(f"Port: {config.frontend_port}")
    print(f"Model: {config.model}")
    print("Communication: HTTP transport")

    # Create frontend component
    frontend = FrontendComponent(config)

    # Start dynamo transport
    await frontend.start_transport()

    try:
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=frontend.app,
            host="0.0.0.0",
            port=config.frontend_port,
            log_level="info",
            access_log=True,
        )

        # Start server
        server = uvicorn.Server(uvicorn_config)
        print(f"[Frontend] Server starting on http://0.0.0.0:{config.frontend_port}")

        await server.serve()

    except KeyboardInterrupt:
        print("[Frontend] Shutting down...")
    finally:
        await frontend.stop_transport()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="vLLM Frontend Component")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--port", type=int, help="Port to run on")

    args = parser.parse_args()

    # Load configuration
    config = VLLMConfig.from_env()
    if args.model:
        config.model = args.model
    if args.port:
        config.frontend_port = args.port

    # Run the frontend server
    asyncio.run(run_frontend(config))


if __name__ == "__main__":
    main()
