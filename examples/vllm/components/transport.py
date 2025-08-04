# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Component Transport Layer
========================

Simple HTTP-based transport with clean interface for future Dynamo integration.
"""

import asyncio
from typing import Any, Dict

import httpx
from config import VLLMConfig


class ComponentTransport:
    """
    Transport abstraction for inter-component communication.

    Currently uses HTTP. Future versions can swap to Dynamo distributed compute
    without changing component code.
    """

    def __init__(self, component_name: str, config: VLLMConfig):
        self.component_name = component_name
        self.config = config

        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            timeout=30.0,
        )

        # Service discovery - maps component names to URLs
        self.service_urls = {
            "frontend": f"http://localhost:{config.frontend_port}",
            "router": "http://localhost:8004",
            "scheduler": f"http://localhost:{config.scheduler_port}",
            "worker": f"http://localhost:{config.get_worker_port(0)}",
            "kv_cache": f"http://localhost:{config.kv_cache_port}",
            "sampler": f"http://localhost:{config.sampler_port}",
        }

    async def send_message(
        self, target: str, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send message to target component.

        Args:
            target: Component name (e.g., "scheduler", "worker")
            endpoint: API endpoint (e.g., "process", "allocate")
            payload: Request data

        Returns:
            Response data
        """
        try:
            url = f"{self.service_urls[target]}/{endpoint}"

            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            return response.json()

        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Transport error: {str(e)}"}

    async def send_with_retry(
        self, target: str, endpoint: str, payload: Dict[str, Any], max_retries: int = 3
    ) -> Dict[str, Any]:
        """Send with exponential backoff retry."""

        for attempt in range(max_retries):
            result = await self.send_message(target, endpoint, payload)

            if "error" not in result:
                return result

            if attempt < max_retries - 1:
                wait_time = 0.1 * (2**attempt)
                await asyncio.sleep(wait_time)

        return result

    async def broadcast(
        self, targets: list, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Send message to multiple targets in parallel."""

        tasks = []
        for target in targets:
            task = self.send_message(target, endpoint, payload)
            tasks.append((target, task))

        results = {}
        for target, task in tasks:
            results[target] = await task

        return results

    async def health_check(self, target: str) -> bool:
        """Check if target component is healthy."""
        try:
            result = await self.send_message(target, "health", {})
            return result.get("status") == "healthy"
        except Exception:
            return False

    async def close(self):
        """Clean up transport resources."""
        await self.client.aclose()


# Future Dynamo implementation (placeholder)
class DynamoTransport(ComponentTransport):
    """
    Dynamo-based transport for distributed compute.

    TODO: Implement when integrating with real Dynamo distributed systems.
    """

    async def send_message(
        self, target: str, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Future: Use Dynamo distributed compute primitives
        # return await dynamo.remote_call(target, endpoint, payload)

        # For now, fallback to HTTP
        return await super().send_message(target, endpoint, payload)


def create_transport(
    component_name: str, config: VLLMConfig, use_dynamo: bool = False
) -> ComponentTransport:
    """Factory function to create appropriate transport."""

    if use_dynamo:
        return DynamoTransport(component_name, config)
    else:
        return ComponentTransport(component_name, config)
