# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import sys

import sglang as sgl

from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Configuration constants
HEALTH_CHECK_INTERVAL = int(os.getenv("SGLANG_HEALTH_CHECK_INTERVAL", "20"))
MAX_FAILURES_BEFORE_SHUTDOWN = int(
    os.getenv("SGLANG_MAX_FAILURES_BEFORE_SHUTDOWN", "3")
)
HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", "10"))

# SGLang server status values
HEALTHY_STATUS = {"Up", "Starting"}
UNHEALTHY_STATUS = {"UnHealthy"}


class SglangEngineMonitor:
    """
    Monitors the health of the SGLang engine and initiates a shutdown if the engine is unresponsive.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        engine: sgl.Engine,
        endpoint_name: str = "generate",
    ):
        if not isinstance(runtime, DistributedRuntime):
            raise ValueError(
                f"{self.__class__.__name__} requires an instance of DistributedRuntime."
            )
        # Note: We can't use isinstance with sgl.Engine since it's a LazyImport
        # The type of sgl.Engine is <class 'sglang.utils.LazyImport'>
        # Just check if it has the expected method
        if not hasattr(engine, "get_server_info"):
            raise ValueError(
                f"{self.__class__.__name__} requires an SGLang Engine with get_server_info method."
            )

        self.runtime = runtime
        self.engine = engine
        self.endpoint_name = endpoint_name
        self.consecutive_failures = 0
        self.last_health_status = None  # Initialize to None so first update works
        self._monitor_task = asyncio.create_task(self._check_engine_health())

        # Mark endpoint as ready initially
        self._update_health_status("Ready")

        logger.info(
            f"{self.__class__.__name__} initialized for endpoint '{endpoint_name}' and health check task started."
        )

    def __del__(self):
        if hasattr(self, "_monitor_task") and self._monitor_task:
            self._monitor_task.cancel()

    def _update_health_status(self, status: str):
        """Update the endpoint health status in the DistributedRuntime."""
        if status == self.last_health_status:
            return  # No change needed

        try:
            logger.info(
                f"Updating endpoint '{self.endpoint_name}' health status: "
                f"{self.last_health_status} -> {status}"
            )
            self.runtime.set_endpoint_health_status(self.endpoint_name, status)
        except Exception as e:
            logger.error(f"Failed to update health status: {e}")
        finally:
            # Always update internal state, even if runtime call fails
            self.last_health_status = status

    def _extract_server_status(self, server_info: dict) -> str:
        """
        Extract server status from potentially nested server info.

        Args:
            server_info: The server info dictionary from SGLang

        Returns:
            The server status string, defaults to "Up" if not found
        """
        internal_states = server_info.get("internal_states", {})

        # Handle list of states (multiple DP ranks)
        if isinstance(internal_states, list):
            if internal_states and internal_states[0]:
                # Take the first DP rank's status if available
                internal_states = internal_states[0]
            else:
                # Empty list or None first element
                internal_states = {}

        # Ensure we have a dictionary before accessing status
        if not isinstance(internal_states, dict):
            internal_states = {}

        return internal_states.get("status", "Up")

    async def _check_engine_health(self):
        """Periodically check if the SGLang engine is responsive."""
        while True:
            try:
                # Use get_server_info() to check if engine is responsive
                # This will raise an exception if the engine is dead
                loop = asyncio.get_event_loop()

                # Add timeout to prevent indefinite blocking
                server_info = await asyncio.wait_for(
                    loop.run_in_executor(None, self.engine.get_server_info),
                    timeout=HEALTH_CHECK_TIMEOUT,
                )

                # Extract server status using helper method
                server_status = self._extract_server_status(server_info)

                # Check if server status indicates a problem
                if server_status in UNHEALTHY_STATUS:
                    logger.warning(
                        f"SGLang engine reporting unhealthy status: {server_status}"
                    )
                    self.consecutive_failures += 1
                    self._update_health_status("NotReady")
                elif server_status in HEALTHY_STATUS:
                    self.consecutive_failures = 0
                    self._update_health_status("Ready")
                else:
                    # Unknown status - treat as unhealthy but log it
                    logger.warning(
                        f"SGLang engine reporting unknown status: {server_status}"
                    )
                    self.consecutive_failures += 1
                    self._update_health_status("NotReady")

            except asyncio.CancelledError:
                # This is expected when the monitor is being shut down
                logger.debug("Health check loop cancelled, shutting down gracefully")
                break
            except asyncio.TimeoutError:
                self.consecutive_failures += 1
                logger.error(
                    f"SGLang engine health check timed out after {HEALTH_CHECK_TIMEOUT}s (attempt {self.consecutive_failures})"
                )
                self._update_health_status("NotReady")
            except (ConnectionError, RuntimeError) as e:
                # Expected communication/engine errors
                self.consecutive_failures += 1
                logger.error(
                    f"SGLang engine health check failed (attempt {self.consecutive_failures}): {e}"
                )
                self._update_health_status("NotReady")
            except Exception as e:
                # Unexpected errors - log with full traceback
                self.consecutive_failures += 1
                logger.exception(
                    f"Unexpected error in SGLang engine health check (attempt {self.consecutive_failures}): {e}"
                )
                self._update_health_status("NotReady")

            # Check if we should shutdown
            if self.consecutive_failures >= MAX_FAILURES_BEFORE_SHUTDOWN:
                logger.warning(
                    f"SGLang engine unresponsive after {self.consecutive_failures} attempts"
                )
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                sys.exit(1)  # More graceful than os._exit(1)

            # Wait before retrying
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def stop(self):
        """Stop the monitor gracefully."""
        logger.info("Stopping SGLang engine monitor")

        self._update_health_status("NotReady")

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.debug("Monitor task cancelled successfully")
                pass
        logger.info("SGLang engine monitor stopped")
