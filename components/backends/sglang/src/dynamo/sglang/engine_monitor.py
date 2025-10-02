# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import sys
import time

import sglang as sgl

from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Configuration constants
HEALTH_CHECK_INTERVAL = int(os.getenv("SGLANG_HEALTH_CHECK_INTERVAL", "120"))
MAX_FAILURES_BEFORE_SHUTDOWN = int(
    os.getenv("SGLANG_MAX_FAILURES_BEFORE_SHUTDOWN", "3")
)
HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", "60"))
STARTUP_WAIT_INTERVAL = int(os.getenv("SGLANG_STARTUP_WAIT_INTERVAL", "5"))
MAX_STARTUP_WAIT_TIME = int(
    os.getenv("SGLANG_MAX_STARTUP_WAIT_TIME", "600")
)  # 10 minutes

# SGLang server status values
HEALTHY_STATUS = {"Up"}
STARTING_STATUS = {"Starting"}
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
        self._monitor_task = asyncio.create_task(self._start_monitoring())
        logger.info(
            f"{self.__class__.__name__} initialized for endpoint '{endpoint_name}' and health check task started."
        )

    def __del__(self):
        if hasattr(self, "_monitor_task") and self._monitor_task:
            self._monitor_task.cancel()

    def _shutdown_unhealthy(self, reason: str):
        """
        Mark endpoint as unhealthy, shutdown runtime, and exit.
        This is called when the engine is unresponsive or failed to start.

        Args:
            reason: Description of why shutdown is happening
        """
        self._update_health_status("NotReady")
        logger.error(f"Shutting down: {reason}")
        self.runtime.shutdown()
        sys.exit(1)

    def _update_health_status(self, status: str):
        """Update the endpoint health status in the DistributedRuntime."""
        if status == self.last_health_status:
            return

        try:
            logger.info(
                f"Updating endpoint '{self.endpoint_name}' health status: "
                f"{self.last_health_status} -> {status}"
            )
            self.runtime.set_endpoint_health_status(self.endpoint_name, status)
        except Exception as e:
            logger.error(f"Failed to update health status: {e}")
        finally:
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

    async def _wait_for_server_up(self) -> bool:
        """
        Wait for the server to report an 'Up' status before starting health checks.

        Returns:
            True if server is up, False if timeout reached
        """
        start_time = time.time()

        logger.info(
            f"Waiting for SGLang server to be ready (max wait: {MAX_STARTUP_WAIT_TIME}s)..."
        )

        iteration_count = 0
        while (time.time() - start_time) < MAX_STARTUP_WAIT_TIME:
            iteration_count += 1
            elapsed_time = int(time.time() - start_time)

            logger.info(
                f"Check iteration #{iteration_count} at {elapsed_time}s elapsed - attempting to get server info..."
            )

            try:
                # Run get_server_info in a thread to avoid event loop conflicts
                # SGLang's get_server_info uses run_until_complete internally
                # logger.info(f"TzuLing using to_thread to get server info")
                # server_info = await asyncio.wait_for(
                #     asyncio.to_thread(self.engine.get_server_info),
                #     timeout=HEALTH_CHECK_TIMEOUT,
                # )
                logger.info("TzuLing using get_running_loop to get server info")
                loop = asyncio.get_running_loop()
                server_info = await asyncio.wait_for(
                    loop.run_in_executor(None, self._get_server_info_with_loop),
                    timeout=HEALTH_CHECK_TIMEOUT,
                )
                logger.info(f"Got server_info response: {server_info}")

                # Extract server status
                server_status = self._extract_server_status(server_info)
                logger.info(f"Extracted server status: {server_status}")

                if server_status in HEALTHY_STATUS:
                    logger.info(
                        f"SGLang server is up and ready! Status: {server_status}"
                    )
                    return True
                elif server_status in STARTING_STATUS:
                    logger.info(
                        f"SGLang server still starting... Status: {server_status} (iteration #{iteration_count}, {elapsed_time}s elapsed)"
                    )
                else:
                    logger.warning(
                        f"SGLang server reporting unexpected status during startup: {server_status} (iteration #{iteration_count})"
                    )

            except asyncio.TimeoutError:
                logger.warning(
                    f"Server info request timed out after {HEALTH_CHECK_TIMEOUT}s (iteration #{iteration_count}, {elapsed_time}s elapsed)"
                )
            except Exception as e:
                logger.info(
                    f"Server not ready yet (iteration #{iteration_count}, {elapsed_time}s elapsed): {type(e).__name__}: {e}"
                )
                # Log full exception details at info level for debugging
                logger.info("Full exception details:", exc_info=True)

            # Wait before retrying
            logger.info(f"Waiting {STARTUP_WAIT_INTERVAL}s before next check...")
            await asyncio.sleep(STARTUP_WAIT_INTERVAL)

        # Timeout reached
        elapsed = int(time.time() - start_time)
        logger.error(
            f"SGLang server failed to become ready after {elapsed}s (timeout: {MAX_STARTUP_WAIT_TIME}s, iterations: {iteration_count})"
        )
        return False

    async def _start_monitoring(self):
        """
        Wait for the server to be up, then start the health check loop.
        """
        try:
            server_is_up = await self._wait_for_server_up()

            if not server_is_up:
                self._shutdown_unhealthy("SGLang server failed to start")

            self._update_health_status("Ready")
            logger.info(
                f"Starting health check monitoring for endpoint '{self.endpoint_name}' "
                f"(interval: {HEALTH_CHECK_INTERVAL}s, timeout: {HEALTH_CHECK_TIMEOUT}s)"
            )

            # Start the regular health check loop
            await self._check_engine_health()

        except asyncio.CancelledError:
            logger.debug("Monitoring task cancelled during startup")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in monitoring task: {e}")
            self._shutdown_unhealthy(f"Unexpected error in monitoring task: {e}")

    def _get_server_info_with_loop(self):
        """Wrapper that creates an event loop for the executor thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return self.engine.get_server_info()
        finally:
            loop.close()

    async def _check_engine_health(self):
        """Periodically check if the SGLang engine is responsive."""
        while True:
            try:
                # Run get_server_info in a thread to avoid event loop conflicts
                # SGLang's get_server_info uses run_until_complete internally
                # logger.info(f"TzuLing using to_thread to get server info"):s

                # server_info = await asyncio.wait_for(
                #     asyncio.to_thread(self.engine.get_server_info),
                #     timeout=HEALTH_CHECK_TIMEOUT,
                # )
                logger.info("TzuLing using get_running_loop to get server info")
                loop = asyncio.get_running_loop()
                server_info = await asyncio.wait_for(
                    loop.run_in_executor(None, self._get_server_info_with_loop),
                    timeout=HEALTH_CHECK_TIMEOUT,
                )
                server_status = self._extract_server_status(server_info)

                if server_status in UNHEALTHY_STATUS:
                    logger.warning(
                        f"SGLang engine reporting unhealthy status: {server_status}"
                    )
                    self.consecutive_failures += 1
                    self._update_health_status("NotReady")
                elif server_status in HEALTHY_STATUS:
                    self.consecutive_failures = 0
                    self._update_health_status("Ready")
                elif server_status in STARTING_STATUS:
                    # Server is still starting, don't count as failure but mark as not ready
                    logger.info(
                        f"SGLang engine is still starting... Status: {server_status}"
                    )
                    self._update_health_status("NotReady")
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
                self._shutdown_unhealthy(
                    f"SGLang engine unresponsive after {self.consecutive_failures} attempts"
                )

            # Wait before retrying
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                logger.debug("Health check loop cancelled during sleep")
                break

    async def stop(self):
        """Stop the monitor gracefully."""
        logger.info("Stopping SGLang engine monitor")

        self._update_health_status("NotReady")

        if hasattr(self, "_monitor_task") and self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.debug("Monitor task cancelled successfully")
                pass
        else:
            logger.debug("No monitor task to stop")

        logger.info("SGLang engine monitor stopped")
