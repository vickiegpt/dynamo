# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
from pathlib import Path

from circus.client import CircusClient
from circus.exc import CallError

logger = logging.getLogger(__name__)


class CircusController:
    """A circus client implementation for Dynamo"""

    def __init__(self, endpoint: str):
        """Initialize connection to arbiter.

        Args:
            endpoint: The circus endpoint (e.g., tcp://127.0.0.1:54927)
        """
        self.endpoint = endpoint
        self.client = CircusClient(endpoint=endpoint, timeout=5.0)

    @classmethod
    def from_state_file(cls, namespace: str) -> "CircusController":
        """
        Create a CircusController from a Dynamo state file.

        Args:
            namespace: The Dynamo namespace

        Returns:
            CircusController instance

        Raises:
            FileNotFoundError: If state file doesn't exist
            ValueError: If no endpoint found in state file
        """
        state_file = (
            Path(
                os.environ.get("DYN_LOCAL_STATE_DIR", Path.home() / ".dynamo" / "state")
            )
            / f"{namespace}.json"
        )
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        with open(state_file, "r") as f:
            state = json.load(f)

        endpoint = state.get("circus_endpoint")
        if not endpoint:
            raise ValueError(f"No endpoint found in state file: {state_file}")

        return cls(endpoint)

    async def add_watcher(
        self, name: str, cmd: str, env: dict = None, **options
    ) -> bool:
        """
        Add a new watcher to circus.

        Args:
            name: Name of the watcher
            cmd: Command to run
            env: Environment variables
            **options: Additional watcher options
        """
        try:
            watcher_options = {
                "copy_env": True,
                "stop_children": True,
                "graceful_timeout": 86400,
            }

            if env:
                watcher_options["env"] = env

            watcher_options.update(options)

            response = self.client.send_message(
                "add",
                name=name,
                cmd=cmd,
                args=[],
                options=watcher_options,
                start=True,
            )

            if response.get("status") != "ok":
                logger.error(
                    f"Failed to add watcher {name}: {response.get('reason', 'unknown error')}"
                )
                return False
            return True

        except (CallError, Exception) as e:
            logger.error(f"Failed to add watcher {name}: {e}")
            return False

    async def remove_watcher(
        self,
        name: str,
        nostop: bool = False,
        waiting: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> bool:
        """
        Remove a watcher. We add retry logic here to ensure that a worker
        is properly removed after its process has been killed.

        Args:
            name: The name of the watcher to remove
            nostop: Whether to stop the processes or not
            waiting: Whether to wait for completion
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if successful
        """
        for attempt in range(1, max_retries + 1):
            num_processes = await self._get_watcher_processes(name)
            if num_processes == 0:
                logger.info(
                    f"Watcher {name} does not exist or has already been removed - skipping"
                )
                return True
            try:
                logger.info(
                    f"Removing watcher {name} (attempt {attempt}/{max_retries})"
                )
                response = self.client.send_message(
                    "rm", name=name, nostop=nostop, waiting=waiting
                )

                if response.get("status") != "ok":
                    error_msg = f"Failed to remove watcher {name}: {response.get('reason', 'unknown error')}"
                    logger.error(error_msg)

                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    return False

                logger.info(f"Successfully removed watcher {name} on attempt {attempt}")
                return True

            except (CallError, Exception) as e:
                logger.error(f"Failed to remove watcher {name}: {e}")

                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"All {max_retries} attempts to remove watcher {name} failed"
                    )
                    return False

        return False

    async def _get_watcher_processes(self, name: str) -> int:
        """Get number of processes for a watcher."""
        try:
            response = self.client.send_message("numprocesses", name=name)
            return int(response.get("numprocesses", 0))
        except (CallError, Exception) as e:
            logger.error(f"Failed to get process count for {name}: {e}")
            return 0

    async def _list_watchers(self) -> list[str]:
        """List all watchers managed by circus."""
        try:
            response = self.client.send_message("list")
            return response.get("watchers", [])
        except (CallError, Exception) as e:
            logger.error(f"Failed to list watchers: {e}")
            return []

    def close(self):
        """Close the connection to the arbiter."""
        if hasattr(self, "client"):
            self.client.stop()
