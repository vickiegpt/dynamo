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
        # TODO: should i just use the async client as this might block?
        self.client = CircusClient(endpoint=endpoint, timeout=5.0)

    @classmethod
    def from_state_file(cls, namespace: str) -> "CircusController":
        """Create a CircusController from a Dynamo state file.

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
        """Add a new watcher to circus.

        Args:
            name: Name of the watcher
            cmd: Command to run
            env: Environment variables
            **options: Additional watcher options
        """
        try:
            # Build the watcher options dict
            watcher_options = {
                "copy_env": True,
                "stop_children": True,
                "graceful_timeout": 86400,
            }

            # Add env if provided
            if env:
                watcher_options["env"] = env

            # Add any additional options
            watcher_options.update(options)

            # Format message exactly as AddWatcher command expects
            response = self.client.send_message(
                "add",
                name=name,  # Required property
                cmd=cmd,  # Required property
                args=[],  # Optional array of args
                options=watcher_options,  # Options dict
                start=True,  # Start immediately
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
        self, name: str, nostop: bool = False, waiting: bool = True
    ) -> bool:
        """Remove a watcher from circus.

        Args:
            name: Name of the watcher
            nostop: If True, don't stop the processes, just remove the watcher
            waiting: If True, wait for complete removal before returning
        """
        try:
            response = self.client.send_message(
                "rm", name=name, nostop=nostop, waiting=waiting
            )
            print("RESPONSE", response)

            if response.get("status") != "ok":
                logger.error(
                    f"Failed to remove watcher {name}: {response.get('reason', 'unknown error')}"
                )
                return False
            return True
        except (CallError, Exception) as e:
            logger.error(f"Failed to remove watcher {name}: {e}")
            return False

    async def get_watcher_processes(self, name: str) -> int:
        """Get number of processes for a watcher."""
        try:
            response = self.client.send_message("numprocesses", name=name)
            return int(response.get("numprocesses", 0))
        except (CallError, Exception) as e:
            logger.error(f"Failed to get process count for {name}: {e}")
            return 0

    async def list_watchers(self) -> list[str]:
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
