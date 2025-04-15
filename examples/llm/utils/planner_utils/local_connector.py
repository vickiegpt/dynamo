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
from typing import Any, Dict, List, Optional

from .circusd import CircusController
from .planner_connector import PlannerConnector

from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_logger

configure_logger()
logger = logging.getLogger(__name__)


class LocalConnector(PlannerConnector):
    """Local connector for managing Dynamo components using CircusController."""

    def __init__(self, namespace: str, runtime: Optional[DistributedRuntime] = None):
        """Initialize LocalConnector.

        Args:
            namespace: The Dynamo namespace
            runtime: Optional DistributedRuntime instance
        """
        self.namespace = namespace
        self.runtime = runtime
        self.state_file = Path.home() / ".dynamo" / "state" / f"{namespace}.json"
        self.circus = CircusController.from_state_file(namespace)

    async def load_state(self) -> Dict[str, Any]:
        """Load state from state file.

        Returns:
            State dictionary
        """
        if not self.state_file.exists():
            raise FileNotFoundError(f"State file not found: {self.state_file}")

        with open(self.state_file, "r") as f:
            return json.load(f)

    async def save_state(self, state: Dict[str, Any]) -> bool:
        """Save state to state file.

        Args:
            state: State dictionary to save

        Returns:
            True if successful
        """
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    async def _get_available_gpus(self) -> List[str]:
        """Get list of unallocated GPU IDs.

        Returns:
            List of available GPU IDs
        """
        try:
            state = await self.load_state()
            system_resources = state.get("environment", {}).get("SYSTEM_RESOURCES", {})
            all_gpus = set(str(gpu) for gpu in system_resources.get("gpu_info", []))

            # Get allocated GPUs
            allocated_gpus = set()
            for component_info in state.get("components", {}).values():
                resources = component_info.get("resources", {})
                gpu_list = resources.get("allocated_gpus", [])
                allocated_gpus.update(str(gpu) for gpu in gpu_list)

            logger.info(f"Allocated GPUs: {allocated_gpus}")
            available = sorted(list(all_gpus - allocated_gpus))
            logger.info(f"Available GPUs: {available}")
            return available
        except Exception as e:
            logger.error(f"Failed to get available GPUs: {e}")
            return []

    async def get_component_replicas(self, component_name: str) -> int:
        """Get the number of replicas for a component.

        Args:
            component_name: Name of the component

        Returns:
            Number of replicas
        """
        watcher_name = f"{self.namespace}_{component_name}"
        try:
            return await self.circus.get_watcher_processes(watcher_name)
        except Exception as e:
            logger.error(f"Failed to get replicas for {component_name}: {e}")
            return 0

    async def add_component(self, component_name: str) -> bool:
        """Add a component to the planner"""
        try:
            state = await self.load_state()
            # Find max suffix
            max_suffix = 0
            for watcher_name in state["components"].keys():
                if watcher_name.startswith(f"{self.namespace}_{component_name}_"):
                    suffix = int(
                        watcher_name.replace(f"{self.namespace}_{component_name}_", "")
                    )
                    max_suffix = max(max_suffix, suffix)

            watcher_name = f"{self.namespace}_{component_name}_{max_suffix + 1}"

            if component_name not in [
                c.replace(f"{self.namespace}_", "") for c in state["components"]
            ]:
                raise ValueError(
                    f"Component {component_name} not found in state configuration"
                )

            # Get base command and config
            component_info = state["components"][f"{self.namespace}_{component_name}"]
            base_cmd = component_info["cmd"].split("--worker-env")[0].strip()
            service_config = state["environment"].get("DYNAMO_SERVICE_CONFIG")

            # Build environment
            watcher_env = os.environ.copy()
            if component_name in ["VllmWorker", "PrefillWorker"]:
                available_gpus = await self._get_available_gpus()
                if not available_gpus:
                    raise ValueError("No GPUs available for allocation")
                gpu_id = available_gpus[0]
                watcher_env["CUDA_VISIBLE_DEVICES"] = gpu_id

            watcher_env["DYNAMO_SERVICE_CONFIG"] = service_config

            # Build worker env list and command
            worker_env_list = [watcher_env]
            worker_env_arg = json.dumps(worker_env_list)
            full_cmd = f"{base_cmd} --worker-env '{worker_env_arg}'"

            # Add watcher through circus controller
            success = await self.circus.add_watcher(
                name=watcher_name, cmd=full_cmd, env=watcher_env, singleton=True
            )

            if success:
                # Update state with new component
                resources = {}
                if component_name in ["VllmWorker", "PrefillWorker"]:
                    resources["allocated_gpus"] = [gpu_id]

                state["components"][watcher_name] = {
                    "watcher_name": watcher_name,
                    "cmd": full_cmd,
                    "resources": resources,
                }
                await self.save_state(state)

            return success

        except Exception as e:
            logger.error(f"Failed to add component {component_name}: {e}")
            if component_name in ["VllmWorker", "PrefillWorker"]:
                await self._release_gpus(component_name)
            return False

    async def _release_gpus(self, component_name: str) -> bool:
        """Release GPUs allocated to a component.

        Args:
            component_name: Name of the component to release GPUs from

        Returns:
            True if GPUs were released successfully
        """
        try:
            logger.info(f"Attempting to release GPUs for component {component_name}")
            state = await self.load_state()
            logger.debug(f"Loaded state for namespace {self.namespace}")
            matching_components = {}

            base_name = f"{self.namespace}_{component_name}"
            base_name_with_underscore = f"{base_name}_"
            logger.debug(f"Looking for components matching {base_name} or {base_name_with_underscore}")

            for watcher_name in state["components"].keys():
                if watcher_name == base_name:  # Exact match for non-numbered watchers
                    logger.debug(f"Found exact match: {watcher_name}")
                    matching_components[0] = watcher_name
                elif watcher_name.startswith(base_name_with_underscore):
                    try:
                        suffix = int(
                            watcher_name.replace(base_name_with_underscore, "")
                        )
                        logger.debug(f"Found numbered match: {watcher_name} with suffix {suffix}")
                        matching_components[suffix] = watcher_name
                    except ValueError:
                        logger.debug(f"Skipping invalid suffix in {watcher_name}")
                        continue

            if not matching_components:
                logger.error(f"No matching components found for {component_name}")
                return False

            highest_suffix = max(matching_components.keys())
            target_watcher = matching_components[highest_suffix]
            logger.info(f"Selected {target_watcher} for GPU release")

            if target_watcher in state["components"]:
                component_info = state["components"][target_watcher]
                if "resources" in component_info:
                    component_info["resources"] = {"allocated_gpus": []}
                    logger.info(f"Cleared GPU allocations for {target_watcher}")
                await self.save_state(state)
                logger.info("Successfully saved updated state")
                return True

            logger.error(f"Target watcher {target_watcher} not found in state components")
            return False

        except Exception as e:
            logger.error(f"Failed to release GPUs for {component_name}: {e}")
            return False

    async def remove_component(self, component_name: str) -> bool:
        """Remove a component from the planner"""
        try:
            logger.info(f"Attempting to remove component {component_name}")
            state = await self.load_state()
            logger.debug(f"Loaded state for namespace {self.namespace}")
            matching_components = {}

            base_name = f"{self.namespace}_{component_name}"
            base_name_with_underscore = f"{base_name}_"
            logger.debug(f"Looking for components matching {base_name} or {base_name_with_underscore}")

            for watcher_name in state["components"].keys():
                if watcher_name == base_name:  # Exact match for non-numbered watchers
                    logger.debug(f"Found exact match: {watcher_name}")
                    matching_components[0] = watcher_name
                elif watcher_name.startswith(base_name_with_underscore):
                    try:
                        suffix = int(
                            watcher_name.replace(base_name_with_underscore, "")
                        )
                        logger.debug(f"Found numbered match: {watcher_name} with suffix {suffix}")
                        matching_components[suffix] = watcher_name
                    except ValueError:
                        logger.debug(f"Skipping invalid suffix in {watcher_name}")
                        continue

            if not matching_components:
                logger.error(f"No matching components found for {component_name}")
                return False

            highest_suffix = max(matching_components.keys())
            target_watcher = matching_components[highest_suffix]
            logger.info(f"Selected {target_watcher} for removal")

            success = await self.circus.remove_watcher(
                name=target_watcher,
            )
            print("SUCC",success)
            logger.info(f"Circus remove_watcher for {target_watcher} {'succeeded' if success else 'failed'}")

            if success:
                logger.info(f"Attempting to release GPUs for {component_name}")
                gpu_release = await self._release_gpus(component_name)
                logger.info(f"GPU release {'succeeded' if gpu_release else 'failed'}")

            return success

        except Exception as e:
            logger.error(f"Failed to remove component {component_name}: {e}")
            return False

    def __del__(self):
        """Cleanup circus controller connection on deletion."""
        if hasattr(self, "circus"):
            self.circus.close()
