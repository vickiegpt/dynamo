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
from pathlib import Path
from typing import Any, Dict, List, Optional

from circusd import CircusController
from planner_connector import PlannerConnector

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
        self.circus = CircusController.from_state_file(namespace)
        self.state_file = Path.home() / ".dynamo" / "state" / f"{namespace}.json"

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

    async def get_available_gpus(self) -> List[str]:
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

    async def allocate_gpus(self, component_name: str, num_gpus: int) -> List[str]:
        """Allocate specified number of GPUs to component.

        Args:
            component_name: Name of the component
            num_gpus: Number of GPUs to allocate

        Returns:
            List of allocated GPU IDs
        """
        try:
            available_gpus = await self.get_available_gpus()
            if len(available_gpus) < num_gpus:
                raise ValueError(
                    f"Not enough GPUs available. Requested: {num_gpus}, Available: {len(available_gpus)}"
                )

            # Take the first num_gpus available
            allocated = available_gpus[:num_gpus]

            # Update state
            state = await self.load_state()
            watcher_name = f"{self.namespace}_{component_name}"

            if watcher_name not in state["components"]:
                raise ValueError(f"Component {component_name} not found in state")

            # Update component resources
            if "resources" not in state["components"][watcher_name]:
                state["components"][watcher_name]["resources"] = {}
            state["components"][watcher_name]["resources"]["allocated_gpus"] = allocated

            # Save state
            await self.save_state(state)
            return allocated
        except Exception as e:
            logger.error(f"Failed to allocate GPUs for {component_name}: {e}")
            return []

    async def get_component_replicas(self, component_name: str) -> int:
        """Get the number of replicas for a component.

        Args:
            component_name: Name of the component

        Returns:
            Number of replicas
        """
        # Normalize component name to match circus watcher format
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
            # find max suffix
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

            # Get base command from state
            component_info = state["components"][f"{self.namespace}_{component_name}"]
            base_cmd = component_info["cmd"].split("--worker-env")[0].strip()

            # Get service config
            service_config = state["environment"].get("DYNAMO_SERVICE_CONFIG")
            if not service_config:
                raise ValueError("DYNAMO_SERVICE_CONFIG not found in environment")

            # If GPU component, handle GPU allocation
            worker_env = {}
            if component_name in ["VllmWorker", "PrefillWorker"]:
                available_gpus = await self.get_available_gpus()
                if not available_gpus:
                    raise ValueError("No GPUs available for allocation")

                gpu_id = available_gpus[0]
                worker_env["CUDA_VISIBLE_DEVICES"] = gpu_id

            # Add service config to worker env
            worker_env["DYNAMO_SERVICE_CONFIG"] = service_config

            # Create the worker env array and convert to JSON string
            worker_env_list = [worker_env]
            worker_env_json = json.dumps(worker_env_list)

            # Construct the full command with properly escaped worker-env
            full_cmd = f"{base_cmd} --worker-env '{worker_env_json}'"

            logger.info(f"Full command: {full_cmd}")

            # Add the watcher
            success = await self.circus.add_watcher(
                watcher_name=watcher_name, command=full_cmd, start=True
            )

            # If watcher was added successfully, update the state file
            if success:
                # Update state with new component and GPU allocation if applicable
                resources = {}
                if component_name in ["VllmWorker", "PrefillWorker"]:
                    resources["allocated_gpus"] = [gpu_id]

                # Add the new component to state
                state["components"][watcher_name] = {
                    "watcher_name": watcher_name,
                    "cmd": full_cmd,
                    "resources": resources,
                }

                # Save updated state
                await self.save_state(state)
                logger.info(f"Updated state file with new component: {watcher_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to add component {component_name}: {e}")
            if component_name in ["VllmWorker", "PrefillWorker"]:
                await self.release_gpus(component_name)
            return False

    async def remove_component(self, component_name: str) -> bool:
        """Remove a component from the planner"""
        try:
            state = await self.load_state()
            matching_components = {}

            # TODO: assume we cannot remove the min replicas (dynamo_vllmworker or dynamo_prefillworker)
            base_name = f"{self.namespace}_{component_name}_"
            for watcher_name in state["components"].keys():
                if watcher_name.startswith(base_name):
                    try:
                        suffix = int(watcher_name.replace(base_name, ""))
                        matching_components[suffix] = watcher_name
                    except ValueError:
                        # note a numeric suffix - skip
                        continue

            # Remove the component with the highest suffix
            if not matching_components:
                logger.error(f"No matching components found for {component_name}")
                return False

            # get highest suffix
            highest_suffix = max(matching_components.keys())
            target_watcher = matching_components[highest_suffix]

            logger.info(f"Removing component {target_watcher}")

            # check gpu allocation and release if needed
            component_info = state["components"].get(target_watcher, {})
            resources = component_info.get("resources", {})

            success = await self.circus.remove_watcher(target_watcher)

            if success:
                if target_watcher in state["components"]:
                    del state["components"][target_watcher]
                    logger.info(f"Removed component {target_watcher} from state")

                    # save updated state
                    await self.save_state(state)
                    logger.info(
                        f"Updated state file with removed component: {target_watcher}"
                    )
                else:
                    logger.warning(f"Component {target_watcher} not found in state")

            return success
        except Exception as e:
            logger.error(f"Failed to remove component {component_name}: {e}")
            return False
