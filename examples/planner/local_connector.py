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

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from planner_connector import PlannerConnector
from circusd import CircusController

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
            
    async def get_component_replicas(self, component_name: str) -> int:
        """Get the number of replicas for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Number of replicas
        """
        # Normalize component name to match circus watcher format
        watcher_name = f"dynamo_{component_name}"
        
        try:
            return await self.circus.get_watcher_processes(watcher_name)
        except Exception as e:
            logger.error(f"Failed to get replicas for {component_name}: {e}")
            return 0

    async def scale_component(self, component_name: str, replicas: int) -> bool:
        """Scale a component to specified number of replicas.
        
        Args:
            component_name: Name of the component
            replicas: Target number of replicas
            
        Returns:
            True if successful
        """
        # Normalize component name to match circus watcher format
        watcher_name = f"dynamo_{component_name}"
        
        if replicas <= 0:
            # Remove the component if scaling to zero
            return await self.circus.remove_watcher(watcher_name)
        
        try:
            current_replicas = await self.get_component_replicas(component_name)
            
            if current_replicas == 0:
                # Component doesn't exist, need to add it
                state = await self.load_state()
                components = state.get("components", {})
                
                if watcher_name not in components:
                    raise ValueError(f"Component {component_name} not found in state file")
                
                component_info = components[watcher_name]
                command = component_info.get("cmd")
                
                if not command:
                    raise ValueError(f"No command found for {component_name}")
                
                # Add the watcher with specified replicas
                await self.circus.add_watcher(watcher_name, command, start=True)
                if replicas > 1:
                    await self.circus.scale_watcher(watcher_name, replicas)
                return True
            else:
                # Component exists, just scale it
                return await self.circus.scale_watcher(watcher_name, replicas)
        except Exception as e:
            logger.error(f"Failed to scale {component_name}: {e}")
            return False

    async def get_resource_usage(self, component_name: str) -> Dict[str, Any]:
        """Get resource usage for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dictionary with resource usage metrics
        """
        # Normalize component name to match circus watcher format
        watcher_name = f"dynamo_{component_name}"
        
        try:
            stats = await self.circus.get_watcher_info(watcher_name)
            return {
                "cpu": stats.get("cpu", 0),
                "memory": stats.get("mem", 0),
                "processes": await self.get_component_replicas(component_name),
                "status": "running" if await self.get_component_replicas(component_name) > 0 else "stopped"
            }
        except Exception as e:
            logger.error(f"Failed to get resource usage for {component_name}: {e}")
            return {
                "cpu": 0,
                "memory": 0,
                "processes": 0,
                "status": "unknown"
            }

    async def get_system_topology(self) -> Dict[str, Any]:
        """Get system topology information.
        
        Returns:
            Dictionary with topology information
        """
        try:
            state = await self.load_state()
            watchers = await self.circus.list_watchers()
            
            # Filter for dynamo components
            components = [w for w in watchers if w.startswith("dynamo_")]
            
            topology = {
                "namespace": self.namespace,
                "components": {}
            }
            
            for component in components:
                # Strip the dynamo_ prefix
                name = component.replace("dynamo_", "", 1)
                replicas = await self.circus.get_watcher_processes(component)
                resources = await self.get_resource_usage(name)
                
                topology["components"][name] = {
                    "replicas": replicas,
                    "resources": resources
                }
            
            return topology
        except Exception as e:
            logger.error(f"Failed to get system topology: {e}")
            return {"namespace": self.namespace, "components": {}}
            
    async def restart_component(self, component_name: str) -> bool:
        """Restart a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            True if successful
        """
        # Normalize component name to match circus watcher format
        watcher_name = f"dynamo_{component_name}"
        
        try:
            return await self.circus.restart_watcher(watcher_name)
        except Exception as e:
            logger.error(f"Failed to restart {component_name}: {e}")
            return False
            
    async def list_components(self) -> List[Dict[str, Any]]:
        """List all components with their status.
        
        Returns:
            List of component dictionaries with status information
        """
        try:
            watchers = await self.circus.list_watchers()
            
            # Filter for dynamo components
            components = [w for w in watchers if w.startswith("dynamo_")]
            
            result = []
            for component in components:
                # Strip the dynamo_ prefix
                name = component.replace("dynamo_", "", 1)
                replicas = await self.circus.get_watcher_processes(component)
                resources = await self.get_resource_usage(name)
                
                result.append({
                    "name": name,
                    "replicas": replicas,
                    "status": resources["status"],
                    "resources": resources
                })
            
            return result
        except Exception as e:
            logger.error(f"Failed to list components: {e}")
            return []
            
    async def get_component_logs(self, component_name: str, lines: int = 100) -> List[str]:
        """Get logs for a component.
        
        Args:
            component_name: Name of the component
            lines: Number of lines to return
            
        Returns:
            List of log lines
        """
        # Normalize component name to match circus watcher format
        watcher_name = f"dynamo_{component_name}"
        
        try:
            return await self.circus.get_watcher_logs(watcher_name, lines)
        except Exception as e:
            logger.error(f"Failed to get logs for {component_name}: {e}")
            return [f"Error retrieving logs: {str(e)}"]
