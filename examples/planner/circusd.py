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
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class CircusController:
    """Wrapper over circusctl commands to manage Dynamo services."""
    
    def __init__(self, endpoint: str):
        """Initialize CircusController with circus endpoint.
        
        Args:
            endpoint: The circus endpoint (e.g., tcp://127.0.0.1:54927)
        """
        self.endpoint = endpoint
    
    @classmethod
    def from_state_file(cls, namespace: str) -> "CircusController":
        """Create a CircusController from a Dynamo state file.
        
        Args:
            namespace: The Dynamo namespace
            
        Returns:
            CircusController instance
        """
        state_file = Path.home() / ".dynamo" / "state" / f"{namespace}.json"
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        endpoint = state.get("circus_endpoint")
        if not endpoint:
            raise ValueError(f"No endpoint found in state file: {state_file}")
        
        return cls(endpoint)
    
    async def list_watchers(self) -> List[str]:
        """List all watchers (components) managed by circus.
        
        Returns:
            List of watcher names
        """
        cmd = ["circusctl", "--endpoint", self.endpoint, "list"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error(f"Failed to list watchers: {error}")
            raise RuntimeError(f"Failed to list watchers: {error}")
        
        output = stdout.decode().strip()
        # Output is comma-separated list
        return [name.strip() for name in output.split(',') if name.strip()]
    
    async def get_watcher_info(self, watcher_name: str) -> Dict:
        """Get detailed information about a watcher.
        
        Args:
            watcher_name: Name of the watcher
            
        Returns:
            Dictionary containing watcher information
        """
        cmd = ["circusctl", "--endpoint", self.endpoint, "stats", watcher_name]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error(f"Failed to get info for {watcher_name}: {error}")
            raise RuntimeError(f"Failed to get info for {watcher_name}: {error}")
        
        output = stdout.decode().strip()
        try:
            # Try to parse as JSON, but handle possible formatting issues
            return json.loads(output)
        except json.JSONDecodeError:
            # If cannot parse as JSON, return as text
            return {"output": output}
    
    async def get_watcher_processes(self, watcher_name: str) -> int:
        """Get the number of processes for a watcher.
        
        Args:
            watcher_name: Name of the watcher
            
        Returns:
            Number of processes
        """
        cmd = ["circusctl", "--endpoint", self.endpoint, "numprocesses", watcher_name]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error(f"Failed to get process count for {watcher_name}: {error}")
            raise RuntimeError(f"Failed to get process count for {watcher_name}: {error}")
        
        output = stdout.decode().strip()
        try:
            return int(output)
        except ValueError:
            logger.error(f"Unexpected output format: {output}")
            raise ValueError(f"Unexpected output format: {output}")
    
    async def add_watcher(self, watcher_name: str, command: str, start: bool = True) -> bool:
        """Add a new watcher to circus.
        
        Args:
            watcher_name: Name of the watcher
            command: The command to run
            start: Whether to start the watcher immediately
            
        Returns:
            True if successful
        """
        cmd = ["circusctl", "--endpoint", self.endpoint, "add"]
        if start:
            cmd.append("--start")
        cmd.extend([watcher_name, command])
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error(f"Failed to add watcher {watcher_name}: {error}")
            raise RuntimeError(f"Failed to add watcher {watcher_name}: {error}")
        
        return True
    
    async def remove_watcher(self, watcher_name: str) -> bool:
        """Remove a watcher from circus.
        
        Args:
            watcher_name: Name of the watcher
            
        Returns:
            True if successful
        """
        cmd = ["circusctl", "--endpoint", self.endpoint, "rm", watcher_name]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error(f"Failed to remove watcher {watcher_name}: {error}")
            raise RuntimeError(f"Failed to remove watcher {watcher_name}: {error}")
        
        return True
    
    async def scale_watcher(self, watcher_name: str, num_processes: int) -> bool:
        """Scale a watcher to specified number of processes.
        
        Args:
            watcher_name: Name of the watcher
            num_processes: Target number of processes
            
        Returns:
            True if successful
        """
        cmd = ["circusctl", "--endpoint", self.endpoint, "set", watcher_name, "numprocesses", str(num_processes)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error(f"Failed to scale watcher {watcher_name}: {error}")
            raise RuntimeError(f"Failed to scale watcher {watcher_name}: {error}")
        
        return True
    
    async def restart_watcher(self, watcher_name: str) -> bool:
        """Restart a watcher.
        
        Args:
            watcher_name: Name of the watcher
            
        Returns:
            True if successful
        """
        cmd = ["circusctl", "--endpoint", self.endpoint, "restart", watcher_name]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error(f"Failed to restart watcher {watcher_name}: {error}")
            raise RuntimeError(f"Failed to restart watcher {watcher_name}: {error}")
        
        return True
    
    async def get_watcher_logs(self, watcher_name: str, lines: int = 100) -> List[str]:
        """Get logs for a watcher.
        
        Args:
            watcher_name: Name of the watcher
            lines: Number of lines to return
            
        Returns:
            List of log lines
        """
        # This is a simplified implementation - actual implementation may need to 
        # parse log files directly if circusctl doesn't provide this functionality
        
        cmd = ["circusctl", "--endpoint", self.endpoint, "dumpconfig"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error = stderr.decode().strip()
            logger.error(f"Failed to get config: {error}")
            raise RuntimeError(f"Failed to get config: {error}")
        
        # Parse the output to find log file location
        # This is a simplification - actual implementation would need to parse the config properly
        
        # For now, return a placeholder
        return [f"Log retrieval not fully implemented for {watcher_name}"]