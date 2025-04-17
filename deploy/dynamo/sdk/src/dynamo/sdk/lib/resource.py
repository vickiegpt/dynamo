#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

# TODO: this should be used for planner as well and should leverage proper nvml bindings

from __future__ import annotations

import functools
import logging
import math
import os
import re
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import psutil
try:
    import pynvml
    PYNVML_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
NVIDIA_GPU = "nvidia.com/gpu"

class ResourceError(Exception):
    """Base exception for resource-related errors."""
    pass

######COPIED#######
@functools.lru_cache(maxsize=1)
def query_cgroup_cpu_count() -> float:
    """
    Query active CPU processor count using cgroup v1/v2 API, similar to OpenJDK's implementation.
    Returns the number of CPUs available to the current process.
    """
    # Query active cpu processor count using cgroup v1 API, based on OpenJDK
    # implementation for `active_processor_count` using cgroup v1:
    # https://github.com/openjdk/jdk/blob/master/src/hotspot/os/linux/cgroupSubsystem_linux.cpp
    # For cgroup v2, see:
    # https://github.com/openjdk/jdk/blob/master/src/hotspot/os/linux/cgroupV2Subsystem_linux.cpp
    def _read_cgroup_file(filename: str) -> float:
        with open(filename, "r", encoding="utf-8") as f:
            return int(f.read().strip())

    cgroup_root = "/sys/fs/cgroup/"
    cfs_quota_us_file = os.path.join(cgroup_root, "cpu", "cpu.cfs_quota_us")
    cfs_period_us_file = os.path.join(cgroup_root, "cpu", "cpu.cfs_period_us")
    cpu_max_file = os.path.join(cgroup_root, "cpu.max")

    quota = None

    if os.path.exists(cfs_quota_us_file) and os.path.exists(cfs_period_us_file):
        try:
            quota = _read_cgroup_file(cfs_quota_us_file) / _read_cgroup_file(
                cfs_period_us_file
            )
        except FileNotFoundError as err:
            logger.warning("Caught exception while calculating CPU quota: %s", err)
    # reading from cpu.max for cgroup v2
    elif os.path.exists(cpu_max_file):
        try:
            with open(cpu_max_file, "r", encoding="utf-8") as max_file:
                cfs_string = max_file.read()
                quota_str, period_str = cfs_string.split()
                if quota_str.isnumeric() and period_str.isnumeric():
                    quota = float(quota_str) / float(period_str)
                else:
                    # quota_str is "max" meaning the cpu quota is unset
                    quota = None
        except FileNotFoundError as err:
            logger.warning("Caught exception while calculating CPU quota: %s", err)
    if quota is not None and quota < 0:
        quota = None
    elif quota == 0:
        quota = 1

    os_cpu_count = float(os.cpu_count() or 1.0)

    limit_count = math.inf

    if quota:
        limit_count = quota

    return float(min(limit_count, os_cpu_count))


@functools.lru_cache(maxsize=1)
def query_os_cpu_count() -> int:
    """Return the number of CPUs available in the system."""
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count
    logger.warning("Failed to determine CPU count, using 1 as default.")
    return 1
######COPIED#######

@dataclass
class GPUProcess:
    """Information about a process running on a GPU."""
    
    pid: int
    used_memory: int  # in bytes
    name: str = ""
    
    def __post_init__(self):
        """Get process name if available."""
        try:
            self.name = psutil.Process(self.pid).name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


class GPUInfo:
    """Information about a specific GPU device."""
    
    def __init__(self, index: int, total_memory: int, name: str, uuid: str):
        self.index = index
        self.total_memory = total_memory  # in bytes
        self.name = name
        self.uuid = uuid
        self.available = True  # Can be set to False if GPU is reserved/in use
        self.temperature = 0  # in Celsius
        self.utilization = 0  # in percent (0-100)
        self.processes: list[GPUProcess] = []
        
    def __repr__(self) -> str:
        return f"GPUInfo(index={self.index}, name='{self.name}', total_memory={self.total_memory/1024/1024:.0f}MB, available={self.available})"


class GPUManager:
    """
    Manages GPU resources using NVML.
    
    This class provides methods to:
    - Discover available GPUs
    - Query GPU properties and status
    - Track GPU processes
    - Allocate and release GPUs
    - Generate CUDA_VISIBLE_DEVICES environment variables
    """
    
    def __init__(self):
        """Initialize the GPU manager."""
        self.gpus: list[GPUInfo] = []
        self._initialized = False
        # List to track fractional GPU allocations
        # Each item is (gpu_index, fraction_used, fraction_size)
        # E.g. (0, 0.5, 0.5) means GPU 0 has 0.5 used with fraction size of 0.5
        self._gpu_fractions: list[tuple[int, float, float]] = []
        self._init_nvml()
        
    def _init_nvml(self):
        """Initialize NVML and discover GPUs."""
        if not PYNVML_AVAILABLE:
            logger.warning("PyNVML not available. GPU functionality will be limited.")
            return
            
        try:
            pynvml.nvmlInit()
            self._initialized = True
            self._discover_gpus()
        except (pynvml.NVMLError_LibraryNotFound, pynvml.NVMLError_DriverNotLoaded, OSError) as e:
            logger.warning(f"Failed to initialize NVML: {e}")
            self._initialized = False
    
    def __del__(self):
        """Clean up NVML."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pylint: disable=broad-except
                pass
    
    def _discover_gpus(self):
        """Discover available GPUs and their properties."""
        if not self._initialized:
            return
            
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            self.gpus = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                
                gpu_info = GPUInfo(
                    index=i,
                    total_memory=memory_info.total,
                    name=name,
                    uuid=uuid
                )
                
                # Get additional GPU information if available
                try:
                    gpu_info.temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError:
                    logger.debug(f"Could not get temperature for GPU {i}")
                
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info.utilization = utilization.gpu
                except pynvml.NVMLError:
                    logger.debug(f"Could not get utilization for GPU {i}")
                
                # Get processes running on GPU
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    gpu_info.processes = [
                        GPUProcess(pid=p.pid, used_memory=p.usedGpuMemory)
                        for p in processes
                    ]
                except pynvml.NVMLError:
                    logger.debug(f"Could not get processes for GPU {i}")
                
                self.gpus.append(gpu_info)
                
            logger.info(f"Discovered {len(self.gpus)} GPUs")
        except pynvml.NVMLError as e:
            logger.warning(f"Error discovering GPUs: {e}")
    
    def update_gpu_stats(self):
        """Update GPU statistics (utilization, memory, temperature, etc.)."""
        if not self._initialized:
            return
        
        for gpu in self.gpus:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
                
                # Update memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu.total_memory = memory_info.total
                
                # Update temperature
                try:
                    gpu.temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError:
                    pass
                
                # Update utilization
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu.utilization = utilization.gpu
                except pynvml.NVMLError:
                    pass
                
                # Update processes
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    gpu.processes = [
                        GPUProcess(pid=p.pid, used_memory=p.usedGpuMemory)
                        for p in processes
                    ]
                except pynvml.NVMLError:
                    pass
                
            except pynvml.NVMLError as e:
                logger.warning(f"Error updating GPU {gpu.index} stats: {e}")
    
    def get_gpu_count(self) -> int:
        """Return the number of available GPUs."""
        return len(self.gpus)
    
    def get_available_gpus(self) -> list[int]:
        """Return a list of available GPU indices."""
        return [gpu.index for gpu in self.gpus if gpu.available]
    
    def get_gpu_memory(self, index: int) -> tuple[int, int]:
        """
        Return (total memory, free memory) in bytes for a specific GPU.
        
        Args:
            index: GPU index
            
        Returns:
            Tuple of (total memory, free memory) in bytes
        """
        if not self._initialized or index >= len(self.gpus):
            return (0, 0)
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (memory_info.total, memory_info.free)
        except pynvml.NVMLError as e:
            logger.warning(f"Error getting GPU memory for GPU {index}: {e}")
            return (0, 0)
    
    def get_gpu_utilization(self, index: int) -> int:
        """
        Return GPU utilization percentage for a specific GPU.
        
        Args:
            index: GPU index
            
        Returns:
            GPU utilization percentage (0-100)
        """
        if not self._initialized or index >= len(self.gpus):
            return 0
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu  # Returns GPU utilization percentage (0-100)
        except pynvml.NVMLError as e:
            logger.warning(f"Error getting GPU utilization for GPU {index}: {e}")
            return 0
    
    def get_gpu_temperature(self, index: int) -> int:
        """
        Return GPU temperature for a specific GPU.
        
        Args:
            index: GPU index
            
        Returns:
            GPU temperature in Celsius
        """
        if not self._initialized or index >= len(self.gpus):
            return 0
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError as e:
            logger.warning(f"Error getting GPU temperature for GPU {index}: {e}")
            return 0
    
    def get_gpu_processes(self, index: int) -> list[GPUProcess]:
        """
        Return processes running on a specific GPU.
        
        Args:
            index: GPU index
            
        Returns:
            List of processes running on the GPU
        """
        if not self._initialized or index >= len(self.gpus):
            return []
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            return [
                GPUProcess(pid=p.pid, used_memory=p.usedGpuMemory)
                for p in processes
            ]
        except pynvml.NVMLError as e:
            logger.warning(f"Error getting GPU processes for GPU {index}: {e}")
            return []
    
    def assign_gpus(self, count: float) -> list[int]:
        """
        Assign GPUs for use. It can handle fractional GPU requests.
        
        Args:
            count: Number of GPUs to assign (can be fractional)
            
        Returns:
            List of GPU indices that were assigned
        """
        available_gpus = self.get_available_gpus()
        
        if count > len(available_gpus):
            logger.warning(
                f"Requested {count} GPUs, but only {len(available_gpus)} are available. "
                "Service may fail due to inadequate GPU resources."
            )
        
        # Handle fractional GPU allocation
        if count < 1:
            # Try to find a GPU with the same fraction size
            try:
                # Find a GPU where we've already used the same fraction size
                gpu_idx, used_fraction = next(
                    (idx, used) for idx, used, frac_size in self._gpu_fractions 
                    if frac_size == count and used < 1.0
                )
                
                # Update the usage for this GPU
                for i, (idx, used, frac_size) in enumerate(self._gpu_fractions):
                    if idx == gpu_idx and frac_size == count:
                        new_used = used + count
                        if new_used > 1.0:
                            new_used = 1.0  # Cap at 1.0
                        self._gpu_fractions[i] = (idx, new_used, frac_size)
                        break
                
                return [gpu_idx]
            except StopIteration:
                # No existing fraction of this size, find a free GPU
                if available_gpus:
                    gpu_idx = available_gpus[0]
                    self._gpu_fractions.append((gpu_idx, count, count))
                    return [gpu_idx]
                else:
                    # No available GPUs, return the first GPU (or log warning)
                    if self.gpus:
                        logger.warning("No available GPUs, using GPU 0 by default")
                        self._gpu_fractions.append((0, count, count))
                        return [0]
                    else:
                        logger.error("No GPUs available for allocation")
                        return []
        
        # Integer GPU allocation
        if count >= 1:
            if int(count) != count:
                raise ResourceError("Fractional GPU count greater than 1 is not supported")
                
            count_int = int(count)
            assigned_gpus = available_gpus[:count_int]
            
            # Mark these GPUs as fully used
            for gpu_idx in assigned_gpus:
                # Check if this GPU is already in _gpu_fractions
                if not any(idx == gpu_idx for idx, _, _ in self._gpu_fractions):
                    self._gpu_fractions.append((gpu_idx, 1.0, 1.0))
                else:
                    # Update the existing entry
                    for i, (idx, _, frac_size) in enumerate(self._gpu_fractions):
                        if idx == gpu_idx:
                            self._gpu_fractions[i] = (idx, 1.0, frac_size)
                
                # Mark this GPU as unavailable for future requests
                for gpu in self.gpus:
                    if gpu.index == gpu_idx:
                        gpu.available = False
            
            return assigned_gpus
            
        return []
    
    def get_best_gpu_for_memory(self, required_memory: int) -> int:
        """
        Return the index of the GPU with the most available memory that meets the requirement.
        
        Args:
            required_memory: Required memory in bytes
            
        Returns:
            GPU index, or -1 if no suitable GPU was found
        """
        if not self._initialized:
            return -1
            
        best_gpu = -1
        max_free = 0
        
        for gpu in self.gpus:
            if not gpu.available:
                continue
                
            _, free = self.get_gpu_memory(gpu.index)
            if free > required_memory and free > max_free:
                max_free = free
                best_gpu = gpu.index
        
        return best_gpu

    def reset_allocations(self):
        """Reset all GPU allocations."""
        self._gpu_fractions = []
        for gpu in self.gpus:
            gpu.available = True
    
    def get_gpu_stats(self) -> list[dict[str, t.Any]]:
        """
        Get detailed statistics for all GPUs.
        
        Returns:
            List of dictionaries with GPU statistics
        """
        self.update_gpu_stats()
        
        stats = []
        for gpu in self.gpus:
            total_memory, free_memory = self.get_gpu_memory(gpu.index)
            stats.append({
                "index": gpu.index,
                "name": gpu.name,
                "uuid": gpu.uuid,
                "total_memory": total_memory,
                "free_memory": free_memory,
                "used_memory": total_memory - free_memory,
                "memory_utilization": (total_memory - free_memory) / total_memory * 100 if total_memory > 0 else 0,
                "gpu_utilization": gpu.utilization,
                "temperature": gpu.temperature,
                "process_count": len(gpu.processes),
                "processes": [
                    {
                        "pid": process.pid,
                        "name": process.name,
                        "used_memory": process.used_memory
                    }
                    for process in gpu.processes
                ],
                "available": gpu.available
            })
        
        return stats


def get_cpu_count() -> float:
    """Return the number of CPUs available to the current process."""
    if psutil.POSIX:
        return query_cgroup_cpu_count()
    else:
        return float(query_os_cpu_count())

def system_resources() -> dict[str, t.Any]:
    """
    Get available system resources (CPU and GPU).
    
    Returns:
        Dictionary of resources with keys 'cpu' and 'nvidia.com/gpu'
    """
    resources = {}
    
    # Get CPU resources
    resources['cpu'] = get_cpu_count()
    
    # Get GPU resources
    gpu_manager = GPUManager()
    resources[NVIDIA_GPU] = gpu_manager.get_available_gpus()
    
    return resources 