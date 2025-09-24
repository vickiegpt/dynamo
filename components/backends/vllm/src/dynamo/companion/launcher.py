# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launcher for vLLM companion processes."""

import logging
from typing import Optional
import multiprocessing as mp
import torch

# Import companion modules from vLLM
from vllm.companion.multiproc_coordinator import run_coordinator
from vllm.companion.utils import get_free_port
from vllm.config import CompanionConfig

logger = logging.getLogger(__name__)


class CompanionLauncher:
    """Launcher for companion coordinator."""
    
    def __init__(
        self,
        coordinator_port: Optional[int] = None,
        companion_master_port: Optional[int] = None,
    ):
        """
        Initialize the companion launcher.
        
        Args:
            coordinator_port: Port for coordinator. If None, uses default from CompanionConfig.
            companion_master_port: Master port for distributed init. If None, a free port is chosen.
        """
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available but companion processes require GPU access")
        
        # Use default from CompanionConfig if not specified
        if coordinator_port is None:
            coordinator_port = CompanionConfig.coordinator_port
        self.coordinator_port = coordinator_port
        self.companion_master_port = companion_master_port or get_free_port()
        self.coordinator_process: Optional[mp.Process] = None
        
        logger.info(
            "Companion launcher initialized: coordinator_port=%d, master_port=%d",
            self.coordinator_port, self.companion_master_port
        )
    
    def start(self) -> None:
        """Start the companion coordinator in a separate process."""
        if self.coordinator_process is not None and self.coordinator_process.is_alive():
            logger.warning("Coordinator already running")
            return
        
        # Start coordinator process
        ctx = mp.get_context('spawn')
        self.coordinator_process = ctx.Process(
            target=run_coordinator,
            args=(self.coordinator_port, self.companion_master_port),
            name="CompanionCoordinator",
            daemon=False  # Not daemon so it can have child processes
        )
        self.coordinator_process.start()
        
        logger.info(
            "Started companion coordinator (PID: %d) on port %d",
            self.coordinator_process.pid, self.coordinator_port
        )
    
    
    def stop(self) -> None:
        """Stop the companion coordinator and all servers."""
        if self.coordinator_process is None or not self.coordinator_process.is_alive():
            logger.warning("Coordinator not running")
            return
        
        logger.info("Stopping companion coordinator...")
        self.coordinator_process.terminate()
        self.coordinator_process.join(timeout=10)
        
        if self.coordinator_process.is_alive():
            logger.warning("Coordinator did not stop gracefully, killing...")
            self.coordinator_process.kill()
            self.coordinator_process.join()
        
        logger.info("Companion coordinator stopped")
    
    def wait(self) -> None:
        """Wait for the coordinator process to complete."""
        if self.coordinator_process is not None:
            self.coordinator_process.join()
    
    def is_running(self) -> bool:
        """Check if the coordinator is running."""
        return (self.coordinator_process is not None and 
                self.coordinator_process.is_alive())
    
    def get_coordinator_address(self) -> str:
        """Get the coordinator address for clients."""
        return f"tcp://127.0.0.1:{self.coordinator_port}"
    


def launch_companion(
    coordinator_port: Optional[int] = None,
    companion_master_port: Optional[int] = None,
) -> CompanionLauncher:
    """
    Launch companion coordinator.
    
    Args:
        coordinator_port: Port for coordinator. If None, a free port is chosen.
        companion_master_port: Master port for distributed init. If None, a free port is chosen.
    
    Returns:
        CompanionLauncher instance.
    """
    # Create launcher
    launcher = CompanionLauncher(
        coordinator_port=coordinator_port,
        companion_master_port=companion_master_port,
    )
    
    # Start coordinator
    launcher.start()
    
    return launcher
