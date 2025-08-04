#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM V1 Modular Deployment Launcher
===================================

Orchestrates the startup of all modular vLLM components using HTTP transport.
This demonstrates a modularized vLLM aggregated service where components like
router, scheduler, and API server are also modularized out.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import threading
from typing import Dict, List

# Add components directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "components"))
from config import VLLMConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


class ComponentManager:
    """Manages a single component process with log streaming."""

    def __init__(self, name: str, port: int, logger: logging.Logger):
        self.name = name
        self.port = port
        self.logger = logger
        self.process: subprocess.Popen = None
        self.log_thread: threading.Thread = None
        self.startup_complete = False

    def start(self, cmd: List[str], cwd: str):
        """Start the component process with log streaming."""
        self.logger.info(f"Starting {self.name} on port {self.port}")
        self.logger.info(f"Command: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Start log streaming thread
            self.log_thread = threading.Thread(target=self._stream_logs, daemon=True)
            self.log_thread.start()

            return self.process

        except Exception as e:
            self.logger.error(f"Failed to start {self.name}: {e}")
            return None

    def _stream_logs(self):
        """Stream component logs to main logger."""
        startup_indicators = [
            "Application startup complete",
            "Uvicorn running on",
            "Component running on port",
            "server initialized successfully",
        ]

        if self.process:
            for line in iter(self.process.stdout.readline, ""):
                if line:
                    line = line.strip()
                    # Mark startup as complete when we see certain indicators
                    if any(indicator in line for indicator in startup_indicators):
                        self.startup_complete = True

                    self.logger.info(f"[{self.name.upper()}] {line}")

    def is_running(self) -> bool:
        """Check if the component process is still running."""
        return self.process is not None and self.process.poll() is None

    def terminate(self):
        """Terminate the component process."""
        if self.process:
            self.logger.info(f"Terminating {self.name} (PID {self.process.pid})")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Force killing {self.name}")
                self.process.kill()


class VLLMModularLauncher:
    """Launcher for modular vLLM deployment with HTTP transport."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.logger = setup_logging()
        self.components: Dict[str, ComponentManager] = {}

        # Component startup order (dependencies matter)
        self.startup_order = [
            "sampler",
            "kv_cache_manager",
            "worker",
            "scheduler",
            "router",
            "frontend",
        ]

        self.logger.info("=" * 60)
        self.logger.info("vLLM V1 Modular Deployment")
        self.logger.info(f"Model: {config.model}")
        self.logger.info(
            "Communication: HTTP transport (clean abstraction for future Dynamo)"
        )
        self.logger.info(f"Components: {len(self.startup_order)} modular components")
        self.logger.info("=" * 60)

    def create_component_cmd(
        self, component_name: str, port: int, extra_args: List[str] = None
    ) -> List[str]:
        """Create command for starting a component."""
        script_path = f"{component_name}.py"

        # Base command
        cmd = [sys.executable, script_path, "--model", self.config.model]

        # Add extra arguments if provided
        if extra_args:
            cmd.extend(extra_args)

        return cmd

    async def start_all_components(self):
        """Start all components in order with proper waiting."""
        self.logger.info("STARTING COMPONENTS")
        self.logger.info("-" * 60)

        components_dir = os.path.join(os.path.dirname(__file__), "components")

        for component in self.startup_order:
            try:
                if component == "sampler":
                    port = self.config.sampler_port
                    manager = ComponentManager(component, port, self.logger)
                    cmd = self.create_component_cmd(component, port)

                elif component == "kv_cache_manager":
                    port = self.config.kv_cache_port
                    manager = ComponentManager(component, port, self.logger)
                    cmd = self.create_component_cmd(component, port)

                elif component == "worker":
                    # Start multiple workers if configured
                    for worker_id in range(self.config.num_workers):
                        port = self.config.get_worker_port(worker_id)
                        worker_name = (
                            f"worker_{worker_id}" if worker_id > 0 else "worker"
                        )
                        manager = ComponentManager(worker_name, port, self.logger)
                        cmd = self.create_component_cmd(
                            "worker", port, ["--worker-id", str(worker_id)]
                        )

                        process = manager.start(cmd, components_dir)
                        if process:
                            self.components[worker_name] = manager

                        # Wait longer for workers (they need to load weights)
                        self.logger.info(
                            f"Waiting for {worker_name} to load model weights..."
                        )
                        await asyncio.sleep(8)
                    continue

                elif component == "scheduler":
                    port = self.config.scheduler_port
                    manager = ComponentManager(component, port, self.logger)
                    cmd = self.create_component_cmd(component, port)

                elif component == "router":
                    port = 8004  # Router port is fixed
                    manager = ComponentManager(component, port, self.logger)
                    cmd = self.create_component_cmd(component, port)

                elif component == "frontend":
                    port = self.config.frontend_port
                    manager = ComponentManager(component, port, self.logger)
                    cmd = self.create_component_cmd(component, port)

                # Start the component
                process = manager.start(cmd, components_dir)
                if process:
                    self.components[component] = manager

                # Wait for startup (longer for workers)
                startup_wait = 5 if component != "worker" else 10
                self.logger.info(f"Waiting {startup_wait}s for {component} startup...")
                await asyncio.sleep(startup_wait)

                # Check if component started successfully
                if not manager.is_running():
                    self.logger.error(f"Component {component} failed to start!")
                    await self.cleanup()
                    return False

            except Exception as e:
                self.logger.error(f"Error starting {component}: {e}")
                await self.cleanup()
                return False

        # Show component summary
        await self.show_component_summary()
        return True

    async def show_component_summary(self):
        """Show summary of running components."""
        self.logger.info("-" * 60)
        self.logger.info("COMPONENT SUMMARY")
        self.logger.info("-" * 60)

        running_count = 0
        for name, manager in self.components.items():
            if manager.is_running():
                status = "RUNNING"
                running_count += 1
            else:
                status = "FAILED"

            self.logger.info(f"{name:<20} Port: {manager.port:<5} Status: {status}")

        self.logger.info(
            f"\nComponents running: {running_count}/{len(self.components)}"
        )

        if running_count == len(self.components):
            self.logger.info("-" * 60)
            self.logger.info("COMMUNICATION ENDPOINTS")
            self.logger.info("-" * 60)
            self.logger.info(
                f"Frontend (Main API)  http://localhost:{self.config.frontend_port}"
            )
            self.logger.info("Router               http://localhost:8004")
            self.logger.info(
                f"Scheduler            http://localhost:{self.config.scheduler_port}"
            )
            self.logger.info(
                f"Workers              http://localhost:{self.config.worker_base_port}+"
            )
            self.logger.info(
                f"KV Cache Manager     http://localhost:{self.config.kv_cache_port}"
            )
            self.logger.info(
                f"Sampler              http://localhost:{self.config.sampler_port}"
            )

            self.logger.info("-" * 60)
            self.logger.info("TEST THE DEPLOYMENT")
            self.logger.info("-" * 60)
            self.logger.info(
                f"curl -X POST http://localhost:{self.config.frontend_port}/v1/chat/completions \\"
            )
            self.logger.info('  -H "Content-Type: application/json" \\')
            self.logger.info(
                f'  -d \'{{"model": "{self.config.model}", "messages": [{{"role": "user", "content": "Hello!"}}]}}\''
            )
            self.logger.info("-" * 60)

    async def monitor_processes(self):
        """Monitor all component processes."""
        self.logger.info("MONITORING COMPONENTS")
        self.logger.info("Press Ctrl+C to shutdown all components")
        self.logger.info("-" * 60)

        try:
            while True:
                # Check if any processes died
                dead_components = []
                for name, manager in self.components.items():
                    if not manager.is_running():
                        dead_components.append(name)

                if dead_components:
                    self.logger.error(f"Components died: {', '.join(dead_components)}")
                    break

                await asyncio.sleep(5)

        except asyncio.CancelledError:
            pass

    async def cleanup(self):
        """Cleanup all running components."""
        self.logger.info("-" * 60)
        self.logger.info("SHUTTING DOWN")
        self.logger.info("-" * 60)

        for name, manager in self.components.items():
            if manager.is_running():
                manager.terminate()

        # Wait a bit for graceful shutdown
        await asyncio.sleep(2)

        self.logger.info("All components shut down")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            # This will be handled by the main loop
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main launcher function."""
    # Load configuration
    config = VLLMConfig.from_env()

    # Create launcher
    launcher = VLLMModularLauncher(config)
    launcher.setup_signal_handlers()

    try:
        # Start all components
        success = await launcher.start_all_components()

        if success:
            # Monitor components
            await launcher.monitor_processes()
        else:
            logging.getLogger(__name__).error("Failed to start all components")

    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Received interrupt, shutting down...")
    except Exception as e:
        logging.getLogger(__name__).error(f"Error: {e}")
    finally:
        await launcher.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
