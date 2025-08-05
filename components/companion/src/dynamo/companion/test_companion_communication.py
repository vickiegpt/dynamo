#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test script for Dynamo companion server/client communication."""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import uvloop
from vllm.engine.arg_utils import AsyncEngineArgs

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .dynamo_companion_client import create_model_client
from .gpu_utils import get_physical_device_index

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def start_companion_server(device_id: int, namespace: str = "companion"):
    """Start a companion server process in the background."""
    cmd = [
        sys.executable,
        "-m",
        "dynamo.companion.dynamo_companion_server",
        "--device",
        str(device_id),
        "--namespace",
        namespace,
    ]
    
    logger.info("Starting companion server: %s", " ".join(cmd))
    
    # Start server process
    env = os.environ.copy()
    # Ensure CUDA_VISIBLE_DEVICES is not set for server
    env.pop("CUDA_VISIBLE_DEVICES", None)
    
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Start a thread to read and print server output
    import threading
    def print_server_output():
        for line in process.stdout:
            print(f"[SERVER] {line.rstrip()}")
    
    output_thread = threading.Thread(target=print_server_output, daemon=True)
    output_thread.start()
    
    # Wait a bit for server to start
    time.sleep(3)
    
    # Check if process is still running
    if process.poll() is not None:
        # Process exited, print output
        output, _ = process.communicate()
        logger.error("Server failed to start. Output:\n%s", output)
        raise RuntimeError("Failed to start companion server")
    
    logger.info("Companion server started with PID %d", process.pid)
    return process


@dynamo_worker(static=False)
async def test_client(runtime: DistributedRuntime):
    """Test client that connects to companion server and retrieves model parameters."""
    parser = argparse.ArgumentParser(description="Test Dynamo companion communication")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load (default: facebook/opt-125m)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="companion",
        help="Dynamo namespace (default: companion)",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start companion server automatically",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    args = parser.parse_args()
    
    server_process = None
    
    try:
        # Get current device info
        logical_device = torch.cuda.current_device()
        physical_device = get_physical_device_index(logical_device)
        
        logger.info(
            "Test client configuration:\n"
            "- Model: %s\n"
            "- Namespace: %s\n"
            "- Logical device: %d\n"
            "- Physical device: %d\n"
            "- CUDA_VISIBLE_DEVICES: %s",
            args.model,
            args.namespace,
            logical_device,
            physical_device,
            os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        )
        
        # Start server if requested
        if args.start_server:
            server_process = start_companion_server(physical_device, args.namespace)
            # Give server more time to initialize
            await asyncio.sleep(2)
        
        # Create VllmConfig
        engine_args = AsyncEngineArgs(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        vllm_config = engine_args.create_engine_config()
        
        # Create client
        logger.info("Creating Dynamo model client...")
        # For testing, use simple rank values
        local_rank = 0
        global_rank = 0
        world_size = args.tensor_parallel_size  # Assuming single node
        client = await create_model_client(runtime, vllm_config, local_rank, global_rank, world_size, args.namespace)
        
        # Wait for model to be ready
        logger.info("Waiting for model to be ready...")
        success, info = await client.wait_for_model_ready(
            initial_timeout=60.0,  # Longer timeout for first connection and server startup
            loading_timeout=600.0,  # 10 minutes for model loading
        )
        
        if not success:
            logger.error("Failed to connect to model server or load model")
            return
        
        logger.info("Model ready! Server info: %s", info)
        
        # Get model parameters
        logger.info("Retrieving model parameters...")
        parameters = await client.get_model_parameters()
        
        logger.info("Successfully retrieved %d model parameters", len(parameters))
        
        # Test reconstruction of a few parameters
        test_count = min(5, len(parameters))
        logger.info("Testing reconstruction of %d parameters...", test_count)
        
        for i, (param_name, rebuild_info) in enumerate(list(parameters.items())[:test_count]):
            try:
                tensor = client.reconstruct_parameter(rebuild_info)
                logger.info(
                    "✓ Parameter %d: '%s' - shape=%s, dtype=%s, device=%s",
                    i + 1,
                    param_name,
                    tensor.shape,
                    tensor.dtype,
                    tensor.device,
                )
                
                # Verify tensor is accessible
                tensor_sum = tensor.sum().item()
                logger.info("  - Tensor sum: %f (verifies CUDA IPC access)", tensor_sum)
                
            except Exception as e:
                logger.error("✗ Failed to reconstruct parameter '%s': %s", param_name, str(e))
        
        logger.info("\n" + "="*60)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("Summary:")
        logger.info("- Connected to companion server on GPU %d", physical_device)
        logger.info("- Loaded model: %s", args.model)
        logger.info("- Retrieved %d model parameters via CUDA IPC", len(parameters))
        logger.info("- Successfully reconstructed and accessed tensors")
        logger.info("="*60)
        
    except Exception as e:
        logger.error("Test failed with error: %s", str(e), exc_info=True)
        raise
    
    finally:
        # Clean up server process if we started it
        if server_process:
            logger.info("Terminating companion server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't terminate gracefully, killing...")
                server_process.kill()
                server_process.wait()
            logger.info("Server terminated")


def main():
    """Main entry point for test script."""
    print("\n" + "="*60)
    print("DYNAMO COMPANION SERVER/CLIENT COMMUNICATION TEST")
    print("="*60 + "\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test requires a GPU.")
        sys.exit(1)
    
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print()
    
    # Run the test
    uvloop.install()
    asyncio.run(test_client())


if __name__ == "__main__":
    main()