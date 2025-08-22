# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-based companion server for CUDA IPC weight sharing."""

import argparse
import asyncio
import logging
import os
import threading
import time
from typing import Optional

import torch
import uvloop
from vllm.config import VllmConfig

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .companion_messages import (
    GetModelParametersRequest,
    ModelParametersResponse,
    StatusUpdateMessage,
)
from .model_instance_manager import ModelInstanceManager

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class DynamoModelServer:
    """Model server that loads models on demand and shares parameters via CUDA IPC."""
    
    def __init__(self, device_id: int, namespace: str = "companion", companion_master_port: int = 29700):
        """Initialize model server for a specific GPU.
        
        Args:
            device_id: Physical GPU device ID to use. This must be the actual
                      physical device ID on the system, not a logical ID from
                      CUDA_VISIBLE_DEVICES mapping.
            namespace: Dynamo namespace for service discovery
        
        Note:
            The server MUST run without CUDA_VISIBLE_DEVICES set to ensure
            correct physical GPU mapping for CUDA IPC between server and client.
        """
        self.device_id = device_id
        self.namespace = namespace
        self.companion_master_port = companion_master_port
        self.current_status = "ready"
        self.status_message = "Server ready, waiting for model request"
        self.loaded_model_name: Optional[str] = None
        self.model_parameters: Optional[dict] = None
        self.model_manager: Optional[ModelInstanceManager] = None
        self.loaded_vllm_config: Optional[VllmConfig] = None
        self.loaded_config_hash: Optional[str] = None
        self.loaded_local_rank: Optional[int] = None
        self.loaded_global_rank: Optional[int] = None
        self.loaded_world_size: Optional[int] = None
        self.load_lock = threading.Lock()
        
        # Check CUDA_VISIBLE_DEVICES
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            logger.error(
                "CUDA_VISIBLE_DEVICES is set to '%s'. "
                "The model server must be run without CUDA_VISIBLE_DEVICES to ensure "
                "correct physical GPU mapping. Please unset it and restart.",
                cuda_visible_devices,
            )
            raise RuntimeError(
                "Model server cannot run with CUDA_VISIBLE_DEVICES set. "
                "Please unset it to ensure correct physical GPU mapping."
            )
        
        # Set the device
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(self.device)
        
        logger.info(
            "Dynamo model server initializing:\n"
            "- Device: cuda:%d\n"
            "- Namespace: %s\n"
            "- Component: model_server_device_%d\n"
            "- Ready to load models on demand",
            self.device_id,
            self.namespace,
            self.device_id,
        )
    
    async def _load_model_async(
        self,
        vllm_config: VllmConfig,
        config_hash: str,
        local_rank: int,
        global_rank: int,
        world_size: int,
    ):
        """Load model asynchronously with proper thread safety."""
        try:
            self.current_status = "loading"
            self.status_message = f"Loading model {vllm_config.model_config.model}"
            
            logger.info(
                "Loading model %s with parallel config: TP=%d, PP=%d, DP=%d, "
                "Expert parallel enabled: %s, "
                "local_rank=%d, global_rank=%d, world_size=%d",
                vllm_config.model_config.model,
                vllm_config.parallel_config.tensor_parallel_size,
                vllm_config.parallel_config.pipeline_parallel_size,
                vllm_config.parallel_config.data_parallel_size,
                vllm_config.parallel_config.enable_expert_parallel,
                local_rank,
                global_rank,
                world_size,
            )
            
            start_time = time.time()
            
            # Step 1: Create model manager
            logger.info("[MAIN-THREAD] Creating ModelInstanceManager")
            self.model_manager = ModelInstanceManager(
                vllm_config=vllm_config,
                device_id=self.device_id,
                local_rank=local_rank,
                global_rank=global_rank,
                world_size=world_size,
                companion_master_port=self.companion_master_port,
            )
            
            # Step 2: Initialize distributed environment in MAIN thread
            # This MUST happen in the main thread to avoid PyTorch distributed deadlocks
            logger.info("[MAIN-THREAD] Initializing distributed environment in main thread")
            self.model_manager.initialize_distributed()
            logger.info("[MAIN-THREAD] ✓ Distributed environment initialized")
            
            # Step 3: Load model weights in thread pool (heavy I/O operation)
            logger.info("[MAIN-THREAD] Starting model weight loading in thread pool")
            loop = asyncio.get_event_loop()
            
            def load_weights_in_thread():
                """Function to run in thread pool - only does I/O, no distributed ops."""
                logger.info("[THREAD-POOL] Loading model weights")
                self.model_manager.load_model_weights()
                logger.info("[THREAD-POOL] ✓ Model weights loaded")
                return self.model_manager.get_model_parameters_ipc_info()
            
            model_parameters = await loop.run_in_executor(None, load_weights_in_thread)
            load_time = time.time() - start_time
            
            # Update server state
            with self.load_lock:
                self.loaded_model_name = vllm_config.model_config.model
                self.model_parameters = model_parameters
                self.loaded_vllm_config = vllm_config
                self.loaded_config_hash = config_hash
                self.loaded_local_rank = local_rank
                self.loaded_global_rank = global_rank
                self.loaded_world_size = world_size
                self.current_status = "loaded"
                self.status_message = (
                    f"Model {vllm_config.model_config.model} loaded successfully "
                    f"in {load_time:.2f} seconds"
                )
            
            logger.info(
                "Model %s loaded successfully in %.2f seconds. "
                "Loaded %d parameters on cuda:%d",
                vllm_config.model_config.model,
                load_time,
                len(model_parameters),
                self.device_id,
            )
            
        except Exception as e:
            logger.error("Failed to load model: %s", str(e), exc_info=True)
            self.current_status = "error"
            self.status_message = f"Failed to load model: {str(e)}"
            raise
    
    @dynamo_endpoint(GetModelParametersRequest, ModelParametersResponse)
    async def get_model_parameters(self, request: GetModelParametersRequest):
        """Handle model parameter requests."""
        # Extract VllmConfig from request (already unpickled by get_vllm_config)
        vllm_config = request.get_vllm_config()
        model_name = vllm_config.model_config.model
        
        logger.info(
            "Received model parameters request:\n"
            "- Model: %s\n"
            "- Client device: %d\n"
            "- Local rank: %d, Global rank: %d, World size: %d",
            model_name,
            request.device_id,
            request.local_rank,
            request.global_rank,
            request.world_size,
        )
        
        # Validate device match
        if request.device_id != self.device_id:
            error_msg = (
                f"Device mismatch: Client is on device {request.device_id} "
                f"but server is on device {self.device_id}. "
                f"CUDA IPC requires both processes to use the same physical GPU."
            )
            logger.error(error_msg)
            yield ModelParametersResponse.from_model_parameters(
                model_parameters=None,
                model_name="",
                device_id=self.device_id,
                local_rank=request.local_rank,
                global_rank=request.global_rank,
                world_size=request.world_size,
                error=error_msg,
            ).model_dump()
            return
        
        with self.load_lock:
            loaded_model = self.loaded_model_name
            model_params = self.model_parameters
            
        # Case 1: No model loaded yet - load it!
        if loaded_model is None:
            logger.info(
                "No model loaded yet. Loading model %s as requested by client...",
                model_name
            )
            try:
                # Load the model asynchronously
                await self._load_model_async(
                    vllm_config,
                    request.config_hash,
                    request.local_rank,
                    request.global_rank,
                    request.world_size,
                )
                
                # Re-read the loaded state
                with self.load_lock:
                    loaded_model = self.loaded_model_name
                    model_params = self.model_parameters
                    
            except Exception as e:
                logger.error("Failed to load model %s: %s", model_name, str(e), exc_info=True)
                yield ModelParametersResponse.from_model_parameters(
                    model_parameters=None,
                    model_name="",
                    device_id=self.device_id,
                    local_rank=request.local_rank,
                    global_rank=request.global_rank,
                    world_size=request.world_size,
                    error=f"Failed to load model: {str(e)}",
                ).model_dump()
                return
        
        # Case 2: Different model loaded
        if loaded_model != model_name:
            error_msg = (
                f"Model mismatch. Server has '{loaded_model}' loaded "
                f"but client requested '{model_name}'. "
                f"Server can only serve one model at a time."
            )
            yield ModelParametersResponse.from_model_parameters(
                model_parameters=None,
                model_name=loaded_model,
                device_id=self.device_id,
                local_rank=request.local_rank,
                global_rank=request.global_rank,
                world_size=request.world_size,
                error=error_msg,
            ).model_dump()
            return
        
        # Case 3: Same model - verify config matches
        config_matches = self.loaded_config_hash == request.config_hash and self.loaded_world_size == request.world_size and self.loaded_local_rank == request.local_rank and self.loaded_global_rank == request.global_rank
        
        if not config_matches:
            error_msg = (
                f"Parallel configuration mismatch for model '{loaded_model}'. "
                f"Server loaded with different configuration than requested."
            )
            yield ModelParametersResponse.from_model_parameters(
                model_parameters=None,
                model_name=loaded_model,
                device_id=self.device_id,
                local_rank=request.local_rank,
                global_rank=request.global_rank,
                world_size=request.world_size,
                error=error_msg,
            ).model_dump()
            return
        
        # Case 4: Verify rank information matches
        rank_matches = (
            self.loaded_local_rank == request.local_rank and
            self.loaded_global_rank == request.global_rank and
            self.loaded_world_size == request.world_size
        )
        
        if not rank_matches:
            error_msg = (
                f"Rank mismatch for model '{loaded_model}'. "
                f"Server loaded with ranks (local={self.loaded_local_rank}, "
                f"global={self.loaded_global_rank}, world_size={self.loaded_world_size}) "
                f"but client has ranks (local={request.local_rank}, "
                f"global={request.global_rank}, world_size={request.world_size}). "
                f"Server can only serve one rank configuration at a time."
            )
            yield ModelParametersResponse.from_model_parameters(
                model_parameters=None,
                model_name=loaded_model,
                device_id=self.device_id,
                local_rank=request.local_rank,
                global_rank=request.global_rank,
                world_size=request.world_size,
                error=error_msg,
            ).model_dump()
            return
        
        # Everything matches - send parameters
        logger.info(
            "Sending model parameters to client:\n"
            "- Model: %s\n"
            "- Parameter count: %d",
            loaded_model,
            len(model_params) if model_params else 0,
        )
        
        # Create response using the helper method that handles pickling
        response = ModelParametersResponse.from_model_parameters(
            model_parameters=model_params,
            model_name=loaded_model,
            device_id=self.device_id,
            local_rank=request.local_rank,
            global_rank=request.global_rank,
            world_size=request.world_size,
            error=None,
        )
        # Yield as dict for Dynamo endpoint
        yield response.model_dump()
    
    @dynamo_endpoint(dict, StatusUpdateMessage)
    async def status_updates(self, request: dict):
        """Stream status updates to clients."""
        try:
            logger.info("Status updates endpoint called with request: %s", request)
            
            # request is expected to be an empty dict for status updates
            # Send initial status
            initial_msg = StatusUpdateMessage(
                model_name=self.loaded_model_name or "",
                status=self.current_status,
                message=self.status_message,
                device_id=self.device_id,
            )
            logger.info("Yielding initial status: %s", initial_msg)
            # Try yielding as dict to see if that works better
            yield initial_msg.model_dump()
            
            # Continue streaming status updates
            last_status = self.current_status
            last_message = self.status_message
            update_count = 1
            
            while True:
                await asyncio.sleep(1.0)  # Check every second
                
                # Only send update if status changed
                if self.current_status != last_status or self.status_message != last_message:
                    last_status = self.current_status
                    last_message = self.status_message
                    update_count += 1
                    
                    update_msg = StatusUpdateMessage(
                        model_name=self.loaded_model_name or "",
                        status=self.current_status,
                        message=self.status_message,
                        device_id=self.device_id,
                    )
                    logger.info("Yielding status update #%d: %s", update_count, update_msg)
                    # Yield as dict
                    yield update_msg.model_dump()
        except Exception as e:
            logger.error("Error in status_updates: %s", str(e), exc_info=True)
            raise
    

@dynamo_worker(static=False)
async def companion_server(runtime: DistributedRuntime):
    """Main entry point for Dynamo companion server."""
    parser = argparse.ArgumentParser(
        description="Dynamo model server for CUDA IPC weight sharing"
    )
    parser.add_argument(
        "--device", type=int, required=True, help="Physical GPU device ID to use"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="companion",
        help="Dynamo namespace for service discovery",
    )
    parser.add_argument(
        "--companion-master-port",
        type=int,
        default=29700,
        help="Master port for the CPU communication group for companion processes",
    )
    args = parser.parse_args()
    
    # Create server instance
    server = DynamoModelServer(device_id=args.device, namespace=args.namespace, companion_master_port=args.companion_master_port)
    
    # Build component name based on physical device ID
    # The client will discover the server by device ID and then send rank information
    component_name = f"model_server_device_{args.device}"

    component = runtime.namespace(args.namespace).component(component_name)
    await component.create_service()
    
    logger.info(
        "Created Dynamo service: namespace=%s, component=%s",
        args.namespace,
        component_name,
    )
    
    # Create and serve endpoints
    params_endpoint = component.endpoint("get_parameters")
    status_endpoint = component.endpoint("status")
    
    # Start serving endpoints
    serve_task = asyncio.gather(
        params_endpoint.serve_endpoint(server.get_model_parameters),
        status_endpoint.serve_endpoint(server.status_updates),
    )
    
    # Wait for endpoints to be served
    await serve_task


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(companion_server())