# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-based companion client for CUDA IPC weight sharing."""

import asyncio
import copy
import json
import logging
import os
from typing import Dict, Optional

import torch
import uvloop
from vllm.config import VllmConfig

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .companion_messages import (
    GetModelParametersRequest,
    ModelParametersResponse,
    StatusUpdateMessage,
)
from .gpu_utils import get_physical_device_index

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class DynamoModelClient:
    """Client for connecting to Dynamo model server and retrieving model parameters."""
    
    def __init__(
        self,
        runtime: DistributedRuntime,
        vllm_config: VllmConfig,
        namespace: str = "companion",
    ):
        """Initialize client to connect to a model server.
        
        Args:
            runtime: Dynamo distributed runtime instance
            vllm_config: VllmConfig with model and parallel configuration
            namespace: Dynamo namespace for service discovery
        """
        self.runtime = runtime
        self.vllm_config = vllm_config
        self.namespace = namespace
        
        # Get device information
        self.logical_device = torch.cuda.current_device()
        self.physical_device = get_physical_device_index(self.logical_device)
        
        # Build a unique component name based on model + rank identifiers.
        # NOTE: we avoid colon or slash since those are not valid in NATS subjects.
        self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.server_component = (
            f"model_server_g{self.global_rank}_l{self.local_rank}"
        )
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        
        logger.info(
            "Dynamo model client initialized:\n"
            "- Model: %s\n"
            "- Parallel config: TP=%d, PP=%d, DP=%d\n"
            "- Local Rank: %d, Global Rank: %d, World size: %d\n"
            "- Namespace: %s\n"
            "- Target component: %s\n"
            "- Logical device: %d\n"
            "- Physical device: %d\n"
            "- CUDA_VISIBLE_DEVICES: %s",
            vllm_config.model_config.model,
            vllm_config.parallel_config.tensor_parallel_size,
            vllm_config.parallel_config.pipeline_parallel_size,
            vllm_config.parallel_config.data_parallel_size,
            self.local_rank,
            self.global_rank,
            self.world_size,
            namespace,
            self.server_component,
            self.logical_device,
            self.physical_device,
            os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        )
    
    async def wait_for_model_ready(
        self, initial_timeout: float = 15.0, loading_timeout: float = 300.0
    ) -> tuple[bool, dict]:
        """Wait for the model to be ready on the server.
        
        Args:
            initial_timeout: Max time to wait for initial server response
            loading_timeout: Max time to wait for model loading to complete
            
        Returns:
            Tuple of (success, server_info)
        """
        # Get status endpoint
        status_endpoint = (
            self.runtime.namespace(self.namespace)
            .component(self.server_component)
            .endpoint("status")
        )
        
        # Create client and wait for server
        status_client = await status_endpoint.client()
        
        # Debug: log client type and endpoint
        logger.info("Status endpoint: %s", status_endpoint)
        logger.info("Status client type: %s", type(status_client))
        logger.info("Status client methods: %s", dir(status_client))
        
        # Check if this is a generic endpoint (not "generate")
        # For non-"generate" endpoints, methods are accessed via routing functions
        
        try:
            # Wait for at least one server instance
            await asyncio.wait_for(
                status_client.wait_for_instances(), timeout=initial_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                "Server did not respond within %.1f seconds. Is the server running?",
                initial_timeout,
            )
            return False, {}
        
        logger.info("Server is alive! Subscribing to status updates...")
        
        # Subscribe to status updates
        # For endpoints not named "generate", we need to use routing methods
        status_stream = await status_client.round_robin(json.dumps({}))
        
        # First, try to get model parameters which will trigger loading
        asyncio.create_task(self._trigger_model_loading())
        
        # Monitor status updates
        start_time = asyncio.get_event_loop().time()
        
        try:
            update_count = 0
            async for update in status_stream:
                update_count += 1
                logger.debug("Received status update #%d", update_count)
                status_data = update.data()
                
                # Handle dict or StatusUpdateMessage
                if isinstance(status_data, dict):
                    status_msg = StatusUpdateMessage(**status_data)
                else:
                    status_msg = status_data
                    
                logger.info(
                    "Server status: %s - %s", status_msg.status, status_msg.message
                )
                
                # Check if model matches
                if status_msg.model_name and status_msg.model_name != self.vllm_config.model_config.model:
                    logger.error(
                        "Model name mismatch! Expected '%s' but server has '%s'",
                        self.vllm_config.model_config.model,
                        status_msg.model_name,
                    )
                    return False, {}
                
                # Check if model is loaded
                if (
                    status_msg.status == "loaded"
                    and status_msg.model_name == self.vllm_config.model_config.model
                ):
                    logger.info("Model %s is ready!", self.vllm_config.model_config.model)
                    return True, {
                        "model_name": status_msg.model_name,
                        "device_id": status_msg.device_id,
                    }
                
                # Check for errors
                if status_msg.status == "error":
                    logger.error("Model loading failed: %s", status_msg.message)
                    return False, {}
                
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > loading_timeout:
                    logger.error(
                        "Model loading timed out after %.1f seconds", loading_timeout
                    )
                    return False, {}
            
            # If we exit the loop without finding a loaded model, something's wrong
            logger.error("Status stream ended after %d updates without model loaded", update_count)
            return False, {}
        
        except Exception as e:
            logger.error("Error monitoring status updates: %s", str(e))
            return False, {}
    
    async def _trigger_model_loading(self):
        """Try to get model parameters to trigger loading if needed."""
        try:
            # Try to get parameters which may trigger loading
            await self.get_model_parameters()
        except Exception as e:
            # Expected if model is loading or other error
            logger.debug("Model loading check: %s", str(e))
    
    async def get_model_parameters(self) -> Dict[str, Dict]:
        """Get model parameters from the server for IPC sharing.
        
        Returns:
            Dictionary of parameter names to CUDA IPC rebuild info
            
        Raises:
            RuntimeError: If there's an error getting parameters
        """
        # Get parameters endpoint
        params_endpoint = (
            self.runtime.namespace(self.namespace)
            .component(self.server_component)
            .endpoint("get_parameters")
        )
        
        # Create client
        params_client = await params_endpoint.client()
        await params_client.wait_for_instances()
        
        # Prepare request with essential config data only
        request = GetModelParametersRequest(
            model_name=self.vllm_config.model_config.model,
            tensor_parallel_size=self.vllm_config.parallel_config.tensor_parallel_size,
            pipeline_parallel_size=self.vllm_config.parallel_config.pipeline_parallel_size,
            data_parallel_size=self.vllm_config.parallel_config.data_parallel_size,
            device_id=self.physical_device,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )
        
        # Send request and get response
        # For non-"generate" endpoints, we need to use routing methods
        # Convert Pydantic model to JSON for serialization
        response_stream = await params_client.round_robin(request.model_dump_json())
        
        # Get the response (should be a single response, not a stream)
        response = None
        async for resp in response_stream:
            response = resp.data()
            break  # Only expect one response
        
        if response is None:
            raise RuntimeError("No response received from server")
        
        # Handle dict or object response
        if isinstance(response, dict):
            error = response.get("error")
            model_parameters = response.get("model_parameters")
        else:
            error = response.error
            model_parameters = response.model_parameters
        
        # Check for errors
        if error:
            raise RuntimeError(f"Error getting model parameters: {error}")
        
        if model_parameters is None:
            raise RuntimeError("No model parameters received")
        
        logger.info(
            "Received %d model parameters from server",
            len(model_parameters),
        )
        
        return model_parameters
    
    def reconstruct_parameter(self, rebuild_info: Dict) -> torch.Tensor:
        """Reconstruct a model parameter tensor from rebuild info using CUDA IPC.
        
        Args:
            rebuild_info: Dictionary containing CUDA IPC rebuild information
            
        Returns:
            Reconstructed tensor on the current device
        """
        from torch.multiprocessing.reductions import rebuild_cuda_tensor
        import base64
        
        # Deserialize the rebuild info from JSON format
        if isinstance(rebuild_info, dict) and "tensor_type" in rebuild_info:
            # Convert string types back to actual types
            tensor_type = torch.Tensor  # Default to torch.Tensor
            storage_type = getattr(torch, rebuild_info["storage_type"].split("'")[1].split('.')[-1])
            tensor_dtype = getattr(torch, rebuild_info["tensor_dtype"].split('.')[-1])
            
            # Decode base64 encoded bytes
            ipc_handle = base64.b64decode(rebuild_info["ipc_handle"])
            ref_counter_handle = base64.b64decode(rebuild_info["ref_counter_handle"])
            event_handle = base64.b64decode(rebuild_info["event_handle"])
            
            # Convert lists back to tuples/torch.Size
            tensor_size = torch.Size(rebuild_info["tensor_size"])
            tensor_stride = tuple(rebuild_info["tensor_stride"])
            
            # Adjust device to logical device
            server_device = rebuild_info["device"]
            modified_device = self.logical_device
            
            logger.debug(
                "Reconstructing tensor: server_device=%d → client_logical_device=%d",
                server_device,
                modified_device,
            )
            
            # Create rebuild args tuple
            rebuild_args = (
                tensor_type,
                tensor_size,
                tensor_stride,
                rebuild_info["tensor_offset"],
                storage_type,
                tensor_dtype,
                modified_device,  # Use logical device instead of server device
                ipc_handle,
                rebuild_info["storage_size_bytes"],
                rebuild_info["storage_offset_bytes"],
                rebuild_info["tensor_requires_grad"],
                ref_counter_handle,
                rebuild_info["ref_counter_offset"],
                event_handle,
                rebuild_info["event_sync_required"],
            )
            
            # Reconstruct the tensor
            tensor = rebuild_cuda_tensor(*rebuild_args)
            
            return tensor
        else:
            raise ValueError("Invalid rebuild info format")


async def create_model_client(
    runtime: DistributedRuntime,
    vllm_config: VllmConfig,
    namespace: str = "companion",
) -> DynamoModelClient:
    """Factory function to create a DynamoModelClient.
    
    Args:
        runtime: Dynamo distributed runtime
        vllm_config: VllmConfig with model configuration
        namespace: Dynamo namespace for service discovery
        
    Returns:
        Initialized DynamoModelClient instance
    """
    client = DynamoModelClient(runtime, vllm_config, namespace)
    return client


@dynamo_worker(static=False)
async def example_client(runtime: DistributedRuntime):
    """Example client worker for testing."""
    from vllm.config import ParallelConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    
    # Example configuration
    engine_args = AsyncEngineArgs(model="meta-llama/Llama-2-7b-hf")
    vllm_config = engine_args.create_engine_config()
    
    # Create client
    client = await create_model_client(runtime, vllm_config)
    
    # Wait for model to be ready
    success, info = await client.wait_for_model_ready()
    if not success:
        logger.error("Failed to connect to model server")
        return
    
    logger.info("Model ready: %s", info)
    
    # Get model parameters
    try:
        parameters = await client.get_model_parameters()
        logger.info("Successfully retrieved %d parameters", len(parameters))
        
        # Example: reconstruct first parameter
        if parameters:
            param_name = list(parameters.keys())[0]
            tensor = client.reconstruct_parameter(parameters[param_name])
            logger.info(
                "Reconstructed parameter '%s' with shape %s",
                param_name,
                tensor.shape,
            )
    except Exception as e:
        logger.error("Failed to get model parameters: %s", str(e))


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(example_client())