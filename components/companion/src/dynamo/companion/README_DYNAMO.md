# Dynamo Companion Server/Client

This directory contains the Dynamo runtime-based reimplementation of the companion server and client for CUDA IPC weight sharing.

## Overview

The Dynamo companion system allows multiple processes to share GPU model weights using CUDA IPC (Inter-Process Communication), eliminating the need to load the same model multiple times on the same GPU.

### Key Advantages over ZMQ Implementation

1. **Automatic Service Discovery**: No need to hardcode ports or addresses
2. **Built-in Load Balancing**: Support for multiple server instances
3. **Health Monitoring**: Automatic health checks and failover
4. **Streaming Support**: Native support for both request-response and streaming patterns
5. **Better Integration**: Seamless integration with other Dynamo components

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│  Companion Client   │         │  Companion Server   │
│                     │         │                     │
│  - DynamoModelClient│ Dynamo  │  - DynamoModelServer│
│  - Endpoints:       │<------->│  - Endpoints:       │
│    * get_parameters │ Runtime │    * get_parameters │
│    * status         │         │    * status         │
└─────────────────────┘         └─────────────────────┘
         │                                 │
         └────────── CUDA IPC ─────────────┘
                  (Same GPU)
```

## Components

### 1. `dynamo_companion_server.py`
- Loads models on-demand on a specific GPU
- Serves model parameters via CUDA IPC
- Streams status updates to clients
- Registered as a Dynamo component with endpoints

### 2. `dynamo_companion_client.py`
- Connects to companion server via Dynamo runtime
- Retrieves model parameters using CUDA IPC
- Monitors server status
- Reconstructs tensors on the client side

### 3. `companion_messages.py`
- Typed message definitions for client-server communication
- Includes request/response types for model parameters and status updates

## Usage

### Starting the Server

```bash
# Start server on GPU 0
python -m dynamo.companion.dynamo_companion_server --device 0 --namespace companion

# The server will:
# 1. Register as component "model_server_gpu_0" in namespace "companion"
# 2. Create endpoints: "get_parameters" and "status"
# 3. Wait for model loading requests
```

### Using the Client

```python
from dynamo.runtime import DistributedRuntime
from dynamo.companion import create_model_client
from vllm.engine.arg_utils import AsyncEngineArgs

# Create VllmConfig
engine_args = AsyncEngineArgs(model="facebook/opt-125m")
vllm_config = engine_args.create_engine_config()

# Create client (inside a dynamo_worker)
@dynamo_worker(static=False)
async def my_worker(runtime: DistributedRuntime):
    client = await create_model_client(runtime, vllm_config, namespace="companion")
    
    # Wait for model to be ready
    success, info = await client.wait_for_model_ready()
    if success:
        # Get model parameters
        parameters = await client.get_model_parameters()
        
        # Reconstruct tensors
        for name, rebuild_info in parameters.items():
            tensor = client.reconstruct_parameter(rebuild_info)
            print(f"Parameter {name}: shape={tensor.shape}")
```

### Running the Test

```bash
# Run test with automatic server startup
python -m dynamo.companion.test_companion_communication \
    --model facebook/opt-125m \
    --start-server

# Or start server manually first, then run test
python -m dynamo.companion.test_companion_communication \
    --model facebook/opt-125m
```

## How It Works

1. **Service Discovery**: 
   - Server registers in etcd under `/services/companion/model_server_gpu_X/`
   - Client watches etcd for available servers on the same GPU

2. **Communication Flow**:
   - Client sends `GetModelParametersRequest` with model config
   - Server loads model if not already loaded
   - Server responds with `ModelParametersResponse` containing CUDA IPC handles
   - Client reconstructs tensors using CUDA IPC

3. **Status Monitoring**:
   - Server streams status updates (loading, loaded, error)
   - Client can subscribe to status endpoint for real-time updates

## Requirements

- CUDA-capable GPU
- PyTorch with CUDA support
- Dynamo runtime with etcd and NATS
- vLLM for model loading

## Important Notes

1. **Same GPU Requirement**: Client and server must be on the same physical GPU for CUDA IPC
2. **No CUDA_VISIBLE_DEVICES for Server**: The server must run without CUDA_VISIBLE_DEVICES to ensure correct physical GPU mapping
3. **One Model Per Server**: Each server can only serve one model at a time
4. **Parallel Config Must Match**: Client and server must use the same parallel configuration

## Comparison with ZMQ Implementation

| Feature | ZMQ Implementation | Dynamo Implementation |
|---------|-------------------|----------------------|
| Service Discovery | Manual (hardcoded ports) | Automatic (etcd) |
| Communication | ZMQ REQ/REP + PUB/SUB | Dynamo endpoints |
| Load Balancing | Manual | Built-in |
| Health Checks | Manual | Automatic |
| Streaming | ZMQ PUB/SUB | Native streaming |
| Integration | Standalone | Dynamo ecosystem |