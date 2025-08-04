<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# vLLM V1 Modular Components Demo

This example demonstrates vLLM V1's modular architecture using dynamo transport for efficient inter-process communication. Each component runs as an independent rank 0 worker microservice, enabling independent scaling and deployment in containerized environments. This approach provides production-ready modularity with high-performance IPC.

## vLLM V1 Microservices Architecture

vLLM V1 represents a comprehensive re-architecture with these core modular components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Router        │    │   Scheduler     │
│ (OpenAI API)    │    │ (Load Balancer) │    │ (Token Alloc)   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └──dynamo transport──────┼──dynamo transport─────┘
                                 │
         ┌─────────────────┐    │    ┌─────────────────┐
         │  KV Cache Mgr   │    │    │     Worker      │
         │ (Memory Mgmt)   │◄───┘───►│ (Model Exec)    │
         │                 │         │                 │
         └─────────────────┘         └─────────────────┘
                                             │
                                    ┌─────────────────┐
                                    │    Sampler      │
                                    │ (Token Logic)   │
                                    │                 │
                                    └─────────────────┘
```

## Components

### 1. Frontend (`frontend.py`)
- **Purpose**: OpenAI-compatible API server
- **Features**: Handles client requests, forwards to router via dynamo transport
- **Port**: 8001
- **Communication**: Dynamo transport to router

### 2. Router (`router.py`)
- **Purpose**: Traffic management and load balancing
- **Features**: Routes requests, manages worker pool, provides traffic shaping
- **Port**: 8004
- **Communication**: Dynamo transport to scheduler and workers

### 3. Scheduler (`scheduler.py`)
- **Purpose**: Request scheduling and token allocation
- **Features**: Uses `vllm.core.scheduler.Scheduler`, coordinates inference pipeline
- **Port**: 8002
- **Communication**: Dynamo transport to KV cache and workers

### 4. Worker (`worker.py`)
- **Purpose**: Model execution and inference
- **Features**: Uses `vllm.LLM` engine, handles actual inference
- **Port**: 8003+ (multiple workers supported)
- **Communication**: Dynamo transport to scheduler and sampler

### 5. KV Cache Manager (`kv_cache_manager.py`)
- **Purpose**: Key-value cache and memory management
- **Features**: Uses `vllm.core.block_manager_v1.BlockSpaceManagerV1`
- **Port**: 8005
- **Communication**: Dynamo transport to scheduler

### 6. Sampler (`sampler.py`)
- **Purpose**: Token sampling and generation logic
- **Features**: Uses `vllm.model_executor.layers.sampler.Sampler`
- **Port**: 8006
- **Communication**: Dynamo transport from workers

## Quick Start

### Prerequisites

```bash
# Install vLLM V1 and dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
# Make the demo script executable
chmod +x run_demo.sh

# Start all components
./run_demo.sh
```

### Test the API

```bash
# Test completion
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Check component health
curl http://localhost:8001/health  # Frontend
curl http://localhost:8004/health  # Router
curl http://localhost:8002/health  # Scheduler
curl http://localhost:8003/health  # Worker
curl http://localhost:8005/health  # KV Cache
curl http://localhost:8006/health  # Sampler
```

## Features Demonstrated

### Core vLLM V1 Features
- **Unified Token Allocation**: Scheduler handles both prompt and output tokens uniformly
- **Chunked Prefill**: Efficient processing of long prompts in chunks
- **Prefix Caching**: KV cache manager optimizes repeated prompt segments
- **Block-based Memory Management**: Efficient memory allocation and deallocation

### Microservices Architecture
- **Independent Scaling**: Each component scales independently
- **Container-Ready**: Each component can run in separate containers
- **High-Performance IPC**: Dynamo transport for efficient communication
- **Production-Ready**: Real vLLM components, not simplified demos

### Request Flow
1. **Frontend** receives OpenAI API request
2. **Frontend** forwards to **Router** via dynamo transport
3. **Router** coordinates and forwards to **Scheduler**
4. **Scheduler** tokenizes, requests KV cache allocation
5. **KV Cache Manager** allocates memory blocks
6. **Scheduler** forwards inference request to **Worker**
7. **Worker** notifies **Sampler** about sampling parameters
8. **Worker** runs vLLM inference using real model
9. Response flows back through the chain
10. **Frontend** returns OpenAI-compatible response

## Configuration

Edit `components/config.py` to adjust:

```python
class VLLMConfig:
    # Model configuration
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_model_len: Optional[int] = 8192

    # Performance settings
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 2048

    # Component ports
    frontend_port: int = 8001
    scheduler_port: int = 8002
    worker_port: int = 8003
    router_port: int = 8004
    kv_cache_port: int = 8005
    sampler_port: int = 8006

    # Workers
    num_workers: int = 1
```

## Development

### Adding New Components

1. Create new component file in `components/`
2. Inherit from `MicroserviceBase` in `dynamo_transport.py`
3. Implement request handlers using `MessageType` enum
4. Register component in `launch.py`

### Scaling Components

```bash
# Scale workers
python components/worker.py --worker-id 1 --port 8007 &
python components/worker.py --worker-id 2 --port 8008 &

# Scale other components
python components/scheduler.py --port 8009 &
```

### Container Deployment

Each component can run in separate containers:

```dockerfile
# Example Dockerfile for worker component
FROM python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY components/ ./components/
CMD ["python", "components/worker.py"]
```

## Architecture Benefits

### Modularity
- **Separation of Concerns**: Each component has a single responsibility
- **Independent Development**: Teams can work on components independently
- **Easier Testing**: Components can be tested in isolation

### Scalability
- **Horizontal Scaling**: Scale components based on workload
- **Resource Optimization**: Allocate resources per component needs
- **Load Distribution**: Router manages traffic distribution

### Production Readiness
- **Real vLLM Components**: Uses actual vLLM V1 scheduler, block manager, etc.
- **Efficient Communication**: Dynamo transport for high-performance IPC
- **Container Native**: Designed for containerized deployments
- **Observability**: Rich logging and health checks

### Performance
- **Zero-Copy Communication**: Dynamo transport minimizes data copying
- **Async Processing**: Non-blocking communication between components
- **Memory Efficiency**: Dedicated KV cache management
- **GPU Utilization**: Optimized worker-to-GPU mapping

## Troubleshooting

### Common Issues

1. **Port conflicts**: Check `lsof -i :PORT` and kill conflicting processes
2. **Model loading errors**: Ensure sufficient GPU memory
3. **Component communication**: Check component logs for transport errors

### Logs

Each component provides detailed logging:
- Request flow tracking
- Performance metrics
- Error diagnostics
- Health status

### Monitoring

- Health endpoints on each component
- Component-specific statistics
- Request tracing across the pipeline

## Next Steps

This demo provides a foundation for:
- **Production Deployment**: Deploy components in Kubernetes/Docker
- **Advanced Features**: Add authentication, rate limiting, circuit breakers
- **Monitoring**: Integrate with Prometheus/Grafana
- **Auto-scaling**: Implement HPA based on component metrics
