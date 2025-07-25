# TensorRT-LLM Disaggregated Workflow Diagram

This document provides comprehensive diagrams demonstrating the disaggregated workflow for TensorRT-LLM backend in Dynamo.

## Overview

The disaggregated workflow separates LLM inference into two specialized workers:
- **Prefill Worker**: Handles the initial processing of input tokens (prefill phase)
- **Decode Worker**: Handles the generation of subsequent tokens (decode phase)

This separation allows for optimized resource allocation and improved throughput.

## 1. Basic Disaggregated Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        C[HTTP Client]
    end
    
    subgraph "Frontend Layer"
        F[dynamo.frontend<br/>HTTP Port: 8000]
        P[Processor]
    end
    
    subgraph "Worker Layer"
        subgraph "Prefill Worker"
            PW[Prefill Worker<br/>CUDA_VISIBLE_DEVICES: 0<br/>Mode: prefill]
            PW_CONFIG[Engine Config:<br/>• max_batch_size: 16<br/>• max_num_tokens: 8192<br/>• enable_chunked_prefill: true<br/>• disable_overlap_scheduler: true<br/>• use_cuda_graph: false]
        end
        
        subgraph "Decode Worker"
            DW[Decode Worker<br/>CUDA_VISIBLE_DEVICES: 1<br/>Mode: decode]
            DW_CONFIG[Engine Config:<br/>• max_batch_size: 16<br/>• max_num_tokens: 8192<br/>• enable_chunked_prefill: true<br/>• disable_overlap_scheduler: false<br/>• use_cuda_graph: true]
        end
    end
    
    subgraph "Infrastructure"
        E[etcd]
        N[NATS]
    end
    
    C -->|HTTP Request| F
    F --> P
    P -->|Route Request| PW
    P -->|Route Request| DW
    PW -->|KV Cache Transfer| DW
    DW -->|Response| P
    P --> F
    F -->|HTTP Response| C
    
    E -.->|Service Discovery| F
    N -.->|Event Bus| PW
    N -.->|Event Bus| DW
```

## 2. Disaggregation Strategies

### 2.1 Decode-First Strategy (Default)

```mermaid
sequenceDiagram
    participant Client
    participant Frontend
    participant Processor
    participant DecodeWorker
    participant PrefillWorker
    participant NATS
    
    Client->>Frontend: HTTP Request
    Frontend->>Processor: Forward Request
    Processor->>DecodeWorker: Route to Decode Worker (Round-robin)
    
    alt Prefill Required
        DecodeWorker->>PrefillWorker: Forward for Prefill
        PrefillWorker->>PrefillWorker: Process Prefill Phase
        PrefillWorker->>DecodeWorker: Return with KV Cache
        Note over PrefillWorker,DecodeWorker: KV Cache Transfer via UCX/NIXL
    end
    
    DecodeWorker->>DecodeWorker: Process Decode Phase
    DecodeWorker->>Processor: Return Results
    Processor->>Frontend: Forward Response
    Frontend->>Client: HTTP Response
    
    Note over DecodeWorker,NATS: Publish KV Events & Metrics
```

### 2.2 Prefill-First Strategy (with KV Routing)

```mermaid
sequenceDiagram
    participant Client
    participant Frontend
    participant Processor
    participant PrefillWorker
    participant DecodeWorker
    participant KVRouter
    participant NATS
    
    Client->>Frontend: HTTP Request
    Frontend->>Processor: Forward Request
    Processor->>PrefillWorker: Route Directly to Prefill Worker
    
    PrefillWorker->>PrefillWorker: Process Prefill Phase
    PrefillWorker->>KVRouter: Publish KV Cache Events
    Note over PrefillWorker,KVRouter: KV Cache Transfer via UCX/NIXL
    
    PrefillWorker->>DecodeWorker: Forward for Decode
    DecodeWorker->>DecodeWorker: Process Decode Phase
    DecodeWorker->>Processor: Return Results
    Processor->>Frontend: Forward Response
    Frontend->>Client: HTTP Response
    
    Note over PrefillWorker,NATS: Publish Events & Metrics
```

## 3. KV Cache Transfer Mechanisms

### 3.1 UCX (Default Method)

```mermaid
graph LR
    subgraph "Prefill Worker"
        PW[Prefill Worker]
        PW_MEM[GPU Memory<br/>KV Cache]
    end
    
    subgraph "Decode Worker"
        DW[Decode Worker]
        DW_MEM[GPU Memory<br/>KV Cache]
    end
    
    subgraph "Network Layer"
        UCX[UCX<br/>Unified Communication X]
    end
    
    PW_MEM -->|High-performance<br/>GPU-to-GPU Transfer| UCX
    UCX -->|Optimized for<br/>GPU Communication| DW_MEM
    
    style UCX fill:#e1f5fe
    style PW_MEM fill:#f3e5f5
    style DW_MEM fill:#f3e5f5
```

### 3.2 NIXL (Experimental Method)

```mermaid
graph LR
    subgraph "Prefill Worker"
        PW[Prefill Worker]
        PW_MEM[GPU Memory<br/>KV Cache]
    end
    
    subgraph "Decode Worker"
        DW[Decode Worker]
        DW_MEM[GPU Memory<br/>KV Cache]
    end
    
    subgraph "Network Layer"
        NIXL[NIXL<br/>NVIDIA Inference Xfer Library]
    end
    
    PW_MEM -->|Experimental<br/>High-performance Transfer| NIXL
    NIXL -->|Distributed GPU<br/>Communication| DW_MEM
    
    style NIXL fill:#fff3e0
    style PW_MEM fill:#f3e5f5
    style DW_MEM fill:#f3e5f5
```

## 4. Multi-Node Disaggregated Deployment

```mermaid
graph TB
    subgraph "Node 1 (Head Node)"
        F[dynamo.frontend<br/>HTTP Port: 8000]
        E[etcd]
        N[NATS]
    end
    
    subgraph "Nodes 2-5 (Prefill Cluster)"
        subgraph "Node 2"
            PW1[Prefill Worker<br/>GPU 0-3]
        end
        subgraph "Node 3"
            PW2[Prefill Worker<br/>GPU 0-3]
        end
        subgraph "Node 4"
            PW3[Prefill Worker<br/>GPU 0-3]
        end
        subgraph "Node 5"
            PW4[Prefill Worker<br/>GPU 0-3]
        end
    end
    
    subgraph "Nodes 6-9 (Decode Cluster)"
        subgraph "Node 6"
            DW1[Decode Worker<br/>GPU 0-3]
        end
        subgraph "Node 7"
            DW2[Decode Worker<br/>GPU 0-3]
        end
        subgraph "Node 8"
            DW3[Decode Worker<br/>GPU 0-3]
        end
        subgraph "Node 9"
            DW4[Decode Worker<br/>GPU 0-3]
        end
    end
    
    F -->|MPI Tasks| PW1
    F -->|MPI Tasks| PW2
    F -->|MPI Tasks| PW3
    F -->|MPI Tasks| PW4
    
    F -->|MPI Tasks| DW1
    F -->|MPI Tasks| DW2
    F -->|MPI Tasks| DW3
    F -->|MPI Tasks| DW4
    
    PW1 -.->|KV Cache Transfer| DW1
    PW2 -.->|KV Cache Transfer| DW2
    PW3 -.->|KV Cache Transfer| DW3
    PW4 -.->|KV Cache Transfer| DW4
    
    E -.->|Service Discovery| F
    N -.->|Event Bus| PW1
    N -.->|Event Bus| PW2
    N -.->|Event Bus| PW3
    N -.->|Event Bus| PW4
    N -.->|Event Bus| DW1
    N -.->|Event Bus| DW2
    N -.->|Event Bus| DW3
    N -.->|Event Bus| DW4
    
    style F fill:#e8f5e8
    style E fill:#fff3e0
    style N fill:#fff3e0
    style PW1 fill:#e3f2fd
    style PW2 fill:#e3f2fd
    style PW3 fill:#e3f2fd
    style PW4 fill:#e3f2fd
    style DW1 fill:#f3e5f5
    style DW2 fill:#f3e5f5
    style DW3 fill:#f3e5f5
    style DW4 fill:#f3e5f5
```

## 5. Request Flow Comparison

### 5.1 Aggregated vs Disaggregated

```mermaid
graph LR
    subgraph "Aggregated Workflow"
        A1[Single Worker<br/>Handles Both<br/>Prefill + Decode]
    end
    
    subgraph "Disaggregated Workflow"
        D1[Prefill Worker<br/>Specialized for<br/>Initial Processing]
        D2[Decode Worker<br/>Specialized for<br/>Token Generation]
    end
    
    A1 -->|Sequential Processing| A1
    D1 -->|Parallel Processing| D2
    
    style A1 fill:#ffebee
    style D1 fill:#e8f5e8
    style D2 fill:#e3f2fd
```

## 6. Configuration Examples

### 6.1 Environment Variables

```bash
# Basic Disaggregated Setup
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
export SERVED_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
export DISAGGREGATION_STRATEGY="decode_first"  # or "prefill_first"
export PREFILL_ENGINE_ARGS="engine_configs/prefill.yaml"
export DECODE_ENGINE_ARGS="engine_configs/decode.yaml"
export PREFILL_CUDA_VISIBLE_DEVICES="0"
export DECODE_CUDA_VISIBLE_DEVICES="1"

# Multi-Node Setup
export NUM_PREFILL_NODES=4
export NUM_DECODE_NODES=4
export NUM_GPUS_PER_NODE=4

# KV Cache Transfer Method
export TRTLLM_USE_NIXL_KVCACHE=1  # For NIXL (experimental)
# export TRTLLM_USE_UCX_KVCACHE=1  # For UCX (default)
```

### 6.2 Launch Commands

```bash
# Basic Disaggregated
./launch/disagg.sh

# Disaggregated with KV Routing
./launch/disagg_router.sh

# Multi-Node Disaggregated
./multinode/srun_disaggregated.sh
```

## 7. Performance Considerations

### 7.1 Resource Allocation

- **Prefill Worker**: Optimized for batch processing with larger memory allocation
- **Decode Worker**: Optimized for single-token generation with CUDA graphs enabled
- **KV Cache Transfer**: High-bandwidth network connection between workers

### 7.2 Scaling Strategies

- **Horizontal Scaling**: Add more prefill/decode workers
- **Vertical Scaling**: Increase GPU memory and compute resources per worker
- **Hybrid Scaling**: Combine both approaches for optimal performance

### 7.3 Monitoring

- **Throughput**: Requests per second
- **Latency**: End-to-end response time
- **KV Cache Hit Rate**: Efficiency of cache reuse
- **GPU Utilization**: Resource usage across workers 