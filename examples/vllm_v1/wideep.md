# vLLM Distributed Setup Guide

This guide explains how to set up vLLM in a distributed environment across multiple nodes.

## Prerequisites

- Docker installed on all nodes
- Network connectivity between nodes
- Proper network interface configuration

## Build Docker Container

First, build the Docker container using the provided script:

```bash
./container/build.sh --framework VLLM_V1 --target dev
```

## Environment Setup

Set the following environment variables on all nodes:

```bash
export VLLM_ALL2ALL_BACKEND="deepep_low_latency"
export VLLM_USE_DEEP_GEMM=1
export HOST=$(hostname -I)
export GLOO_SOCKET_IFNAME=eth3
```

## Node Configuration

### Node 1 (Primary Node)

Run the following command on the primary node:

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --served_model_name deepseek-ai/DeepSeek-R1 \
    --data_parallel_size 16 \
    --data_parallel_size_local 8 \
    --data_parallel_address <node 1 ip> \
    --data_parallel_rpc_port 13345 \
    --max-model-len 10240 \
    --enable-expert-parallel \
    --trust-remote-code
```

### Node 2 (Secondary Node)

Run the following command on the secondary node:

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --served_model_name deepseek-ai/DeepSeek-R1 \
    --data_parallel_size 16 \
    --data_parallel_size_local 8 \
    --data_parallel_address <node 1 ip> \
    --data_parallel_rpc_port 13345 \
    --max-model-len 10240 \
    --enable-expert-parallel \
    --trust-remote-code \
    --data_parallel_start_rank 8 \
    --headless
```

## Configuration Notes

- Replace `<node 1 ip>` with the actual IP address of Node 1
- The setup uses a total of 16 data parallel workers (`--data_parallel_size 16`)
- Each node runs 8 local workers (`--data_parallel_size_local 8`)
- Node 2 starts with rank 8 (`--data_parallel_start_rank 8`)
- Expert parallel mode is enabled (`--enable-expert-parallel`)
- Maximum model length is set to 10240 tokens
- The RPC port is set to 13345