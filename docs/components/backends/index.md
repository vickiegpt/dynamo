# Backend Components

Dynamo supports multiple inference backends, each with unique strengths.

## Available Backends

- [vLLM](vllm/README.md) - High-throughput inference with PagedAttention
- [SGLang](sglang/README.md) - Efficient serving with advanced parallelism
- [TensorRT-LLM](trtllm/README.md) - NVIDIA-optimized inference

## Backend Selection Guide

### vLLM
**Best for:**
- High throughput requirements
- OpenAI compatibility needs
- Extensive model support

**Key Features:**
- PagedAttention for efficient memory usage
- Continuous batching
- Full Dynamo feature support

### SGLang
**Best for:**
- Large-scale distributed deployments
- Advanced parallelism requirements
- RadixAttention optimization

**Key Features:**
- WideEP and expert parallelism
- Efficient multi-node scaling
- ZMQ-based communication

### TensorRT-LLM
**Best for:**
- Maximum inference performance
- NVIDIA GPU optimization
- Multimodal workloads

**Key Features:**
- TensorRT acceleration
- INT8/FP8 quantization
- Advanced kernel fusion

## Custom Backends

- [Write Your Own Python Worker](../../guides/backend.md) - Create custom backend implementations