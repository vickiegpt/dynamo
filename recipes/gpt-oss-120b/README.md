# GPT-OSS-120B Recipe Guide

This guide will help you run the GPT-OSS-120B language model using Dynamo's optimized setup.

## Prerequisites

follow the instructions in recipe [README.md](../README.md) to create a namespace and kubernetes secret for huggingface token.

## Quick Start

To run the model, simply execute this command in your terminal:

```bash
cd recipe
./run.sh --model gpt-oss-120b --framework trtllm agg
```

## (Alternative) Step by Step Guide

### 1. Download the Model

```bash
cd recipes/gpt-oss-120b
kubectl apply -n $NAMESPACE -f ./model-cache
```

### 2. Deploy and Benchmark the Model

```bash
cd recipes/gpt-oss-120b
kubectl apply -n $NAMESPACE -f ./trtllm/agg
```

### Container Image
This recipe was tested with dynamo trtllm runtime container for ARM64 processors.

**Pre-built Image:**
```
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:7fdf50fec2cae9112224f5cea26cef3dde78506f-35606896-trtllm-arm64
```

### Building Your Own Image (Optional)

If you need to build the container image yourself (for example, if you're using different hardware or want to customize the setup):

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ai-dynamo/dynamo.git
   cd dynamo
   ```

2. **Switch to the specific version:**
   ```bash
   git checkout 7fdf50fec2cae9112224f5cea26cef3dde78506f
   ```

3. **Build the container:**
   ```bash
   ./container/build.sh --framework TRTLLM --target runtime
   ```

## Notes
1. The benchmark container image uses a specific commit of aiperf to ensure reproducible results and compatibility with the benchmarking setup.

2. storage class is not specified in the recipe, you need to specify it in the `deploy.yaml` file.