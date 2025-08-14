<!-- # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Dynamo Benchmarking Guide

This guide provides instructions for benchmarking NVIDIA Dynamo deployments using industry-standard tools and methodologies. Currently, this guide is only for vLLM and Kubernetes.

## Overview

Dynamo benchmarking enables you to:

- Benchmark aggregated vs. disaggregated vs. vanilla vLLM deployments
- Identify the best configuration for your workload and SLA requirements
- Ensure your deployment meets latency and throughput requirements
- Understand performance characteristics across different concurrency levels

## Benchmarking Types

### 1. Pre-Deployment Profiling

For optimizing Dynamo configurations before deployment, see [Pre-Deployment Profiling](pre_deployment_profiling.md). This helps you:

- Determine optimal tensor parallelism settings
- Choose between aggregated and disaggregated topologies
- Configure planner parameters for your SLA requirements

### 2. Deployment Benchmarking

Compare different deployment types with real workloads using GenAI-Perf. This guide covers:

- **Dynamo Aggregated**: Single-stage inference with shared prefill/decode
- **Dynamo Disaggregated**: Separate prefill and decode workers
- **Vanilla vLLM**: Standard vLLM deployment for baseline comparison

### 3. Custom Benchmarking

Bring your own benchmarking scripts and tools (covered in [Custom Benchmarking](#custom-benchmarking)).

## Prerequisites

### Software Requirements

1. **kubectl** - Kubernetes command-line tool
2. **GenAI-Perf** - NVIDIA's LLM benchmarking tool
3. **Python 3.8+** - For result analysis and plotting
4. **Docker** (for containerized benchmarking)

All of these are included within Dynamo-built containers.

## Quick Start

Run benchmarks on Dynamo deployments.

```bash
# Set your configuration
export NAMESPACE=hzhou-dynamo
export MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
export INPUT_SEQUENCE_LENGTH=200
export INPUT_SEQUENCE_STD=10
export OUTPUT_SEQUENCE_LENGTH=200
export AGG_CONFIG=components/backends/vllm/deploy/agg.yaml
export DISAGG_CONFIG=components/backends/vllm/deploy/disagg.yaml
export OUTPUT_DIR=benchmarks/results

python3 -u -m benchmarks.utils.benchmark \
   --agg $AGG_CONFIG \
   --disagg $DISAGG_CONFIG \
   --isl $INPUT_SEQUENCE_LENGTH \
   --std $INPUT_SEQUENCE_STD \
   --osl $OUTPUT_SEQUENCE_LENGTH \
   --namespace $NAMESPACE \
   --output-dir $OUTPUT_DIR

# Generate plots from results
python3 benchmarks/utils/plot.py --data-dir ./benchmark_results --output-dir ./benchmark_results/plots
```

## Configuration Options

### Environment Variables

| Variable | Required? | Default | Description |
|----------|-----------|---------|-------------|
| `NAMESPACE` | `Yes` | `None` | Kubernetes namespace |
| `MODEL` | `No` |`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | HuggingFace model name |
| `ISL` | `No` | `200` | Input sequence length (tokens) |
| `STD` | `No` | `10 ` | Input sequence standard deviation |
| `OSL` | `No` | `200` | Output sequence length (tokens) |
| `OUTPUT_DIR` | `No` | `./benchmarks/results` | Local output directory |

### Command Line Options

```bash
./benchmarks/benchmark.sh [OPTIONS]

Options:
  -h, --help         Show help message
  -m, --model        Model name
  -i, --isl          Input sequence length
  -o, --osl          Output sequence length
  -s, --scale        Tensor parallel scale (for multinode)
  -n, --namespace    Kubernetes namespace
  -d, --output-dir   Output directory
  -e, --endpoint     Endpoint URL for existing deployment
  --skip-deployment  Skip deployment creation
```

## Generated Results

### Performance Plots

The benchmark script generates four key performance plots:

1. **`p50_inter_token_latency_vs_concurrency.png`**
   - P50 inter-token latency across concurrency levels
   - Shows how latency degrades with increased load

2. **`avg_inter_token_latency_vs_concurrency.png`**
   - Average inter-token latency across concurrency levels
   - Indicates overall latency trends

3. **`request_throughput_vs_concurrency.png`**
   - Request throughput (req/s) across concurrency levels
   - Shows system capacity and optimal operating points

4. **`avg_time_to_first_token_vs_concurrency.png`**
   - Time to first token across concurrency levels
   - Critical for user experience and response time

### Data Files

- **`benchmark_summary.csv`** - Complete metrics for all tests
- **`BENCHMARK_REPORT.md`** - Human-readable summary report
- **`raw_results/`** - Raw GenAI-Perf output files
- **`logs/`** - Detailed execution logs
- **`configs/`** - Generated Kubernetes configurations

### Result Analysis

#### Interpreting Latency-Throughput Curves

The generated plots show classic latency-throughput trade-offs:

1. **Low Concurrency (1-10)**: Minimum latency, linear throughput scaling
2. **Medium Concurrency (10-100)**: Optimal operating range for most workloads
3. **High Concurrency (100+)**: Latency increases rapidly, throughput plateaus

#### Comparing Deployment Types

- **Dynamo Aggregated**: Best for mixed workloads, good latency and throughput balance
- **Dynamo Disaggregated**: Optimized for high-throughput scenarios with separate prefill/decode
- **Vanilla vLLM**: Baseline comparison, simpler deployment model

## Advanced Usage

### Custom Concurrency Levels

Modify the benchmark script to test specific concurrency patterns:

```python
# Edit the CONCURRENCIES array in benchmarks/utils/genai.py
CONCURRENCIES=[1, 2, 5, 10, 50, 100, 250]
```

### Multi-Node Testing (TODO)

For multi-node Dynamo deployments:

1. Ensure your DGD configuration spans multiple nodes
2. Use appropriate tensor parallelism settings (`SCALE > 1`)
3. Monitor resource utilization across nodes

## Custom Benchmarking

### Bring Your Own Scripts

For custom benchmarking scenarios, you can pass in your benchmarking file to `benchmarks/benchmark.sh`:

```bash
CUSTOM_SCRIPT=<YOUR_SCRIPT_PATH> ./benchmarks/benchmark.sh
```
