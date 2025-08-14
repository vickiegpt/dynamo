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

Your Kubernetes namespace must be set up for Dynamo deployments. Follow the setup instructions [here](../../deploy/utils/README.md#kubernetes-setup-one-time-per-namespace).

## Quick Start

Run complete benchmarks on Dynamo deployments using the automated script. Plots are generated and saved to `$OUTPUT_DIR/plots`.

### Basic Usage

Use the provided example manifests (configure them for your desired model):

```bash
export NAMESPACE=benchmarking
export AGG_CONFIG=components/backends/vllm/deploy/agg.yaml
export DISAGG_CONFIG=components/backends/vllm/deploy/disagg.yaml
export VANILLA_VLLM_CONFIG=benchmarks/utils/templates/vanilla-vllm.yaml
export OUTPUT_DIR=benchmarks/results

# Complete benchmark with example manifests
./benchmarks/benchmark.sh \
   --namespace $NAMESPACE \
   --agg $AGG_CONFIG \
   --disagg $DISAGG_CONFIG \
   --vanilla $VANILLA_VLLM_CONFIG
```

### Custom Configuration

```bash
# Custom model, sequence lengths, and your own manifests
./benchmarks/benchmark.sh \
   --namespace $NAMESPACE \
   --agg my-custom-agg.yaml \
   --disagg my-custom-disagg.yaml \
   --vanilla my-custom-vanilla.yaml \
   --model "meta-llama/Meta-Llama-3-8B" \
   --isl 512 \
   --osl 512 \
   --std 20 \
   --output-dir benchmark_results
```

**Note**: The deployment manifests determine which model is actually deployed and benchmarked. Make sure your manifests are configured for the model you want to test.

### Direct Python Execution

For direct control over the benchmark workflow:

```bash
# Run benchmark directly with Python
python3 -u -m benchmarks.utils.benchmark \
   --agg $AGG_CONFIG \
   --disagg $DISAGG_CONFIG \
   --vanilla $VANILLA_VLLM_CONFIG \
   --isl 200 \
   --std 10 \
   --osl 200 \
   --namespace $NAMESPACE \
   --output-dir $OUTPUT_DIR

# Generate plots separately
python3 -m benchmarks.utils.plot --data-dir $OUTPUT_DIR
```

## Configuration Options

All configuration is done via command line arguments:

```bash
./benchmarks/benchmark.sh --namespace NAMESPACE --agg CONFIG --disagg CONFIG --vanilla CONFIG [OPTIONS]

REQUIRED:
  -n, --namespace NAMESPACE     Kubernetes namespace
  --agg CONFIG                  Aggregated deployment manifest
  --disagg CONFIG               Disaggregated deployment manifest
  --vanilla CONFIG              Vanilla vLLM deployment manifest

OPTIONS:
  -h, --help                    Show help message and examples
  -m, --model MODEL             Model name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
  -i, --isl LENGTH              Input sequence length (default: 200)
  -s, --std STDDEV              Input sequence standard deviation (default: 10)
  -o, --osl LENGTH              Output sequence length (default: 200)
  -d, --output-dir DIR          Output directory (default: ./benchmarks/results)
  --verbose                     Enable verbose output
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

The benchmark generates structured output in your specified `OUTPUT_DIR`:

```
benchmarks/results/
├── SUMMARY.txt                  # Human-readable benchmark summary
├── plots/                       # Performance visualization plots
│   ├── p50_inter_token_latency_vs_concurrency.png
│   ├── avg_inter_token_latency_vs_concurrency.png
│   ├── request_throughput_vs_concurrency.png
│   └── avg_time_to_first_token_vs_concurrency.png
├── agg/                         # Aggregated deployment results
│   ├── c1/                      # Concurrency level 1
│   │   └── profile_export_genai_perf.json
│   ├── c2/                      # Concurrency level 2
│   └── ...                      # Other concurrency levels
├── disagg/                      # Disaggregated deployment results
│   └── c*/                      # Same structure as agg/
└── vanilla/                     # Vanilla vLLM deployment results
    └── c*/                      # Same structure as agg/
```

Each concurrency directory contains:
- **`profile_export_genai_perf.json`** - Structured metrics from GenAI-Perf
- **`profile_export.json`** - Raw GenAI-Perf results
- **`inputs.json`** - Generated test inputs

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

For custom benchmarking scenarios, you can:

1. **Create custom deployment manifests**: Configure your own agg, disagg, and vanilla manifests for your specific models and hardware configurations

2. **Modify concurrency levels**: Edit `benchmarks/utils/genai.py` to customize test parameters
   ```python
   CONCURRENCIES = [1, 5, 10, 25, 50, 100]  # Your custom levels
   ```

3. **Use direct Python modules**: Call the Python modules directly for full control
   ```bash
   # Custom benchmark workflow
   python3 -m benchmarks.utils.benchmark --help

   # Custom plot generation
   python3 -m benchmarks.utils.plot --help
   ```

4. **Extend the workflow**: Modify `benchmarks/utils/workflow.py` to add custom deployment types or metrics collection

5. **Generate different plots**: Modify `benchmarks/utils/plot.py` to generate a different set of plots for whatever you wish to visualize.

The `benchmark.sh` script provides a complete end-to-end benchmarking experience. For more granular control, use the Python modules directly.
