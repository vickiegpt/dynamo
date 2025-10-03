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

# AIConfigurator End-to-End Benchmarking Guide

This guide demonstrates how to use AIConfigurator to optimize Dynamo deployments and validate its performance improvements through end-to-end benchmarking.

## Overview

This benchmark compares two configurations:

- **Baseline (A)**: Standard TensorRT-LLM disaggregated deployment (1 prefill + 1 decode worker, 1 GPU each)
- **AIConfigurator-Optimized (B)**: AIConfigurator-recommended deployment (aggregated or disaggregated) with optimized parameters

**Goal**: Demonstrate the relative performance increase conferred by using AIConfigurator and validate the accuracy of AIConfigurator's projections versus benchmarked reality. AIConfigurator will automatically choose the best configuration type (aggregated vs disaggregated) and parameters for the given constraints.

## Test Configuration

- **Model**: Qwen3-32B
- **System**: H200 SXM
- **Total GPUs**: 8
- **Baseline Configuration**: Disaggregated (1P1D) - 1 prefill worker + 1 decode worker, 1 GPU each
- **AIConfigurator Configuration**: Determined through running AIConfigurator CLI
- **Input Sequence Length (ISL)**: 4000 tokens
- **Output Sequence Length (OSL)**: 500 tokens
- **SLA Target**: AIC defaults
- **Kubernetes Namespace**: `hannahz`
- **Docker Image**: `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.0`

## AIConfigurator Predicted Results

AIConfigurator will analyze the configuration space and provide:
- **Baseline Performance**: Performance predictions for the disaggregated baseline (1P1D)
- **Recommended Configuration**: Best configuration type (aggregated or disaggregated) with optimized parameters
- **Expected Improvement**: Performance optimization through configuration and parameter tuning

The AIConfigurator-recommended configuration will use:
- Optimal deployment pattern (aggregated vs disaggregated)
- Optimized GPU allocation and tensor parallelism
- Tuned batch sizes and memory settings
- Optimized engine parameters for the specific workload

**Note**: The specific configuration details will be shown in the AIConfigurator output after running Step 2.

## Prerequisites

1. **Kubernetes cluster** with NVIDIA GPUs (H200 SXM) and Dynamo installed
2. **Storage and service account** configured (see [deploy/utils README](../../deploy/utils/README.md))
3. **Model cache PVC** with Qwen3-32B model available at `/workspace/model_hub/qwen3-32b` (if using model cache)
4. **HuggingFace token secret** named `hf-token-secret` (if needed for model access)
5. **Docker image pull secret** named `docker-imagepullsecret` (if using private registry)
6. **GPU Requirements**:
   - **Recommended**: 2x8 H200 GPUs (16 total) for parallel deployment
   - **Minimum**: 8 H200 GPUs for sequential deployment

## Step-by-Step Benchmark Instructions

### Step 0: Set Environment Variables

```bash
export NAMESPACE=hannahz
export NAMESPACE_2=hannahz-2 # note: if you set a different NAMESPACE_2, you'll need to modify the namespace in the benchmark job(s)
```

**Note**: This guide assumes you have 2x8 H200 GPUs available to run both deployments simultaneously. If you only have 8 H200 GPUs, see the [Single GPU Setup](#single-gpu-setup) section for sequential deployment instructions.

### Step 1: Install AIConfigurator

First, install AIConfigurator from source:

```bash
cd /home/ubuntu/dynamo

# Clone aiconfigurator repository
git clone https://github.com/ai-dynamo/aiconfigurator.git
cd aiconfigurator

# Install Git LFS and pull performance data files
# AIConfigurator requires performance data files stored in Git LFS
git lfs install
git lfs pull
git lfs checkout

# Install in editable mode
python3 -m pip install -e .

# Verify installation
aiconfigurator --help
```

### Step 2: Generate AIConfigurator-Optimized Configuration

Generate the optimized configuration using AIConfigurator CLI:

```bash
cd /home/ubuntu/dynamo

# Create output directory
mkdir -p aiconf_save

# Run AIConfigurator
aiconfigurator cli \
  --model QWEN3_32B \
  --system h200_sxm \
  --total_gpus 8 \
  --isl 4000 --osl 500 \
  --save_dir ./aiconf_save \
  --model_path Qwen/Qwen3-32B \
  --served_model_name Qwen/Qwen3-32B \
  --k8s_use_model_cache false \
  --k8s_namespace $NAMESPACE_2 \
  --k8s_image nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.0 \
  --cache_transceiver_backend default

# Set the AIConfigurator output directory (replace with your actual directory name)
# The AIC logs will show something like "saving results to ./aiconf_save/QWEN3_32B_isl4000_osl500_ttft300_tpot10_442838"
export AICONF_DIR=./aiconf_save/QWEN3_32B_XXXXXX

# Check which configuration AIConfigurator recommended:
# - Look for "Overall best system chosen: agg" or "Overall best system chosen: disagg" in the output
# - Or check within $AICONF_DIR/aiconfigurator_result.json
```

**Output**: AIConfigurator will analyze the configuration space and generate:
- Performance predictions and Pareto frontier analysis
- Kubernetes deployment YAML for the recommended configuration (aggregated or disaggregated)
- Engine configuration files with optimized parameters

**Note**: If the generated deployment includes NIXL wheel installation commands that fail, manually remove the NIXL-related lines from the generated `k8s_deploy.yaml` file. The deployment should work with the default communication backend.

The generated files will be located at:
```
aiconf_save/QWEN3_32B_isl4000_osl500_*/
├── aiconfigurator_config.yaml
└── backend_configs/
    └── {agg|disagg}/
        ├── k8s_deploy.yaml
        └── {agg|disagg}_config.yaml
```

**Note**:
- The `--k8s_use_model_cache false` flag tells AIConfigurator to use Hugging Face model names directly instead of expecting a local model cache. TODO: add model cache support to this doc

**Note the predictions** from the AIConfigurator output - we'll compare these against actual benchmark results.

### Step 3: Deploy Baseline Configuration

Deploy the standard disaggregated TensorRT-LLM configuration:

```bash
# Deploy baseline configuration
kubectl apply -f components/backends/trtllm/deploy/disagg_baseline.yaml -n $NAMESPACE

# Wait for deployment to be ready
kubectl wait --for=condition=ready pod -l dynamoNamespace=trtllm-disagg-baseline -n $NAMESPACE --timeout=600s

# Verify deployment
kubectl get pods -n $NAMESPACE -l dynamoNamespace=trtllm-disagg-baseline
```

**Configuration Details**:
- 1 prefill worker, 1 GPU
- 1 decode worker, 1 GPU
- Disaggregated mode (separate prefill and decode workers)
- Model path: `Qwen/Qwen3-32B` (HuggingFace)
- Service name: `trtllm-disagg-baseline-frontend`

### Step 4: Deploy AIConfigurator Configuration

Deploy the AIConfigurator-optimized configuration in the second namespace:

**If AIConfigurator recommended aggregated configuration:**
```bash
# Create ConfigMap for aggregated engine config
kubectl create configmap engine-configs \
  --from-file=agg_config.yaml=$AICONF_DIR/backend_configs/agg/agg_config.yaml \
  -n $NAMESPACE_2 \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy AIConfigurator-optimized aggregated configuration
kubectl apply -f $AICONF_DIR/backend_configs/agg/k8s_deploy.yaml -n $NAMESPACE_2

# Wait for deployment to be ready
kubectl wait --for=condition=ready pod -l dynamoNamespace=trtllm-agg -n $NAMESPACE_2 --timeout=600s

# Verify deployment
kubectl get pods -n $NAMESPACE_2 -l dynamoNamespace=trtllm-agg
```

**If AIConfigurator recommended disaggregated configuration:**
```bash
# Create ConfigMap for disaggregated engine config
kubectl create configmap engine-configs \
  --from-file=disagg_config.yaml=$AICONF_DIR/backend_configs/disagg/disagg_config.yaml \
  -n $NAMESPACE_2 \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy AIConfigurator-optimized disaggregated configuration
kubectl apply -f $AICONF_DIR/backend_configs/disagg/k8s_deploy.yaml -n $NAMESPACE_2

# Wait for deployment to be ready
kubectl wait --for=condition=ready pod -l dynamoNamespace=trtllm-disagg -n $NAMESPACE_2 --timeout=600s

# Verify deployment
kubectl get pods -n $NAMESPACE_2 -l dynamoNamespace=trtllm-disagg
```

**Configuration Details**:
- Configuration type (aggregated or disaggregated) determined by AIConfigurator
- Optimized GPU allocation and tensor parallelism
- Optimized batch sizes and memory settings
- Tuned engine parameters for better performance
- Service names: `trtllm-agg-frontend` (if aggregated) or `trtllm-disagg-frontend` (if disaggregated)

**Why ConfigMap is needed**: The AIConfigurator-optimized deployment uses a custom engine configuration file with tuned parameters, while the baseline uses the built-in engine configs.

### Step 5: Benchmark Both Configurations (Parallel)

Run the in-cluster benchmarks for both deployments simultaneously:

```bash
# Deploy benchmark job for baseline
kubectl apply -f benchmarks/incluster/benchmark_baseline_job.yaml -n $NAMESPACE
```

**If AIConfigurator recommended aggregated configuration:**
```bash
# Deploy AIConfigurator benchmark job for aggregated
kubectl apply -f benchmarks/incluster/benchmark_aic_agg_job.yaml -n $NAMESPACE
```

**If AIConfigurator recommended disaggregated configuration:**
```bash
# Deploy AIConfigurator benchmark job for disaggregated
kubectl apply -f benchmarks/incluster/benchmark_aic_disagg_job.yaml -n $NAMESPACE
```

**Monitor and wait for both benchmarks:**
```bash
# Monitor both benchmarks
kubectl logs -f job/dynamo-benchmark-baseline -n $NAMESPACE &
kubectl logs -f job/dynamo-benchmark-aic -n $NAMESPACE &

# Wait for both to complete
kubectl wait --for=condition=complete job/dynamo-benchmark-baseline -n $NAMESPACE --timeout=3600s
kubectl wait --for=condition=complete job/dynamo-benchmark-aic -n $NAMESPACE --timeout=3600s
```

### Step 6: Retrieve and Analyze Results

Download the benchmark results from the PVC:

```bash
# Download baseline results
python3 -m deploy.utils.download_pvc_results \
  --namespace $NAMESPACE \
  --output-dir ./aiconf_save/benchmark_results/qwen3-32b-trtllm-baseline \
  --folder /data/results/qwen3-32b-trtllm-baseline \
  --no-config

# Download AIConfigurator results
python3 -m deploy.utils.download_pvc_results \
  --namespace $NAMESPACE \
  --output-dir ./aiconf_save/benchmark_results/qwen3-32b-trtllm-aic \
  --folder /data/results/qwen3-32b-trtllm-aic \
  --no-config

# Generate comparison plots
python3 -m benchmarks.utils.plot \
  --data-dir ./benchmarks/results \
  --benchmark-name qwen3-32b-trtllm-baseline \
  --benchmark-name qwen3-32b-trtllm-aic
```

### Step 8: Compare Results

The plotting script will generate:

```
benchmarks/results/plots/
├── SUMMARY.txt                                     # Performance comparison summary
├── p50_inter_token_latency_vs_concurrency.png      # Token generation speed
├── avg_time_to_first_token_vs_concurrency.png      # Time to first token
├── request_throughput_vs_concurrency.png           # Requests per second
├── efficiency_tok_s_gpu_vs_user.png                # GPU efficiency
└── avg_inter_token_latency_vs_concurrency.png      # Average latency
```

**Key Metrics to Compare**:

1. **Throughput (tokens/s/GPU)**: Compare disaggregated baseline vs AIConfigurator-recommended configuration
   - Baseline: 1P1D disaggregated performance
   - AIConfigurator: Optimized configuration (aggregated or disaggregated) performance

2. **User Throughput (tokens/s/user)**: End-to-end throughput from user perspective
   - Compare baseline disaggregated vs AIConfigurator-recommended configuration

3. **TTFT (Time to First Token)**: Response latency
   - Compare baseline vs AIConfigurator-optimized latency

4. **TPOT (Time Per Output Token)**: Token generation latency
   - Compare baseline vs AIConfigurator-optimized token generation speed

5. **Configuration Efficiency**: Evaluate whether AIConfigurator chose aggregated or disaggregated and why

6. **Accuracy of Predictions**: Compare AIConfigurator predictions vs actual benchmarked results

## Understanding the Results

### Expected Outcomes

Based on AIConfigurator's analysis, we expect:

1. **Optimized Performance**: The AIConfigurator-recommended configuration should achieve better performance due to:
   - Optimal choice between aggregated and disaggregated deployment patterns
   - Tuned batch sizes and memory settings
   - Optimized engine parameters for the specific workload
   - Better resource utilization and GPU allocation

2. **Performance Comparison**: Direct comparison of throughput and latency between baseline and optimized configurations

3. **Improved Efficiency**: AIConfigurator should provide better tokens/s/GPU through configuration and parameter optimization

4. **Configuration Insights**: AIConfigurator will demonstrate whether aggregated or disaggregated is better for this specific workload

5. **Validated Predictions**: The actual benchmark results should validate AIConfigurator's performance predictions

### Validating AIConfigurator Accuracy

Compare the actual benchmark results against AIConfigurator's predictions to evaluate:
- Prediction accuracy for throughput metrics
- Accuracy of latency predictions (TTFT, TPOT)
- Real-world performance gain vs predicted improvement

## Configuration Files Reference

### Baseline Configuration
- **Deployment**: `components/backends/trtllm/deploy/disagg_baseline.yaml`
- **Engine Config**: Built-in `engine_configs/prefill.yaml` and `engine_configs/decode.yaml`
- **Workers**: 1 prefill worker (1 GPU) + 1 decode worker (1 GPU)

### AIConfigurator-Optimized Configuration
- **Deployment**: `aiconf_save/QWEN3_32B_*/backend_configs/{agg|disagg}/k8s_deploy.yaml`
- **ConfigMap**: Created dynamically from `{agg|disagg}_config.yaml`
- **Engine Config**: Configuration file with optimized parameters
- **Workers**: Determined by AIConfigurator (aggregated or disaggregated)

### Benchmark Jobs
- **Baseline**: `benchmarks/incluster/benchmark_baseline_job.yaml`
- **AIConfigurator Aggregated**: `benchmarks/incluster/benchmark_aic_agg_job.yaml`
- **AIConfigurator Disaggregated**: `benchmarks/incluster/benchmark_aic_disagg_job.yaml`
- **Parameters**: ISL=4000, OSL=500, matching AIConfigurator input

### Cleanup

To clean up all resources:

```bash
# Delete baseline deployment
kubectl delete dynamographdeployment trtllm-disagg-baseline -n $NAMESPACE

# Delete AIConfigurator deployment (choose based on what was deployed)
kubectl delete dynamographdeployment trtllm-agg -n $NAMESPACE_2      # If aggregated
kubectl delete dynamographdeployment trtllm-disagg -n $NAMESPACE_2   # If disaggregated

# Delete ConfigMaps
kubectl delete configmap engine-configs -n $NAMESPACE_2

# Delete benchmark jobs
kubectl delete job dynamo-benchmark-baseline -n $NAMESPACE
kubectl delete job dynamo-benchmark-aic -n $NAMESPACE
```

---

## Customizing the Benchmark

This guide provides a template for benchmarking AIConfigurator with different models, hardware, and configurations. Here's how to adapt it:

### Different Models

To benchmark different models, modify:
1. **AIConfigurator command**: `--model` parameter (e.g., `LLAMA2_70B`, `MIXTRAL_8X7B`)
2. **Deployment YAMLs**: Update `--model-path` and `--served-model-name` in both baseline and AIConfigurator configs
3. **Benchmark jobs**: Update `--model` parameter in benchmark job files
4. **Baseline choice**: Consider whether 1P1D disaggregated is still the appropriate baseline for larger models

### Different Hardware

For different GPU types:
1. **AIConfigurator command**: `--system` parameter (e.g., `h100_sxm`, `a100_sxm`)
2. **GPU allocation**: Adjust `--total_gpus` based on available hardware
3. **Deployment configs**: Update GPU resource limits in YAML files
4. **Baseline configuration**: May need different baseline (e.g., 2P2D for larger models)

### Different Workload Parameters

Modify AIConfigurator parameters based on your use case:
- `--isl` and `--osl`: Input/output sequence lengths for your workload
- `--ttft` and `--tpot`: Optional latency constraints if you have SLA requirements
- `--total_gpus`: Available GPU budget
