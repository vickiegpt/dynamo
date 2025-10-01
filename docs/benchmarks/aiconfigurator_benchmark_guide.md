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

- **Baseline (A)**: Standard TensorRT-LLM aggregated deployment with default configuration
- **AIConfigurator-Optimized (B)**: AIConfigurator-optimized aggregated deployment with tuned parameters

**Goal**: Demonstrate the relative performance increase conferred by using AIConfigurator and validate the accuracy of AIConfigurator's projections versus benchmarked reality.

## Test Configuration

- **Model**: Qwen3-32B
- **System**: H200 SXM
- **Total GPUs**: 8
- **Input Sequence Length (ISL)**: 4000 tokens
- **Output Sequence Length (OSL)**: 500 tokens
- **SLA Target**: TTFT ≤ 500ms, TPOT ≤ 7ms
- **Kubernetes Namespace**: `hannahz`
- **Docker Image**: `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.0`

## AIConfigurator Predicted Results

AIConfigurator analysis predicts:
- **Baseline Aggregated**: 339.51 tokens/s/GPU (143.14 tokens/s/user) | TTFT: 189.94ms, TPOT: 6.99ms
- **AIConfigurator Aggregated**: Optimized aggregated configuration with tuned parameters
- **Expected Improvement**: Performance optimization through parameter tuning

The AIConfigurator-optimized aggregated configuration uses:
- 1 worker with 8 GPUs (TP=8)
- Optimized batch size (12) and memory settings
- Tuned engine parameters for better performance

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
export NAMESPACE_2=hannahz-2
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
  --ttft 500 --tpot 7 \
  --save_dir ./aiconf_save \
  --model_path Qwen/Qwen3-32B \
  --served_model_name Qwen/Qwen3-32B \
  --k8s_use_model_cache false \
  --k8s_namespace $NAMESPACE_2 \
  --k8s_image nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.0 \
  --cache_transceiver_backend default
```

**Output**: AIConfigurator will analyze the configuration space and generate:
- Performance predictions and Pareto frontier analysis
- Kubernetes deployment YAML for aggregated configuration
- Engine configuration files with optimized parameters

**Note**: If the generated deployment includes NIXL wheel installation commands that fail, manually remove the NIXL-related lines from the generated `k8s_deploy.yaml` file. The deployment should work with the default communication backend.

The generated files will be located at:
```
aiconf_save/QWEN3_32B_isl4000_osl500_ttft500_tpot7_*/
├── aiconfigurator_config.yaml
└── backend_configs/
    └── agg/
        ├── k8s_deploy.yaml
        └── agg_config.yaml
```

**Note**:
- The `--k8s_use_model_cache false` flag tells AIConfigurator to use Hugging Face model names directly instead of expecting a local model cache. TODO: add model cache support to this doc

**Note the predictions** from the AIConfigurator output - we'll compare these against actual benchmark results.

### Step 3: Deploy Baseline Configuration

Deploy the standard aggregated TensorRT-LLM configuration:

```bash
# Deploy baseline configuration
kubectl apply -f components/backends/trtllm/deploy/agg_baseline.yaml -n $NAMESPACE

# Wait for deployment to be ready
kubectl wait --for=condition=ready pod -l dynamoNamespace=trtllm-agg-baseline -n $NAMESPACE --timeout=600s

# Verify deployment
kubectl get pods -n $NAMESPACE -l dynamoNamespace=trtllm-agg-baseline
```

**Configuration Details**:
- 8 workers, 1 GPU each
- Aggregated mode (prefill and decode on same workers)
- Model path: `/workspace/model_hub/qwen3-32b`
- Service name: `trtllm-agg-baseline-frontend`

### Step 4: Benchmark Baseline Configuration

Run the in-cluster benchmark for the baseline:

```bash
# Deploy benchmark job
kubectl apply -f benchmarks/incluster/benchmark_baseline_job.yaml -n $NAMESPACE

# Monitor the benchmark
kubectl logs -f job/dynamo-benchmark-baseline -n $NAMESPACE

# Wait for completion
kubectl wait --for=condition=complete job/dynamo-benchmark-baseline -n $NAMESPACE --timeout=3600s
```

The benchmark will:
- Test at various concurrency levels (1, 2, 5, 10, 50, 100, 250)
- Measure TTFT, TPOT, throughput, and latency
- Store results in `/data/results/qwen3-32b-trtllm-agg-baseline/`

### Step 5: Deploy AIConfigurator Configuration (Parallel)

Deploy the AIConfigurator-optimized configuration in the second namespace:

```bash
# Create ConfigMap for AIConfigurator engine configs dynamically
AICONF_DIR=$(ls -d aiconf_save/QWEN3_32B_isl4000_osl500_ttft500_tpot7_* | head -n1)
kubectl create configmap engine-configs \
  --from-file=agg_config.yaml=$AICONF_DIR/backend_configs/agg/agg_config.yaml \
  -n $NAMESPACE_2 \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy AIConfigurator-optimized aggregated configuration
kubectl apply -f $AICONF_DIR/backend_configs/agg/k8s_deploy.yaml -n $NAMESPACE_2

# Wait for deployment to be ready
kubectl wait --for=condition=ready pod -l dynamoNamespace=trtllm-agg-aic -n $NAMESPACE_2 --timeout=600s

# Verify deployment
kubectl get pods -n $NAMESPACE_2 -l dynamoNamespace=trtllm-agg-aic
```

**Configuration Details**:
- 1 worker with 8 GPUs (TP=8)
- Optimized batch size (12) and memory settings
- Tuned engine parameters for better performance
- Service name: `trtllm-agg-aic-frontend`

**Why ConfigMap is needed**: The AIConfigurator-optimized deployment uses a custom engine configuration file (`agg_config.yaml`) with tuned parameters, while the baseline uses the built-in `engine_configs/agg.yaml`. The ConfigMap is created dynamically from the generated `agg_config.yaml` file.

### Step 6: Benchmark Both Configurations (Parallel)

Run the in-cluster benchmarks for both deployments simultaneously:

```bash
# Deploy benchmark job for baseline
kubectl apply -f benchmarks/incluster/benchmark_baseline_job.yaml -n $NAMESPACE

# Deploy benchmark job for AIConfigurator
kubectl apply -f benchmarks/incluster/benchmark_aic_agg_job.yaml -n $NAMESPACE

# Monitor both benchmarks
kubectl logs -f job/dynamo-benchmark-baseline -n $NAMESPACE &
kubectl logs -f job/dynamo-benchmark-aic-agg -n $NAMESPACE &

# Wait for both to complete
kubectl wait --for=condition=complete job/dynamo-benchmark-baseline -n $NAMESPACE --timeout=3600s
kubectl wait --for=condition=complete job/dynamo-benchmark-aic-agg -n $NAMESPACE --timeout=3600s
```

### Step 7: Retrieve and Analyze Results

Download the benchmark results from the PVC:

```bash
# Download baseline results
python3 -m deploy.utils.download_pvc_results \
  --namespace $NAMESPACE \
  --output-dir ./benchmarks/results/qwen3-32b-trtllm-agg-baseline \
  --folder /data/results/qwen3-32b-trtllm-agg-baseline \
  --no-config

# Download AIConfigurator results
python3 -m deploy.utils.download_pvc_results \
  --namespace $NAMESPACE \
  --output-dir ./benchmarks/results/qwen3-32b-trtllm-aic-agg \
  --folder /data/results/qwen3-32b-trtllm-aic-agg \
  --no-config

# Generate comparison plots
python3 -m benchmarks.utils.plot \
  --data-dir ./benchmarks/results \
  --benchmark-name qwen3-32b-trtllm-agg-baseline \
  --benchmark-name qwen3-32b-trtllm-aic-agg
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

1. **Throughput (tokens/s/GPU)**: Compare actual vs AIConfigurator prediction
   - AIConfigurator predicted: 339.51 (baseline) vs optimized aggregated configuration

2. **User Throughput (tokens/s/user)**: End-to-end throughput from user perspective
   - AIConfigurator predicted: 143.14 (baseline) vs optimized configuration

3. **TTFT (Time to First Token)**: Response latency
   - AIConfigurator predicted: 189.94ms (baseline) vs optimized configuration

4. **TPOT (Time Per Output Token)**: Token generation latency
   - AIConfigurator predicted: 6.99ms (baseline) vs optimized configuration

5. **Accuracy of Predictions**: Compare AIConfigurator predictions vs actual benchmarked results

## Understanding the Results

### Expected Outcomes

Based on AIConfigurator's analysis, we expect:

1. **Optimized Performance**: The AIConfigurator-optimized aggregated configuration should achieve better performance due to:
   - Tuned batch sizes and memory settings
   - Optimized engine parameters for the specific workload
   - Better resource utilization

2. **SLA Compliance**: Both configurations should meet the SLA targets (TTFT ≤ 500ms, TPOT ≤ 7ms)

3. **Improved Efficiency**: AIConfigurator should provide better tokens/s/GPU through parameter optimization

4. **Validated Predictions**: The actual benchmark results should validate AIConfigurator's performance predictions

### Validating AIConfigurator Accuracy

Compare the actual benchmark results against AIConfigurator's predictions to evaluate:
- Prediction accuracy for throughput metrics
- Accuracy of SLA compliance (TTFT, TPOT)
- Real-world performance gain vs predicted improvement

## Configuration Files Reference

### Baseline Configuration
- **Deployment**: `components/backends/trtllm/deploy/agg_baseline.yaml`
- **Engine Config**: Built-in `engine_configs/agg.yaml`
- **Workers**: 8x 1-GPU aggregated workers

### AIConfigurator-Optimized Configuration
- **Deployment**: `aiconf_save/QWEN3_32B_*/backend_configs/agg/k8s_deploy.yaml`
- **ConfigMap**: Created dynamically from `agg_config.yaml`
- **Engine Config**: `agg_config.yaml` (TP=8, batch_size=12, free_gpu_memory=0.8)
- **Workers**: 1 worker (8 GPUs)

### Benchmark Jobs
- **Baseline**: `benchmarks/incluster/benchmark_baseline_job.yaml`
- **AIConfigurator**: `benchmarks/incluster/benchmark_aic_agg_job.yaml`
- **Parameters**: ISL=4000, OSL=500, matching AIConfigurator input

### Cleanup

To clean up all resources:

```bash
# Delete deployments
kubectl delete -f components/backends/trtllm/deploy/agg_baseline.yaml -n $NAMESPACE
AICONF_DIR=$(ls -d aiconf_save/QWEN3_32B_isl4000_osl500_ttft500_tpot7_* | head -n1)
kubectl delete -f $AICONF_DIR/backend_configs/agg/k8s_deploy.yaml -n $NAMESPACE_2

# Delete ConfigMaps
kubectl delete configmap engine-configs -n $NAMESPACE_2

# Delete benchmark jobs
kubectl delete job dynamo-benchmark-baseline -n $NAMESPACE
kubectl delete job dynamo-benchmark-aic-agg -n $NAMESPACE_2
```

## Single GPU Setup

If you only have 8 H200 GPUs available, you can run the benchmark sequentially by tearing down deployments between steps:

### Sequential Deployment Steps

1. **Deploy and benchmark baseline** (Steps 3-4)
2. **Teardown baseline**:
   ```bash
   kubectl delete -f components/backends/trtllm/deploy/agg_baseline.yaml -n $NAMESPACE
   kubectl wait --for=delete pod -l dynamoNamespace=trtllm-agg-baseline -n $NAMESPACE --timeout=300s
   ```
3. **Deploy and benchmark AIConfigurator** (Steps 5-6, but use `$NAMESPACE` instead of `$NAMESPACE_2`)
4. **Teardown AIConfigurator**:
   ```bash
   AICONF_DIR=$(ls -d aiconf_save/QWEN3_32B_isl4000_osl500_ttft500_tpot7_* | head -n1)
   kubectl delete -f $AICONF_DIR/backend_configs/agg/k8s_deploy.yaml -n $NAMESPACE
   kubectl delete configmap engine-configs -n $NAMESPACE
   ```

### Modified Commands for Single GPU Setup

Replace `$NAMESPACE_2` with `$NAMESPACE` in the AIConfigurator deployment commands:

```bash
# AIConfigurator command
aiconfigurator cli \
  --model QWEN3_32B \
  --system h200_sxm \
  --total_gpus 8 \
  --isl 4000 --osl 500 \
  --ttft 500 --tpot 7 \
  --save_dir ./aiconf_save \
  --model_path Qwen/Qwen3-32B \
  --served_model_name Qwen/Qwen3-32B \
  --k8s_use_model_cache false \
  --k8s_namespace $NAMESPACE \
  --k8s_image nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.0 \
  --cache_transceiver_backend default

# Deploy AIConfigurator configuration
AICONF_DIR=$(ls -d aiconf_save/QWEN3_32B_isl4000_osl500_ttft500_tpot7_* | head -n1)
kubectl create configmap engine-configs \
  --from-file=agg_config.yaml=$AICONF_DIR/backend_configs/agg/agg_config.yaml \
  -n $NAMESPACE
kubectl apply -f $AICONF_DIR/backend_configs/agg/k8s_deploy.yaml -n $NAMESPACE

# Benchmark AIConfigurator
kubectl apply -f benchmarks/incluster/benchmark_aic_agg_job.yaml -n $NAMESPACE
```

---

## Customizing the Benchmark

### Different Models

To benchmark different models, modify:
1. AIConfigurator command: `--model` parameter
2. Deployment YAMLs: `--model-path` and `--served-model-name`
3. Benchmark jobs: `--model` parameter

### Different Hardware

For different GPU types (H100, A100):
1. AIConfigurator command: `--system` parameter (e.g., `h100_sxm`)
2. Adjust GPU counts and topology as needed

### Different SLA Targets

Modify AIConfigurator parameters:
- `--isl` and `--osl`: Input/output sequence lengths
- `--ttft` and `--tpot`: Target latency constraints

### Different Concurrency Levels

Set the `CONCURRENCIES` environment variable in benchmark jobs:
```yaml
env:
  - name: CONCURRENCIES
    value: "1,2,5,10,25,50,100"
```

## Additional Resources

- [Dynamo Benchmarking Guide](./benchmarking.md)
- [AIConfigurator Documentation](https://github.com/ai-dynamo/aiconfigurator)
- [TensorRT-LLM Backend Documentation](../../components/backends/trtllm/README.md)
- [Deploy Utils README](../../deploy/utils/README.md)

## Conclusion

This benchmark demonstrates:
1. How to use AIConfigurator to optimize Dynamo deployments
2. How to validate AIConfigurator's predictions through real-world benchmarking
3. The performance benefits of using AIConfigurator for deployment optimization
4. A reproducible methodology for comparing different deployment configurations

The results provide quantitative evidence of AIConfigurator's value in optimizing LLM serving deployments and validate its prediction accuracy for real-world workloads.
