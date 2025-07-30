#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e


# run ingress with KV router
python -m dynamo.frontend --router-mode kv &

# run decode worker on GPU 0, without enabling LMCache
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --no-enable-prefix-caching &

# wait for decode worker to initialize
sleep 20

# Enable LMCache
export ENABLE_LMCACHE=1

# Set LMCache configuration environment variables
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=20

# run prefill worker on GPU 1 with LMCache
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --is-prefill-worker