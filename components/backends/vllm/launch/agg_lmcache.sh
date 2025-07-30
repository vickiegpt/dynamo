#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

# Enable LMCache
export ENABLE_LMCACHE=1

# Set LMCache configuration environment variables
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=20

# run ingress
python -m dynamo.frontend &

# run worker with LMCache enabled
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --no-enable-prefix-caching