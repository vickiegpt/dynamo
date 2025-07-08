#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

WORKER_PIDS=()

cleanup() {
    echo "Cleaning up..."
    kill $DYNAMO_PID "${WORKER_PIDS[@]}" 2>/dev/null || true
    wait $DYNAMO_PID "${WORKER_PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT ERR INT TERM

# run ingress
DYN_LOG=debug dynamo run in=http out=dyn --router-mode kv &
DYNAMO_PID=$!

# routing will happen between the two decode workers
CUDA_VISIBLE_DEVICES=0 python3 components/main.py --model Qwen/Qwen3-0.6B --enforce-eager &
WORKER_PIDS+=($!)

CUDA_VISIBLE_DEVICES=1 python3 components/main.py --model Qwen/Qwen3-0.6B --enforce-eager &
WORKER_PIDS+=($!)

CUDA_VISIBLE_DEVICES=2 python3 components/main.py \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --is-prefill-worker
