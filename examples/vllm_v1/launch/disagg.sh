#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cleanup() {
    echo "Cleaning up..."
    kill $DYNAMO_PID $WORKER_PID 2>/dev/null || true
    wait $DYNAMO_PID $WORKER_PID 2>/dev/null || true
}
trap cleanup EXIT ERR INT TERM

# run ingress
dynamo run in=http out=dyn &
DYNAMO_PID=$!

CUDA_VISIBLE_DEVICES=0 python3 main.py --model Qwen/Qwen3-0.6B --enforce-eager &
WORKER_PID=$!

CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --is-prefill-worker
