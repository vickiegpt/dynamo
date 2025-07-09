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
DYN_LOG=debug $HOME/dynamo/.build/target/debug/dynamo-run in=http out=dyn --router-mode kv &
DYNAMO_PID=$!

# Data Parallel Attention / Expert Parallelism
# Routing to DP workers managed by Dynamo
# Chose Qwen3-30B because its a small MOE that can fit on smaller GPUs (L40S for example)
for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i python3 components/main.py \
    --model Qwen/Qwen3-30B-A3B \
    --data-parallel-rank $i \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --enforce-eager \
    --kv-events-port 49500 &
    WORKER_PIDS+=($!)
done

# Wait for all background processes to finish
echo "All workers starting. (press Ctrl+C to stop)..."
wait $DYNAMO_PID "${WORKER_PIDS[@]}"
