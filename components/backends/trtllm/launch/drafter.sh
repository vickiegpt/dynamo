#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
export MODEL_PATH=${MODEL_PATH:-"/tmp/huggingface_cache/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"engine_configs/drafter.yaml"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID 2>/dev/null || true
    wait $DYNAMO_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run clear_namespace
python3 utils/clear_namespace.py --namespace dynamo

# run drafter worker
PYTHONPATH="/workspace/components/backends/trtllm/src:$PYTHONPATH" \
python3 -m dynamo.drafter \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS"
