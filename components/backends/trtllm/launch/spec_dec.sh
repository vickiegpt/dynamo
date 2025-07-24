#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
# TODO: remove the local model path
export MODEL_PATH=${MODEL_PATH:-"/tmp/huggingface_cache/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
export DRAFTER_MODEL_NAME=${DRAFTER_MODEL_NAME:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
export VERIFIER_ENGINE_ARGS=${VERIFIER_ENGINE_ARGS:-"engine_configs/verifier.yaml"}
export DRAFTER_ENGINE_ARGS=${DRAFTER_ENGINE_ARGS:-"engine_configs/drafter.yaml"}
export VERIFIER_CUDA_VISIBLE_DEVICES=${VERIFIER_CUDA_VISIBLE_DEVICES:-"0"}
export DRAFTER_CUDA_VISIBLE_DEVICES=${DRAFTER_CUDA_VISIBLE_DEVICES:-"1"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $VERIFIER_PID 2>/dev/null || true
    wait $DYNAMO_PID $VERIFIER_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

echo -e "\n"
echo "----------------------------------------------------------------------------------------------------"
echo "STARTING SET UP"
echo "----------------------------------------------------------------------------------------------------"
echo -e "Clearing namespace..."

# run clear_namespace
python3 utils/clear_namespace.py --namespace dynamo

echo -e "----------------------------------------------------------------------------------------------------"
echo -e "Running frontend..."
echo -e "Running verifier worker..."
echo -e "Running drafter worker..."
echo -e "\n\n" 

# run frontend
python3 -m dynamo.frontend --http-port 8000 &
DYNAMO_PID=$!

# run verifier worker with speculative decoding
CUDA_VISIBLE_DEVICES=$VERIFIER_CUDA_VISIBLE_DEVICES \
PYTHONPATH="/workspace/components/backends/trtllm/src:$PYTHONPATH" \
python3 -m dynamo.verifier \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$VERIFIER_ENGINE_ARGS" &
VERIFIER_PID=$!

# run drafter worker without speculative decoding
CUDA_VISIBLE_DEVICES=$DRAFTER_CUDA_VISIBLE_DEVICES \
PYTHONPATH="/workspace/components/backends/trtllm/src:$PYTHONPATH" \
python3 -m dynamo.drafter \
  --endpoint "dyn://dynamo.tensorrt_llm.generate_draft" \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DRAFTER_ENGINE_ARGS"
