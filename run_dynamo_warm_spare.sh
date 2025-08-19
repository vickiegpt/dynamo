#!/bin/bash
# Run Dynamo vLLM with warm spare enabled

# Environment variables for vLLM V1 and two-phase init
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_TWO_PHASE_INIT=1
export VLLM_SKIP_WARM_SPARE_MEMORY_PROFILING=1
export VLLM_USE_V1=1

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TP="${TP:-2}"

echo "Running Dynamo vLLM with warm spare support"
echo "Model: $MODEL"
echo "Tensor Parallel: $TP"
echo ""
echo "Command:"
echo "python3 -m dynamo.vllm --model $MODEL -tp $TP --enable-ipc-loading --use-warm-spare"
echo ""

# Run with warm spare enabled
python3 -m dynamo.vllm \
    --model "$MODEL" \
    -tp "$TP" \
    --enable-ipc-loading \
    --use-warm-spare \
    "$@"
