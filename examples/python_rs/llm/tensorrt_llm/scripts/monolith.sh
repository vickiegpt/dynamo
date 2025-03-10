#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <model-name> <world-size> <num-workers> <gpus-per-worker> [--kv-aware]"
  echo "Example: $0 TinyLlama/TinyLlama-1.1B-Chat-v1.0 4 2 2 --kv-aware"
  exit 1
fi

MODEL_NAME=$1
WORLD_SIZE=$2
NUM_WORKERS=$3
GPUS_PER_WORKER=$4
KV_AWARE=false

# Check for the KV-aware flag
if [ "$#" -eq 5 ] && [ "$5" == "--kv-aware" ]; then
  KV_AWARE=true
fi

# Calculate total GPUs needed
TOTAL_GPUS=$((NUM_WORKERS * GPUS_PER_WORKER))

# Check if the world size matches the total GPUs
if [ "$WORLD_SIZE" -ne "$TOTAL_GPUS" ]; then
  echo "Error: World size must equal the total number of GPUs (num-workers * gpus-per-worker)."
  exit 1
fi

# Set environment variables
export DYN_LOG=DEBUG

# Start the HTTP server with debug logging
echo "Starting HTTP server..."
http &

# Wait for the server to start
sleep 5

# Add models to the server
echo "Adding models to the server..."
llmctl http remove chat $MODEL_NAME
llmctl http remove completion $MODEL_NAME

if [ "$KV_AWARE" = true ]; then
  component_str="router"
else
  component_str="tensorrt-llm"
fi
llmctl http add chat $MODEL_NAME dynamo.$component_str.chat/completions
llmctl http add completion $MODEL_NAME dynamo.$component_str.completions

llmctl http list

# Launch router
if [ "$KV_AWARE" = true ]; then
  echo "Launching router..."
  python3 -m monolith.router --min-workers $NUM_WORKERS --engine_args llm_api_config.yaml 1>router.log 2>&1 &
fi

# Launch workers
for ((i=0; i<NUM_WORKERS; i++)); do
  START_GPU=$((i * GPUS_PER_WORKER))
  END_GPU=$((START_GPU + GPUS_PER_WORKER - 1))
  GPU_DEVICES=$(seq -s, $START_GPU $END_GPU)

  echo "Launching worker $((i+1)) on GPUs $GPU_DEVICES..."

  if [ "$KV_AWARE" = true ]; then
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES mpirun --allow-run-as-root -n 1 --oversubscribe python3 -m monolith.worker --publish-stats --publish-kv-cache-events --engine_args llm_api_config.yaml 1>worker$((i+1)).log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES mpirun --allow-run-as-root -n 1 --oversubscribe python3 -m monolith.worker --engine_args llm_api_config.yaml 1>worker$((i+1)).log 2>&1 &
  fi
done

# Wait for the workers to start
sleep 5

echo "Monolithic deployment initialized with $NUM_WORKERS workers"
