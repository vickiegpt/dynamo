#!/bin/bash
# ASSUMPTION: dynamo and its dependencies are properly installed
# i.e. nats and etcd are running

# Overview:
# This script deploys dynamo disaggregated serving with LMCache enabled on port 8080
# Used for LMCache correctness testing
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Arguments:
MODEL_URL=$1

if [ -z "$MODEL_URL" ]; then
    echo "Usage: $0 <MODEL_URL>"
    echo "Example: $0 Qwen/Qwen3-0.6B"
    exit 1
fi

echo "🚀 Starting dynamo disaggregated serving setup with LMCache:"
echo "   Model: $MODEL_URL"
echo "   Port: 8080"
echo "   Mode: Disaggregated (prefill + decode workers) + LMCache"
echo "   !! Remember to kill the old dynamo processes otherwise the port will be busy !!"

# Get script directory and navigate there
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Navigate to dynamo examples directory
DYNAMO_DIR="$SCRIPT_DIR/../../examples/vllm"
cd $DYNAMO_DIR

# Kill any existing dynamo processes
echo "🧹 Cleaning up any existing dynamo processes..."
pkill -f "dynamo-run" || true
sleep 2

# Enable LMCache
export ENABLE_LMCACHE=1

# Set LMCache configuration environment variables
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=20

echo "🔧 Starting dynamo disaggregated serving with LMCache enabled..."

dynamo run in=http out=dyn &

CUDA_VISIBLE_DEVICES=0 python3 components/main.py --model $MODEL_URL --enforce-eager &

CUDA_VISIBLE_DEVICES=1 python3 components/main.py \
    --model $MODEL_URL \
    --enforce-eager \
    --is-prefill-worker
