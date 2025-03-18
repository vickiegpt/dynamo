#!/usr/bin/env bash

set -e

export HF_HUB_ENABLE_HF_TRANSFER=1
uv pip install hf-transfer

rm -rf default.etcd

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../utils.sh

SERVED_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
HOST="127.0.0.1"
PORT="8000"
env | grep VLLM_

setup_cleanup_trap

nats-server -js -p 4222 -m 8222 &
etcd \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://0.0.0.0:2379 &
sleep 3

cd /workspace/examples/llm
dynamo serve graphs.agg_router:Frontend -f ./configs/agg_router.yaml &

wait_for_server "$HOST:$PORT" $SERVED_MODEL_NAME || exit 1

echo "Checking if responses are deterministic..."
first_response=""
for i in {1..10}; do
    echo "Request $i"
    response=$(curl -s -X POST http://$HOST:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'$SERVED_MODEL_NAME'",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is capital of France?"
                    }
                ],
                "seed": 1,
                "temperature": 0.0,
                "stream": false,
                "max_tokens": 100
            }')

    completion=$(echo "$response" | jq .choices[0].message.content)

    if [ $i -eq 1 ]; then
        first_response="$completion"
        if [[ "$first_response" != *"Paris"* ]]; then
            echo "=============================================================="
            echo "❌ FAILURE: Expected 'Paris' in the response but it was not found"
            echo "Got: $first_response"
            echo "=============================================================="
            exit 1
        fi
        echo "First response: $first_response"
    else
        if [ "$first_response" != "$completion" ]; then
            echo "=============================================================="
            echo "❌ FAILURE: Response $i differs from first response"
            echo "Expected: $first_response"
            echo "Got: $completion"
            echo "=============================================================="
            exit 1
        fi
    fi
done

echo "=============================================================="
echo "✅ SUCCESS: All responses are identical - deterministic behavior confirmed"
echo "=============================================================="
exit 0