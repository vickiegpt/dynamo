set -euo pipefail

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-32B"}

export HEAD_NODE_IP=${HEAD_NODE_IP:-"0.0.0.0"}  # dynamo_config.head_node_ip Make sure the head node has started etcd and NATs, as the other node will connect to them
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"


export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"agg/agg_config.yaml"}

cleanup(){ kill $(jobs -pr) 2>/dev/null || true; }
trap cleanup EXIT INT TERM

python3 utils/clear_namespace.py --namespace dynamo
python3 -m dynamo.frontend --http-port "8000" &

python3 -m dynamo.trtllm \
        --model-path "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --extra-engine-args "$AGG_ENGINE_ARGS" \
&
wait
