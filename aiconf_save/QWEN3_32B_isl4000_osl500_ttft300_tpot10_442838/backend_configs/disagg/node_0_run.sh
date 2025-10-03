set -euo pipefail

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-32B"}

export HEAD_NODE_IP=${HEAD_NODE_IP:-"0.0.0.0"}  # dynamo_config.head_node_ip Make sure the head node has started etcd and NATs, as the other node will connect to them
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"



export DISAGGREGATION_STRATEGY=${DISAGGREGATION_STRATEGY:-"prefill_first"} # dynamo_config.disaggregation_strategy prefill_first or decode_first


cleanup(){ kill $(jobs -pr) 2>/dev/null || true; }
trap cleanup EXIT INT TERM

python3 utils/clear_namespace.py --namespace dynamo
python3 -m dynamo.frontend --http-port "8000" &

PREFILL_GPU=1
PREFILL_WORKERS=4
for ((w=0; w<PREFILL_WORKERS; w++)); do
  BASE=$(( w * PREFILL_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+PREFILL_GPU-1)))
  CUDA_VISIBLE_DEVICES=$GPU_LIST TRTLLM_ENABLE_XQA_JIT=0 \
    python3 -m dynamo.trtllm \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --extra-engine-args "disagg/prefill_config.yaml" \
      --disaggregation-mode prefill \
      --disaggregation-strategy "$DISAGGREGATION_STRATEGY" \
&
done

DECODE_GPU=4
DECODE_WORKERS=1
DECODE_GPU_OFFSET=4
for ((w=0; w<DECODE_WORKERS; w++)); do
  BASE=$(( DECODE_GPU_OFFSET + w * DECODE_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+DECODE_GPU-1)))
  CUDA_VISIBLE_DEVICES=$GPU_LIST TRTLLM_ENABLE_XQA_JIT=1 \
    python3 -m dynamo.trtllm \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --extra-engine-args "disagg/decode_config.yaml" \
      --disaggregation-mode decode \
      --disaggregation-strategy "$DISAGGREGATION_STRATEGY" \
&
done
wait
