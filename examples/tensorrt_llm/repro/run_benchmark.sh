#!/bin/bash

wait_for_chat_completion_ready() {
  local url="http://localhost:9992/v1/chat/completions"
  local timeout=900   # 15 minutes
  local interval=5     # seconds between retries
  local start_time=$(date +%s)

  echo "Waiting for chat completion endpoint to return HTTP 200..."

  while true; do
    local status_code=$(curl -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" \
      -d '{
        "model": "/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4",
        "messages": [
          {
            "role": "user",
            "content": "hello! how are you?"
          }
        ],
        "stream": false,
        "max_tokens": 20
      }' "$url")

    echo "$(date): HTTP $status_code"

    if [ "$status_code" -eq 200 ]; then
      echo "Success: Received HTTP 200"
      return 0
    fi

    local elapsed=$(( $(date +%s) - start_time ))
    if [ "$elapsed" -ge "$timeout" ]; then
      echo "Timeout: 15 minutes passed without receiving HTTP 200"
      return 1
    fi

    sleep "$interval"
  done
}

ENGINE_EXTRA_CONFIG_FILE="/tmp/engine_config.yaml"

cat > "${ENGINE_EXTRA_CONFIG_FILE}" <<EOF
enable_attention_dp: true
use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_batch_sizes:
- 1
- 2
- 4
- 8
- 16
- 32
- 64
- 128
- 256
kv_cache_dtype: fp8
EOF

# start the trtllm-serve
CUDA_VISIBLE_DEVICES=0,1,2,3 trtllm-serve /lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4 \
    --host 0.0.0.0 \
    --port 9992 \
    --backend pytorch \
    --tp_size 4 \
    --ep_size 4 \
    --max_batch_size 256 \
    --max_num_tokens 8448 \
    --max_seq_len 8448 \
    --kv_cache_free_gpu_memory_fraction 0.30 \
    --extra_llm_api_options $ENGINE_EXTRA_CONFIG_FILE &

wait_for_chat_completion_ready

pip install genai-perf==0.0.13

# Start the benchmarking as a repro
model=/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4
model_folder=/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4

isl=8000
osl=256
concurrency=128

for i in {1..10}; do
    genai-perf profile \
        --model ${model} \
        --tokenizer ${model_folder} \
        --endpoint-type chat \
        --streaming \
        --url http://localhost:9992 \
        --synthetic-input-tokens-mean ${isl} \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean ${osl} \
        --output-tokens-stddev 0 \
        --extra-inputs max_tokens:${osl} \
        --extra-inputs min_tokens:${osl} \
        --extra-inputs ignore_eos:true \
        --concurrency ${concurrency} \
        --request-count $(($concurrency*10)) \
        --warmup-request-count $(($concurrency*2)) \
        --num-dataset-entries 100 \
        --random-seed 0 \
        -- \
        -v \
        --max-threads ${concurrency}
done

wait -n

pkill -f trtllm-serve
