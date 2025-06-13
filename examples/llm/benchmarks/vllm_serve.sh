python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8888 \
  --block-size 64 \
  --max-model-len 16384 \
  --enforce-eager \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \