
# Command Cheat Sheet

## Setup (Already Configured on Brev Instances)

```
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
git checkout nnshah1/0.2.0-msbuild
```

```
docker compose -f deploy/docker-compose.yml up --detach
```

```
docker build -f tutorials/Dockerfile . -t dynamo:msbuild-dev
```

```
./container/run.sh --mount-workspace -it --image dynamo:dev --name dynamo-dev
```

```
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## Running Dynamo Run

```
dynamo run out=vllm deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## Running Dynamo Serve (Aggregate)

```
cd examples/llm
dynamo serve graphs.agg:Frontend -f configs/agg_tp_1_dp_4.yaml
```

## Single Gen ai Perf Command

```
  # NOTE: For Dynamo HTTP OpenAI frontend, use `nvext` for fields like
  # `ignore_eos` since they are not in the official OpenAI spec.
  genai-perf profile \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --goodput output_token_throughput_per_user:40 \
    --streaming \
    --url http://localhost:8000 \
    --synthetic-input-tokens-mean 3000 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 150 \
    --output-tokens-stddev 0 \
    --extra-inputs max_tokens:150 \
    --extra-inputs min_tokens:150 \
    --extra-inputs ignore_eos:true \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --concurrency 16 \
    --request-count 160 \
    --warmup-request-count 32 \
    --num-dataset-entries 192 \
    --random-seed 100 \
    --artifact-dir ${ARTIFACT_DIR:=artifacts} \
    -- \
    -v \
    --max-threads 256 \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'
```

## Benchmark

```
cd examples/llm/benchmark/perf.sh
```

## Disaggregated Serving

```
cd examples/llm
dynamo serve graphs.disagg:Frontend -f configs/disagg_p_tp_2_dp_1_d_tp_2_dp_1.yaml
```

