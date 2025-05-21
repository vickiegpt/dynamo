
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
dynamo run out=vllm deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```


```
cd examples/llm
dynamo serve graphs.agg:Frontend -f configs/agg.yaml
```

```
docker exec -it dynamo-dev /bin/bash
cd examples/llm/benchmark/perf.sh
```

```
cd examples/llm
dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml
```

