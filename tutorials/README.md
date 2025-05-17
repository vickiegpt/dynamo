```
docker compose -f deploy/docker-compose.yml up --detach
```


```
docker build -f tutorials/Dockerfile . -t dynamo:dev
```

```
./container/run.sh --mount-workspace -it --image dynamo:dev --name dynamo-dev
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

