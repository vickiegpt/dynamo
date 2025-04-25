# Tensorrt LLM non KV-router Benchmarking 04/24/2025

This guide provides detailed steps on benchmarking Tensorrt LLM

## Hardware Prerequisites

H100 80GB x2 GPUs are required

## Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../deploy/docker-compose.yml)
```bash
docker compose -f deploy/docker-compose.yml up -d
```

Please follow the [LLM Deployment Examples using TensorRT-LLM](./README.md) for building the docker image. 

Run the built Tensorrt-LLM docker image and run bash inside the docker container. 

```bash
export IMAGE=<Tensorrt LLM Dynamo image id>
docker run --gpus all -d -it --rm --network host --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 -w /workspace --cap-add CAP_SYS_PTRACE --ipc host -v ${HOME}/.cache/huggingface:/root/.cache/huggingface --name benchmark "$IMAGE"

# if the image is not built on the current branches, copy the benchmark dir into the image
docker cp {path_to_dynamo_folder}/examples/tensorrt_llm/benchmark/ benchmark:/workspace/examples/tensorrt_llm/

# login to the benchmark docker container
docker exec -it benchmark bash

# all commands should run under this folder
cd /workspace/examples/tensorrt_llm
```

## Disaggregated benchmarking comparison between Dynamo and trtllm-serve

### Run Dynamo Tensorrt LLM Disaggregated benchmark

```bash
dynamo serve graphs.disagg:Frontend -f ./benchmark/disagg.yaml &
```

After the server started, we can kick off the perf benchmark:

```bash
bash -x ./benchmark/perf.sh
```

### Run trtllm-serve Disaggregated benchmark

Making sure the dynamo disagg process is killed
```bash
pgrep dynamo | xargs kill
```

start the context server:
```bash
CUDA_VISIBLE_DEVICES=0 trtllm-serve meta-llama/Llama-3.1-8B-Instruct --host localhost --port 8001 --backend pytorch --trust_remote_code --extra_llm_api_options ./benchmark/trtllm-serve/extra-llm-api-config-ctx.yaml &> log_ctx_0 &
```

start the generation server:
```bash
CUDA_VISIBLE_DEVICES=1 trtllm-serve meta-llama/Llama-3.1-8B-Instruct --host localhost --port 8002 --backend pytorch --trust_remote_code --extra_llm_api_options ./benchmark/trtllm-serve/extra-llm-api-config-gen.yaml &> log_gen_0 &
```

we can observe if the context server have been started:
```bash
tail -f log_ctx_0
```
```bash
tail -f log_gen_0
```

start the trtllm-serve frontend, also set the request timeout as 720s so that the server won't cancel the long running request.
```bash
trtllm-serve disaggregated -c ./benchmark/trtllm-serve/disagg_config.yaml -r 720 &
```

After the trtllm-serve frontend started, we can kick off the perf benchmark:

```bash
bash -x ./benchmark/perf.sh
```

## Aggregated benchmarking comparison between Dynamo and trtllm-serve

### Run Dynamo Tensorrt LLM Aggregated benchmark in TP2

```bash
dynamo serve graphs.agg:Frontend -f ./benchmark/agg_1_tp2.yaml &
```

After the server started, we can kick off the perf benchmark:

```bash
bash -x ./benchmark/perf.sh
```

### Run trtllm-serve Disaggregated benchmark in TP2

start the trtllm server:
```bash
CUDA_VISIBLE_DEVICES=0,1 trtllm-serve meta-llama/Llama-3.1-8B-Instruct --host localhost --port 8000 --backend pytorch --trust_remote_code --tp_size 2 --extra_llm_api_options ./benchmark/trtllm-serve/extra-llm-api-config-agg.yaml &> log_ctx_0 &
```

After the server started, we can kick off the perf benchmark:

```bash
bash -x ./benchmark/perf.sh
```

## Benchmarking comparison between Dynamo Disaggregated and Dynamo Aggregated

For Dynamo Disaggregated, please follow the [Run Dynamo Tensorrt LLM Disaggregated benchmark](#run-dynamo-tensorrt-llm-disaggregated-benchmark) in above sections. 

For Dyanmo Aggregated in TP2, please follow the [Run Dynamo Tensorrt LLM Aggregated benchmark in TP2](#run-dynamo-tensorrt-llm-aggregated-benchmark-in-tp2) in above sections.

### Run Dynamo Tensorrt LLM Aggregated benchmark in TP1 But 2 workers

```bash
dynamo serve graphs.agg:Frontend -f ./benchmark/agg_2_tp1.yaml &
```

After the server started, we can kick off the perf benchmark:

```bash
bash -x ./benchmark/perf.sh
```
