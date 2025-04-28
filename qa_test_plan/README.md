# QA Test Plan

## Installation

### Pip install

#### Build

Build:

```
docker run -it --name dynamo_pip_install -v$PWD/container/deps:/deps ubuntu:24.04 bash -c "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0 && python3 -m venv /venv && /venv/bin/pip install ai-dynamo[all] && /venv/bin/pip install -r /deps/requirements.test.txt " && docker commit dynamo_pip_install dynamo:latest-pip && docker rm dynamo_pip_install
```

Run:

```
./container/run.sh --image dynamo:latest-pip --mount-workspace -it -- bash -c "source /venv/bin/activate && bash"
```

#### Run

### Docker

#### vLLM

Build:

```
./container/build.sh
```

Run:

```
./container/run.sh --mount-workspace -it
```

#### TRT-LLM

Build:

```
./container/build_trtllm_base_image.sh
./container/build.sh --framework tensorrtllm
```

Run:

```
./container/run.sh --mount-workspace -it --framework tensorrtllm
```

## Local Serving

### Dynamo Run

After running container

```
cd qa_test_plan
pytest -s -v
```

### Dynamo Serve

##### vllm

Follow instructions:

[vllm](../examples/llm/README.md)

##### tensorrtllm

### Benchmarks

##### vllm

##### planner

## Dynamo Deploy

### Helm chart

### Deploy Operator

