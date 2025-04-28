# QA Test Plan

## Install Paths


### Pip install

> TODO


### Docker build

#### vLLM

```
./container/build./sh
```

#### TRT-LLM

```
./container/build_trtllm_base_image.sh
./container/build.sh --framework tensorrtllm
```

