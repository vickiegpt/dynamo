<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Running DeepSeek-R1 Disaggregated with WideEP on GB200s

Dynamo supports SGLang's GB200 implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://lmsys.org/blog/2025-06-16-gb200-part-1/) for more details. Full end to end optimization is still a work in progress but you can get this up and running with the following steps. In ths example, we will run 1 prefill worker on 2 GB200 nodes (4 GPUs each) and 1 decode worker on 12 GB200 nodes (total 56 GPUs).

## Instructions


1. Build the Dynamo container

```bash
cd $DYNAMO_ROOT
docker build \
  -f container/Dockerfile.sglang-wideep \
  -t dynamo-wideep-gb200 \
  --build-arg MODE=blackwell \
  --build-arg SGLANG_IMAGE_TAG=v0.4.9.post6-cu128-gb200 \
  --build-arg ARCH=arm64 \
  --build-arg ARCH_ALT=aarch64 \
  . \
  --no-cache
```

2. You can run this container on each 4xGB200 node using the following command.

> [!IMPORTANT]
> We recommend downloading DeepSeek-R1 and then mounting it to the container. You can find the model [here](https://huggingface.co/deepseek-ai/DeepSeek-R1)

```bash
docker run \
    --gpus all \
    -it \
    --rm \
    --network host \
    --volume /PATH_TO_DSR1_MODEL/:/model/ \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --cap-add CAP_SYS_PTRACE \
    --ipc host \
    dynamo-wideep-gb200:latest
```

4. On the head prefill node, run the helper script provided to generate commands to start the `nats-server`, `etcd`. This script will also tell you which environment variables to export on each node to make deployment easier.

```bash
./utils/gen_env_vars.sh
```

In each container, you should be in the `/sgl-workspace/dynamo/components/backends/sglang` directory.

```bash
git clone https://github.com/ai-dynamo/dynamo.git
git checkout ishan/more-slurm-targets
cd examples/sglang/slurm_jobs
```

4. Ensure you have the proper paths that you can use to mount things to the container

- The path to the DSR1 model which should be mounted to the `--model-dir` flag and `--config-dir` flag 

5. Run the following command to submit the job

```bash
python3 submit_job_script.py \
  --template job_script_template.j2 \
  --model-dir <path-to-dsr1-model> \
  --container-image <image-from-step-2> \
  --account <your-account> \
  --gpus-per-node 4 \
  --config-dir <path-to-dsr1-model> \
  --network-interface enp138s0f0np0 \
  --gpu-type gb200 \
  --use-sglang-commands \
  --prefill-nodes 2 \
  --decode-nodes 12
```

**Note**: if you want to spin up dynamo, you can remove the `--use-sglang-commands` flag.

6. This will create a logs directory in the `examples/sglang/slurm_jobs` directory. You can `cd` into the directory, cd into your job id, and then run `tail -f *_prefill.err *_decode.err` or `tail -f *_prefill.out *_decode.out` to see the logs.