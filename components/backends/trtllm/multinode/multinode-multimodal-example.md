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

# Example: Multi-node TRTLLM Workers with Dynamo on Slurm

To run a single Dynamo+TRTLLM Worker that spans multiple nodes (ex: TP8),
the set of nodes need to be launched together in the same MPI world, such as
via `mpirun` or `srun`. This is true regardless of whether the worker is
aggregated, prefill-only, or decode-only.

In this document we will demonstrate two examples launching multinode workers
on a slurm cluster with `srun`:
1. Deploying a disaggregated `meta-llama/Llama-4-Maverick-17B-128E-Instruct` model with a multi-node
   TP8 (2 nodes) and a multi-node TP8/EP16 decode
   worker (2 nodes) across a total of 4 GB200 nodes.

NOTE: Some of the scripts used in this example like `start_frontend_services.sh` and
`start_trtllm_worker.sh` should be translatable to other environments like Kubernetes, or
using `mpirun` directly, with relative ease.

## Setup

For simplicity of the example, we will make some assumptions about your slurm cluster:
1. First, we assume you have access to a slurm cluster with multiple GPU nodes
   available. For functional testing, most setups should be fine. For performance
   testing, you should aim to allocate groups of nodes that are performantly
   inter-connected, such as those in an NVL72 setup.
2. Second, we assume this slurm cluster has the [Pyxis](https://github.com/NVIDIA/pyxis)
   SPANK plugin setup. In particular, the `srun_aggregated.sh` script in this
   example will use `srun` arguments like `--container-image`,
   `--container-mounts`, and `--container-env` that are added to `srun` by Pyxis.
   If your cluster supports similar container based plugins, you may be able to
   modify the script to use that instead.
3. Third, we assume you have already built a recent Dynamo+TRTLLM container image as
   described [here](https://github.com/ai-dynamo/dynamo/tree/main/components/backends/trtllm#build-docker).
   This is the image that can be set to the `IMAGE` environment variable in later steps.
4. Fourth, we assume you pre-allocate a group of nodes using `salloc`. We
   will allocate 8 nodes below as a reference command to have enough capacity
   to run both examples. If you plan to only run the aggregated example, you
   will only need 4 nodes. If you customize the configurations to require a
   different number of nodes, you can adjust the number of allocated nodes
   accordingly. Pre-allocating nodes is technically not a requirement,
   but it makes iterations of testing/experimenting easier.

   Make sure to set your `PARTITION` and `ACCOUNT` according to your slurm cluster setup:
    ```bash
    # Set partition manually based on your slurm cluster's partition names
    PARTITION=""
    # Set account manually if this command doesn't work on your cluster
    ACCOUNT="$(sacctmgr -nP show assoc where user=$(whoami) format=account)"
    salloc \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --job-name="${ACCOUNT}-dynamo.trtllm" \
      -t 05:00:00 \
      --nodes 8
    ```
5. Lastly, we will assume you are inside an interactive shell on one of your allocated
   nodes, which may be the default behavior after executing the `salloc` command above
   depending on the cluster setup. If not, then you should SSH into one of the allocated nodes.

### Environment Variable Setup

This example aims to automate as much of the environment setup as possible,
but all slurm clusters and environments are different, and you may need to
dive into the scripts to make modifications based on your specific environment.

Assuming you have already allocated your nodes via `salloc`, and are
inside an interactive shell on one of the allocated nodes, set the
following environment variables based:
```bash
# NOTE: IMAGE must be set manually for now
# To build an iamge, see the steps here:
# https://github.com/ai-dynamo/dynamo/tree/main/components/backends/trtllm#build-docker
export IMAGE="<dynamo_trtllm_image>"

# MOUNTS are the host:container path pairs that are mounted into the containers
# launched by each `srun` command.
#
# If you want to reference files, such as $MODEL_PATH below, in a
# different location, you can customize MOUNTS or specify additional
# comma-separated mount pairs here.
#
# NOTE: Currently, this example assumes that the local bash scripts and configs
# referenced are mounted into into /mnt inside the container. If you want to
# customize the location of the scripts, make sure to modify `srun_aggregated.sh`
# accordingly for the new locations of `start_frontend_services.sh` and
# `start_trtllm_worker.sh`.
#
# For example, assuming your cluster had a `/lustre` directory on the host, you
# could add that as a mount like so:
#
# export MOUNTS="${PWD}/../:/mnt,/lustre:/lustre"
export MOUNTS="${PWD}/../:/mnt"

# Can point to local FS as weel
# export MODEL_PATH="/location/to/model"
export MODEL_PATH="meta-llama/Llama-4-Maverick-17B-128E-Instruct"

# The name the model will be served/queried under, matching what's
# returned by the /v1/models endpoint.
#
# By default this is inferred from MODEL_PATH, but when using locally downloaded
# model weights, it can be nice to have explicit control over the name.
export SERVED_MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
```

## Disaggregated Mode

Assuming you have at least 4 4xGB200 nodes allocated (2 for prefill, 2 for decode)
following the setup above, follow these steps below to launch a **disaggregated**
deployment across 4 nodes:

> [!Tip]
> Make sure you have a fresh environment and don't still have the aggregated
> example above still deployed on the same set of nodes.

```bash
# Defaults set in srun_disaggregated.sh, but can customize here.
# export PREFILL_ENGINE_CONFIG="/mnt/engine_configs/multimodal/llama4/prefill.yaml"
# export DECODE_ENGINE_CONFIG="/mnt/engine_configs/multimodal/llama4/decode.yaml"

# Customize NUM_PREFILL_NODES to match the desired parallelism in PREFILL_ENGINE_CONFIG
# Customize NUM_DECODE_NODES to match the desired parallelism in DECODE_ENGINE_CONFIG
# The products of NUM_PREFILL_NODES*NUM_GPUS_PER_NODE and
# NUM_DECODE_NODES*NUM_GPUS_PER_NODE should match the respective number of
# GPUs necessary to satisfy the requested parallelism in each config.
# export NUM_PREFILL_NODES=2
# export NUM_DECODE_NODES=2

# GB200 nodes have 4 gpus per node, but for other types of nodes you can configure this.
# export NUM_GPUS_PER_NODE=4

# Launches:
# - frontend + etcd/nats on current (head) node.
# - one large prefill trtllm worker across multiple nodes via MPI tasks
# - one large decode trtllm worker across multiple nodes via MPI tasks
./srun_disaggregated.sh
```

## Understanding the Output

1. The `srun_disaggregated.sh` launches three srun jobs instead of two. One for frontend, one for prefill   worker, and one for decode worker.

2. The OpenAI frontend will listen for and dynamically discover workers as
   they register themselves with Dynamo's distributed runtime:
   ```
   0: 2025-06-13T02:36:48.160Z  INFO dynamo_run::input::http: Watching for remote model at models
   0: 2025-06-13T02:36:48.161Z  INFO dynamo_llm::http::service::service_v2: Starting HTTP service on: 0.0.0.0:8000 address="0.0.0.0:8000"
   ```
3. The TRTLLM worker will consist of N (N=8 for TP8) MPI ranks, 1 rank on each
   GPU on each node, which will each output their progress while loading the model.
   You can see each rank's output prefixed with the rank at the start of each log line
   until the model succesfully finishes loading:
    ```
     7: rank7 run mgmn worker node with mpi_world_size: 8 ...
    ```
4. After the model fully finishes loading on all ranks, the worker will register itself,
   and the OpenAI frontend will detect it, signaled by this output:
    ```
    0: 2025-06-13T02:46:35.040Z  INFO dynamo_llm::discovery::watcher: added model model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    ```
5. At this point, with the worker fully initialized and detected by the frontend,
   it is now ready for inference.

## Example Request

To verify the deployed model is working, send a `curl` request:
```bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
                    }
                }
            ]
        }
    ],
    "stream": false,
    "max_tokens": 160
}'
```

## Cleanup

To cleanup background `srun` processes launched by `srun_aggregated.sh` or
`srun_disaggregated.sh`, you can run:
```bash
pkill srun
```

## Known Issues

- Loading `meta-llama/Llama-4-Maverick-17B-128E-Instruct` with 8 nodes of H100 with TP=16 is not posssible due to Llama4 Maverick has a config `"num_attention_heads": 40` , trtllm engine asserts on assert `self.num_heads % tp_size == 0`  causing the engine to crash on model loading.