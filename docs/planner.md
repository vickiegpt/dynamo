<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Planner

The planner is a component that monitors the state of the system and makes adjustments to workers to ensure that the system is running efficiently. Currently, planner can scale up and down the number of vllm workers based on the kv cache load and prefill queue size:
* Backend:
  * local ✅
  * kubernetes ✅
* LLM framework:
  * vllm ✅
  * tensorrt-llm ❌
  * SGLang ❌
  * llama.cpp ❌
* Serving type:
  * Aggregated ✅
  * Disaggregated ✅
* Planner actions:
  * Load-based scaling up/down prefill/decode workers ✅
  * SLA-based scaling up/down prefill/decode workers ❌
  * Adjusting engine knobs ❌

## Load-based Scaling Up/Down Prefill/Decode Workers

To adjust the number of prefill/decode workers, planner monitors the following metrics:
* Prefill worker: planner monitors the number of requests pending in the prefill queue to estimate the prefill workload.
* Decode/aggregated worker: planner monitors the average KV cache utilization rate to estimate the decode/aggregated workload.

Every `metric-pulling-interval`, planner will gather the aforementioned metrics. Every `adjustment-interval`, planner compares the aggregated metrics in this interval with pre-set thresholds and decide to scale up/down prefill/decode workers. To avoid over-compensation, planner only changes the number of workers by 1 in one adjustment interval. In addition, when the number of workers is being adjusted, the planner will block the metric pulling and adjustment.

To scale up a prefill/decode worker, planner just need to launch the worker in the correct namespace. The auto-discovery mechanism will pick up the workers and add them to the routers. To scale down a prefill worker, planner send a SIGTERM signal to the prefill worker. The prefill worker store the signal and exit when it finishes the current request pulled from the prefill queue. This ensures that no remote prefill request is dropped. To scale down a decode worker, currently, planner revoke the etcd lease of the decode worker. When the etcd lease is revoked, the corresponding decode worker will be immediately removed from the router and will not get any new requests. The decode worker will then finish all the current requests in their original stream and exit gracefully.

## Usage
After you've deployed a dynamo graph - you can start the planner with the following command:
```bash
python components/planner.py <arguments>
```

Planner takes the following arguments:
* `--namespace` (str, default: "dynamo"): Namespace planner will look at
* `--served-model-name` (str, default: "vllm"): Model name that is being served`
* `--no-operation` (flag): Do not make any adjustments, just observe the metrics and log to tensorboard
* `--log-dir` (str, default: None): Tensorboard logging directory
* `--adjustment-interval` (int, default: 30): Interval in seconds between scaling adjustments
* `--metric-pulling-interval` (int, default: 1): Interval in seconds between metric pulls
* `--max-gpu-budget` (int, default: 8): Maximum number of GPUs to use, planner will not scale up more than this number of GPUs for prefill plus decode workers
* `--min-gpu-budget` (int, default: 1): Minimum number of GPUs to use, planner will not scale down below this number of GPUs for prefill or decode workers
* `--decode-kv-scale-up-threshold` (float, default: 0.9): KV cache utilization threshold to scale up decode workers
* `--decode-kv-scale-down-threshold` (float, default: 0.5): KV cache utilization threshold to scale down decode workers
* `--prefill-queue-scale-up-threshold` (float, default: 0.5): Queue utilization threshold to scale up prefill workers
* `--prefill-queue-scale-down-threshold` (float, default: 0.2): Queue utilization threshold to scale down prefill workers
* `--decode-engine-num-gpu` (int, default: 1): Number of GPUs per decode engine
* `--prefill-engine-num-gpu` (int, default: 1): Number of GPUs per prefill engine

### Tensorboard

Planner logs to tensorboard to visualize the metrics and the scaling actions. You can start tensorboard with the following command:
```bash
tensorboard --logdir=<path-to-tensorboard-log-dir>
```

## Backends
We currently support two backends:
1. `local` - uses circus to start/stop worker subprocesses
2. `kuberentes` - uses the kuberentes api to adjust replicas of each component's resource definition

### Local Backend

Circus is a Python program which can be used to monitor and control processes and sockets. Dynamo serve uses circus to start each node in a graph and monitors each subprocesses. We leverage a core feature to do this called `Watcher`. A `Watcher` is the target program that you would like to run (which in our case is `serve_dynamo.py`). When planner decides to scale up or down, it will either add or remove a watcher from the existing `circus`.

> [!NOTE]
> Although circus allows you to `increment` an existing watcher, it was not designed to allow variables to be passed in which does not allow us to schedule on a GPU. So instead we start a new watcher per process. When planner decdies to add or remove a worker, we have logic to handle this adding/removing and incrementing/decrementing the workers.

#### Statefile

Our statefile looks like the JSON blob above. This state is created when you initially run `dynamo serve` and is filled in with custom leases in `serve_dynamo`. Each worker is called `{namespace}_{component_name}` when it is initially created. The `resources` come from the allocator and allows us to keep track of which GPUs are available. This statefile is read in by the LocalConnector and after each planner update we make the relevant change to the statefile. For now this statefile is locally saved in `~/.dynamo/state/{namespace}.json` (or in `DYN_LOCAL_STATE_DIR `) and is automatically cleaned up when the arbiter dies. It is helpful to debug but is not meant to be used/edited by the user.

Lets use the following example to motivate this section. Say I've spun up 1 Decode worker to start. Remember, when you first run `dynamo serve`, each worker is saved as `{namespace}_{component_name}`. Our current statefile looks like

```json
{
  "dynamo_VllmWorker": {..., resources={...}},
}
```

Lets say I `add` a worker. My statefile now looks like

```json
{
  "dynamo_VllmWorker": {..., resources={...}},
  "dynamo_VllmWorker_1": {..., resources={...}},
}
```

Lets say I now `remove`. My statefile looks will look like the following because we remove the max suffix

```json
{
  "dynamo_VllmWorker": {..., resources={...}},
}
```

If we remove again (think about this as "scale to 0"), our statefile will look like

```json
{
  "dynamo_VllmWorker": {...},
}
```

Note that we keep the initial non-suffix entry in order to know what cmd we will need to spin up another worker. This is the same for prefill workers as well.

> [!NOTE]
> At the moment - planner work best if your initial replicas per worker are 1. This is because if you specify replicas > 1 when you initially start `dynamo serve`, the current implementation in `serving.py` starts each process in the same watcher. We need to refactor this so that we have a watcher per process


# Kubernetes Backend

TODO
