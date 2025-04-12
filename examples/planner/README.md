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

The planner is a component that monitors the state of the system and makes adjustments to the number of workers to ensure that the system is running efficiently.

## Usage
After you've deployed a dynamo graph - you can start the planner with the following command:
```bash
python components/planner.py --namespace <namespace>
```

## Backends
We currently support two backends:
1. `local` - uses circus to start/stop worker subprocesses
2. `kuberentes` - uses the kuberentes api to adjust replicas of each component's resource definition

# Local Backend

Circus is a Python program which can be used to monitor and control processes and sockets. Dynamo serve uses circus to start each node in a graph and monitors each subprocesses. We leverage a core feature to do this called `Watcher`. A `Watcher` is the target program that you would like to run (which in our case is `serve_dynamo.py`). When planner decides to scale up or down, it will either add or remove a watcher from the existing `circus`. You can find our interface for circus in [circusd.py](./circusd.py)

> [!NOTE]
> Although circus allows you to `increment` an existing watcher, it was not designed to allow variables to be passed in which does not allow us to schedule on a GPU. So instead we start a new watcher per process. When planner decdies to add or remove a worker, we have logic to handle this adding/removing and incrementing/decrementing the workers.

# Kubernetes Backend


