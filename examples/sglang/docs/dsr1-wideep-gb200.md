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

Dynamo supports SGLang's GB200 implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://lmsys.org/blog/2025-06-16-gb200-part-1/) for more details. Full end to end optimization is still a work in progress but you can get this up and running with the following steps.

## Instructions

1. Build the SGLang DeepEP container on an ARM64 machine.

```bash
git clone https://github.com/kyleliang-nv/sglang.git # temporary
cd sglang
git checkout sglang_gb200_wideep_docker # temporary
docker build -f docker/Dockerfile -t sgl-blackwell-wideep --build-arg BUILD_TYPE=blackwell --build-arg CUDA_VERSION=12.8.1 .
```

2. Build the Dynamo container

```bash
cd $DYNAMO_ROOT
git checkout ishan/more-slurm-targets # temporary
docker build -f container/Dockerfile.sglang-gb200 . -t dynamo-wideep-gb200 --no-cache
```




