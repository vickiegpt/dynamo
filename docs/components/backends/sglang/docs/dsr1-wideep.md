<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running DeepSeek-R1 Disaggregated with WideEP

Dynamo supports SGLang's implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://www.nvidia.com/en-us/technologies/ai/deepseek-r1-large-scale-p-d-with-wide-expert-parallelism/) for more details.

We provide configurations to deploy this at scale on different GPU types:

## GPU-Specific Deployment Guides

Choose the deployment guide that matches your hardware:

- **[H100 Deployment](dsr1-wideep-h100.md)** - Run 1 prefill worker on 4 H100 nodes and 1 decode worker on 9 H100 nodes (104 total GPUs)
- **[GB200 Deployment](dsr1-wideep-gb200.md)** - Run DeepSeek-R1 with WideEP on GB200 systems

## Overview

Both deployment patterns use:
- **Wide Expert Parallelism (WideEP)** for efficient MoE model serving
- **Disaggregated Serving** with separate prefill and decode workers
- **Large Scale P/D** for high-performance inference at scale

## Prerequisites

- SGLang container with WideEP support (see individual guides for build instructions)
- Multi-node cluster with appropriate GPU interconnects
- Dynamo platform deployed and configured

## Getting Started

1. Choose your target hardware (H100 or GB200)
2. Follow the specific deployment guide for your setup
3. Refer to the [SLURM deployment example](../slurm_jobs/README.md) for cluster deployment
