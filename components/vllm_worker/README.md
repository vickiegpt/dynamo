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

# Component

## VLLM Worker

## Description

The vllm worker integrates the vllm MQLLMEngine into dynamo and
extends its capabilities with support for:

- Disaggregated Serving
- KV Aware Routing

## Details
- **Package Name**: `dynamo.components.vllm_worker`
- **Version**: `0.1.0`
- **Dependencies**: `ai-dynamo`
- **Input/Output**: Open AI chat/completions

## System Diagram

This figure shows an overview of the major components for deploying a
vllm worker in a dynamo graph with all features.

```
                                                 +----------------+
                                          +------| prefill worker |-------+
                                   notify |      |   (optional)   |       |
                                 finished |      +----------------+       | pull
                                          v                               v
+------+      +-----------+      +------------------+    push     +---------------+
| HTTP |----->| processor |----->| decode/monolith  |------------>| prefill queue |
|      |<-----|           |<-----|      worker      | (if disagg) |   (optional)  |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|    (optional)    |
                                 +------------------+
```

## Usage Examples

### Pre-Requisites

The following examples require that you are running within a dynamo
development container.

#### Building Environment

```
./container/build.sh
```

#### Starting Interactive Shell

```
./container/run.sh -it
```

### Example 1: Standalone Worker

#### System Diagram

```
  +------+      +------------------+
  | HTTP |----->| Monolithic Worker|
  |      |<-----|                  |
  +------+      +------------------+

```

#### Command

```
dynamo-llm --model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
```

#### Request

In a seperate terminal use curl to exercise the endpoint:

```
curl localhost:8181/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```


### Example 2: KV Cache Aware Router + Worker

#### System Diagram

```
+------+      +-----------+      +------------------+
| HTTP |----->| processor |----->| monolith         |
|      |<-----|           |<-----|      worker      |
+------+      +-----------+      +------------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|                  |
                                 +------------------+
```

#### Command

```
dynamo-llm --model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' --router kv
```

#### Request

In a seperate terminal use curl to exercise the endpoint:

```
curl localhost:8181/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```
