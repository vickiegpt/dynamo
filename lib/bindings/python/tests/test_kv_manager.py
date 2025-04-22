# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio

from dynamo.llm import KvManager

# pytestmark = pytest.mark.pre_merge


NUM_BLOCKS = 7
NUM_LAYERS = 5
PAGE_SIZE = 4
INNER_DIM = 13
ALIGNMENT = 1
DTYPE = "FP32"


async def test_initialization():
    # Test with default device
    KvManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="CPU"
    )
    # Test with lowercase device
    KvManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="cpu"
    )
    # Test with CUDA device
    KvManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="CUDA"
    )
    # Test with lowercase cuda device
    KvManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="cuda"
    )
    # Test with specific CUDA device
    KvManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="CUDA:0"
    )
    # Test with lowercase specific cuda device
    KvManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="cuda:0"
    )
    # Test with pinned memory
    # KvManager(
    #    NUM_BLOCKS,
    #    NUM_LAYERS,
    #    PAGE_SIZE,
    #    INNER_DIM,
    #    ALIGNMENT,
    #    DTYPE,
    #    device="CUDA",
    #    pin_memory=True,
    # )


if __name__ == "__main__":
    asyncio.run(test_initialization())
