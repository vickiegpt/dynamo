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

import torch

from dynamo.llm import BlockManager

# pytestmark = pytest.mark.pre_merge


NUM_BLOCKS = 7
NUM_LAYERS = 5
PAGE_SIZE = 4
INNER_DIM = 13
ALIGNMENT = 1
DTYPE = "FP32"


async def test_initialization():
    # Test with default device
    BlockManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="CPU"
    )
    # Test with lowercase device
    BlockManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="cpu"
    )
    # Test with CUDA device
    BlockManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="CUDA"
    )
    # Test with lowercase cuda device
    BlockManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="cuda"
    )
    # Test with specific CUDA device
    BlockManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="CUDA:0"
    )
    # Test with lowercase specific cuda device
    BlockManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="cuda:0"
    )
    # Test with pinned memory
    # BlockManager(
    #    NUM_BLOCKS,
    #    NUM_LAYERS,
    #    PAGE_SIZE,
    #    INNER_DIM,
    #    ALIGNMENT,
    #    DTYPE,
    #    device="CUDA",
    #    pin_memory=True,
    # )


async def test_cpu_tensor():
    block_manager = BlockManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="CPU"
    )
    tensor = torch.from_dlpack(block_manager.py_capsule(0, 0))
    assert tensor.get_device() == -1  # CPU
    assert tensor.shape == (PAGE_SIZE, INNER_DIM)
    assert tensor.dtype == torch.float32  # DTYPE
    # print(tensor)
    tensor[0][0] = 1.0
    tensor[PAGE_SIZE - 1][INNER_DIM - 1] = 1.0
    # print(tensor)
    tensor_ = torch.from_dlpack(block_manager.py_capsule(0, 0))
    assert tensor is not tensor_
    assert tensor.shape == tensor_.shape
    assert tensor.dtype == tensor_.dtype
    assert torch.allclose(tensor, tensor_)


async def test_cuda_tensor():
    block_manager = BlockManager(
        NUM_BLOCKS, NUM_LAYERS, PAGE_SIZE, INNER_DIM, ALIGNMENT, DTYPE, device="CUDA"
    )
    tensor = torch.from_dlpack(block_manager.py_capsule(0, 0))
    assert tensor.get_device() == 0  # CUDA:0
    assert tensor.shape == (PAGE_SIZE, INNER_DIM)
    assert tensor.dtype == torch.float32  # DTYPE
    # print(tensor)
    tensor[0][0] = 1.0
    tensor[PAGE_SIZE - 1][INNER_DIM - 1] = 1.0
    # print(tensor)
    tensor_ = torch.from_dlpack(block_manager.py_capsule(0, 0))
    assert tensor is not tensor_
    assert tensor.shape == tensor_.shape
    assert tensor.dtype == tensor_.dtype
    assert torch.allclose(tensor, tensor_)


async def main():
    await test_initialization()
    await test_cpu_tensor()
    await test_cuda_tensor()


if __name__ == "__main__":
    asyncio.run(main())
