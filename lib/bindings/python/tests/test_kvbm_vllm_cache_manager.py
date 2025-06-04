import asyncio

import pytest
import torch

from dynamo.llm import BlockManager, DynamoVllmKvCacheManager, KvRequest

pytestmark = pytest.mark.pre_merge


WORKER_ID = 0
NUM_LAYER = 5
OUTER_DIM = 2
PAGE_SIZE = 4
INNER_DIM = 13
DTYPE, TORCH_DTYPE = "FP32", torch.float32
HOST_NUM_BLOCKS = 16
DEVICE_NUM_BLOCKS = 16
DEVICE_ID = 0


def new_block_manager():
    return DynamoVllmKvCacheManager(
        BlockManager(
            WORKER_ID,
            NUM_LAYER,
            OUTER_DIM,
            PAGE_SIZE,
            INNER_DIM,
            DTYPE,
            HOST_NUM_BLOCKS,
            DEVICE_NUM_BLOCKS,
            DEVICE_ID,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_kvbm_cache_manager_allocate_slots(
    block_manager: DynamoVllmKvCacheManager,
):
    request = KvRequest(
        request_id=1,
        tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        block_size=4,
    )
    blocks = block_manager.get_computed_blocks(request)
    assert len(blocks) == 0


async def main():
    await test_kvbm_cache_manager_allocate_slots(new_block_manager())


if __name__ == "__main__":
    asyncio.run(main())
