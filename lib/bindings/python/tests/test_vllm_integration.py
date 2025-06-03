import asyncio
import hashlib
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import pytest
import torch

from dynamo.llm import BlockManager

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
    return BlockManager(
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


class Request:
    def __init__(self, request_id: str, all_token_ids: list[int], block_size: int):
        self.request_id = request_id
        self.all_token_ids = all_token_ids
        self.block_size = block_size
        self.num_computed_tokens = 0


class BlockHashType(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. By using SHA256 however,
    hash collisions are practically impossible.
    """

    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: tuple[int, ...]
    # Extra keys for the block.
    extra_keys: Optional[Any] = None


def sha256(input) -> int:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        An integer representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return int.from_bytes(hashlib.sha256(input_bytes).digest(), byteorder="big")


# The hash seed for the first block of the prefix block sequence.
#
# Even if the hash function is the builtin hash(), we use sha256 to generate
# the initial hash to simplify the code. This is not performance critical
# as it is done one per process.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed.
# This aligns with the behavior of Python's hash() function, which also uses
# a random seed if PYTHONHASHSEED is not set.
NONE_HASH = (
    int.from_bytes(os.urandom(32), byteorder="big")
    if os.getenv("PYTHONHASHSEED") is None
    else sha256(os.getenv("PYTHONHASHSEED"))
)


def hash_block_tokens(
    hash_function: Callable,
    parent_block_hash: Optional[int],
    curr_block_token_ids: Sequence[int],
    extra_keys: Optional[tuple[Any, ...]] = None,
) -> BlockHashType:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    Args:
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.

    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHashType(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys)),
        curr_block_token_ids_tuple,
        extra_keys,
    )


def hash_request_tokens(
    hash_function: Any, block_size: int, request: Request
) -> list[BlockHashType]:
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. The hash value is used for prefix caching.

    Args:
        block_size: The size of each block.
        request: The request object.

    Returns:
        The list of computed hash values.
    """
    token_ids = request.all_token_ids

    req_extra_keys = None

    ret = []
    parent_block_hash_value = None
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        block_hash = hash_block_tokens(
            hash_function, parent_block_hash_value, block_token_ids, req_extra_keys
        )
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


class KVCacheBlockProtocol(Protocol):
    pass


class KVBMCacheBlock(KVCacheBlockProtocol):
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int

    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    _block_hash: Optional[BlockHashType] = None


@runtime_checkable
@dataclass
class KVCacheBlocksProtocol(Protocol):
    """Protocol defining the structure and behavior of KVCacheBlocks."""

    # blocks: List[KVCacheBlock]
    """The list of KVCacheBlock objects."""

    def __add__(
        self: KVCacheBlockProtocol, other: KVCacheBlockProtocol
    ) -> KVCacheBlockProtocol:
        """Adds two KVCacheBlocks instances."""
        pass

    @classmethod
    def create_empty(cls: type[KVCacheBlockProtocol]) -> KVCacheBlockProtocol:
        """Creates a new KVCacheBlocks instance with no blocks."""
        pass

    def get_block_ids(self) -> List[List[int]]:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            list[list[int]]: A two-level list where
            * the outer list corresponds to KV cache groups (only 1 group now)
            * each inner list contains the block_ids of the blocks in that group
        """
        pass

    def get_unhashed_block_ids(self) -> List[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        pass


class KVBMCacheBlocks(KVCacheBlocksProtocol):
    blocks: list[KVBMCacheBlock]

    @classmethod
    def create_empty(cls) -> "KVBMCacheBlocks":
        return cls([])

    def get_block_ids(self) -> list[list[int]]:
        return [[block.block_id for block in self.blocks]]

    def get_unhashed_block_ids(self) -> list[int]:
        return [block.block_id for block in self.blocks if block.block_hash is None]


@runtime_checkable
class KVCacheManagerProtocol(Protocol):
    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        pass

    def get_computed_blocks(
        self, request: Request
    ) -> tuple[KVCacheBlocksProtocol, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        pass

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocksProtocol] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocksProtocol]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        pass

    # def free(self, request: Request) -> None:
    #     """Free the blocks allocated for the request.
    #     We free the blocks in reverse order so that he tail blocks are evicted
    #     first when caching is enabled.

    #     Args:
    #         request: The request to free the blocks.
    #     """
    #     pass

    # def reset_prefix_cache(self) -> bool:
    #     """Reset prefix cache. This function may be used in RLHF
    #     flows to invalidate prefix caching after the weights are updated,
    #     or used for resetting prefix caching status for benchmarking.

    #     Returns:
    #         bool: True if the prefix cache is successfully reset,
    #         False otherwise.
    #     """
    #     pass

    # def get_num_common_prefix_blocks(
    #     self, request: Request, num_running_requests: int
    # ) -> list[int]:
    #     """Calculate the number of common prefix blocks shared by all requests
    #     in the RUNNING state for each kv cache group.

    #     The function determines this by selecting any request and iterating
    #     through its blocks.  A block is considered a common prefix block if its
    #     `ref_cnt` equals the total number of requests in the RUNNING state.

    #     NOTE(woosuk): The number of requests in the RUNNING state is **greater
    #     than or equal to** the number of requests scheduled in the current step.
    #     This is because the RUNNING state only indicates that:
    #     1. The request has not yet finished, and
    #     2. The request holds its blocks unfreed.

    #     While all scheduled requests must be in the RUNNING state, the inverse
    #     is not necessarily true. There may be RUNNING requests that are not
    #     scheduled in the current step.

    #     This can result in an edge case where the number of common prefix blocks
    #     is 0, even though all scheduled requests share a common prefix. This
    #     occurs because there may be unscheduled RUNNING requests that do not
    #     share the common prefix. Currently, this case cannot be easily detected,
    #     so the function returns 0 in such cases.

    #     Args:
    #         request: Any request in the RUNNING state, used to identify the
    #             common prefix blocks.
    #         num_running_requests: The total number of requests in the RUNNING
    #             state. This can be different from the number of scheduled
    #             requests in the current step.

    #     Returns:
    #         list[int]: The number of common prefix blocks for each kv cache
    #         group.
    #     """
    #     pass

    # def free_block_hashes(self, request: Request) -> None:
    #     """Discard the block hashes for the request.

    #     NOTE: Unlike `free`, this method should be called only when the request
    #     is finished, not when it is preempted.
    #     """
    #     pass

    # def take_events(self) -> list[KVCacheEvent]:
    #     """Take the KV cache events from the block pool.

    #     Returns:
    #         A list of KV cache events.
    #     """
    #     pass

    # def get_block_ids(self, request_id: str) -> list[list[int]]:
    #     """Get the block ids of a request."""
    #     pass

    # def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
    #     """Get (and reset) the prefix cache stats.

    #     Returns:
    #         The current prefix caching stats, or None if logging is disabled.
    #     """
    #     pass


class KVBMCacheManager(KVCacheManagerProtocol):
    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
        self.caching_hash_fn = sha256
        self.block_size = 32
        self.enable_caching = True
        self.num_kv_cache_groups = 1

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[str, list[BlockHashType]] = defaultdict(
            list
        )

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVBMCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for reempted ones.
        self.num_cached_block: dict[str, int] = {}

    def get_computed_blocks(self, request: Request) -> tuple[KVBMCacheBlocks, int]:
        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = hash_request_tokens(
                self.caching_hash_fn, self.block_size, request
            )
            self.req_to_block_hashes[request.request_id] = block_hashes

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        # max_cache_hit_length = request.num_tokens - 1

        # TODO: py binding to call self.block_manager to do RegisterBlocks and MatchSequenceHashes are missing
        # computed_blocks = self.single_type_manager.find_longest_cache_hit(
        #     block_hashes, max_cache_hit_length)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: list[KVBMCacheBlock],
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.

        Returns:
            The number of blocks.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = (
            num_required_blocks
            - len(new_computed_blocks)
            - len(self.req_to_blocks[request_id])
        )
        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it will be changed from a free block
        # to a computed block when the request is allocated, so we also count
        # it as needed to be allocated.
        num_evictable_computed_blocks = sum(
            blk.ref_cnt == 0 for blk in new_computed_blocks
        )
        return (
            num_new_blocks + num_evictable_computed_blocks
        ) * self.num_kv_cache_groups

    def save_new_computed_blocks(
        self, request_id: str, new_computed_blocks: list[KVBMCacheBlock]
    ) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
        """
        if request_id not in self.num_cached_block:
            # A new request.
            req_blocks = self.req_to_blocks[request_id]
            assert len(req_blocks) == 0
            req_blocks.extend(new_computed_blocks)
            self.num_cached_block[request_id] = len(new_computed_blocks)
        else:
            # A running request. Should not have new computed blocks.
            assert len(new_computed_blocks) == 0

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVBMCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVBMCacheBlocks]:
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = []

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )
        num_blocks_to_allocate = self.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
        )

        # TODO: need py binding to get num of inactive blocks
        if num_blocks_to_allocate > self.block_manager.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # TODO: we don't need this on kvbm side, do we?
        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            # self.block_pool.touch(new_computed_block_list)
            pass
        else:
            assert not new_computed_block_list, (
                "Computed blocks should be empty when " "prefix caching is disabled"
            )

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.save_new_computed_blocks(request.request_id, new_computed_block_list)

        # below is code logic from single_type_manager.allocate_new_blocks to make API compatible
        req_blocks = self.req_to_blocks[request.request_id]
        num_required_blocks = cdiv(num_tokens_need_slot, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        block_list = []
        if num_new_blocks > 0:
            # TODO: allocate_device_blocks_blocking should not allocate tensors since vllm already allocated tensors during GPUModelRunner.initialize_kv_cache
            block_list = self.block_manager.allocate_device_blocks_blocking(
                num_new_blocks * self.num_kv_cache_groups
            )
            req_blocks.extend(block_list)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_kvbm_cache_manager_allocate_slots(block_manager: BlockManager):
    kvbm = KVBMCacheManager(block_manager)
    request = Request(
        request_id="test_request",
        all_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        block_size=4,
        num_computed_tokens=3,
    )
    blocks = kvbm.allocate_slots(request)
    assert len(blocks) == 1


async def main():
    await test_kvbm_cache_manager_allocate_slots(new_block_manager())


if __name__ == "__main__":
    asyncio.run(main())
