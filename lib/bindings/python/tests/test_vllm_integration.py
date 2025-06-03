class KVCacheBlockProtocol(Protocol):
    pass


@runtime_checkable
class KVCacheBlocksProtocol(Protocol):
    """Protocol defining the structure and behavior of KVCacheBlocks."""

    # blocks: List[KVCacheBlock]
    """The list of KVCacheBlock objects."""

    def __add__(self: _KVCacheBlocksT, other: _KVCacheBlocksT) -> _KVCacheBlocksT:
        """Adds two KVCacheBlocks instances."""
        ...

    @classmethod
    def create_empty(cls: type[_KVCacheBlocksT]) -> _KVCacheBlocksT:
        """Creates a new KVCacheBlocks instance with no blocks."""
        ...

    def get_block_ids(self) -> List[List[int]]:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            list[list[int]]: A two-level list where
            * the outer list corresponds to KV cache groups (only 1 group now)
            * each inner list contains the block_ids of the blocks in that group
        """
        ...

    def get_unhashed_block_ids(self) -> List[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        ...


@runtime_checkable
class KVCacheManagerProtocol(Protocol):
    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        ...

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        ...

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
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
        ...

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that he tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        ...

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        ...

    def get_num_common_prefix_blocks(
        self, request: Request, num_running_requests: int
    ) -> list[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state for each kv cache group.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        ...

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        ...

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        ...

    def get_block_ids(self, request_id: str) -> list[list[int]]:
        """Get the block ids of a request."""
        ...

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        ...


# @dataclass
# class KVCacheBlock:
#     """KV-cache block metadata."""
#     # Block ID, ranging from 0 to num_gpu_blocks - 1.
#     block_id: int
#     # Reference count.
#     ref_cnt: int = 0
#     # The hash of the block composed of (block hash, tuple of token IDs).
#     # It is only available when the block is full.
#     _block_hash: Optional[BlockHashType] = None

#     # Used to construct a doubly linked list for free blocks.
#     # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
#     prev_free_block: Optional["KVCacheBlock"] = None
#     next_free_block: Optional["KVCacheBlock"] = None

#     def incr_ref(self):
#         self.ref_cnt += 1

#     def decr_ref(self):
#         self.ref_cnt -= 1

#     @property
#     def block_hash(self) -> Optional[BlockHashType]:
#         return self._block_hash

#     @block_hash.setter
#     def block_hash(self, block_hash: BlockHashType):
#         assert self.block_hash is None, (
#             "The block already has a hash. This should not happen.")
#         self._block_hash = block_hash

#     def reset_hash(self):
#         """Reset the block hash when the block is evicted."""
#         self._block_hash = None

#     def __repr__(self) -> str:
#         # Use block_id instead of KVCacheBlock object to avoid calling __repr__
#         # on KVCacheBlock object recursively.
#         prev_block_id = self.prev_free_block.block_id \
#             if self.prev_free_block else None
#         next_block_id = self.next_free_block.block_id \
#             if self.next_free_block else None
#         return (f"KVCacheBlock(block_id={self.block_id}, "
#                 f"ref_cnt={self.ref_cnt}, "
#                 f"_block_hash={self._block_hash}, "
#                 f"prev_free_block={prev_block_id}, "
#                 f"next_free_block={next_block_id})")
