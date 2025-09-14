# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Scheduler implementation that forwards to vLLM's default scheduler.

This module provides a custom scheduler that acts as a springboard to vLLM's
default scheduler implementation, allowing for future customization while
maintaining compatibility with vLLM's scheduling interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple, Union

from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal import MultiModalRegistry
    from vllm.transformers_utils.structured_outputs import StructuredOutputManager
    from vllm.v1.core.kv_cache_manager import KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.scheduler import DraftTokenIds, ModelRunnerOutput, SchedulerStats
    from vllm.v1.outputs import EngineCoreOutputs
    from vllm.v1.request import Request, RequestStatus


class DynamoScheduler(SchedulerInterface):
    """
    Custom scheduler that forwards all operations to vLLM's default Scheduler.

    This scheduler acts as a transparent proxy, allowing for future customization
    of scheduling behavior while maintaining full compatibility with vLLM.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        mm_registry: Optional["MultiModalRegistry"] = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Initialize the DynamoScheduler with a wrapped vLLM Scheduler.

        Args:
            vllm_config: vLLM configuration object
            kv_cache_config: KV cache configuration
            structured_output_manager: Manager for structured outputs
            mm_registry: Multi-modal registry (optional, will use default if None)
            include_finished_set: Whether to include finished requests
            log_stats: Whether to log statistics
        """
        # Import here to handle optional mm_registry parameter
        from vllm.multimodal import MULTIMODAL_REGISTRY

        # Use provided registry or default
        if mm_registry is None:
            mm_registry = MULTIMODAL_REGISTRY

        # Create the underlying vLLM scheduler
        self._scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

    def schedule(self) -> "SchedulerOutput":
        """
        Schedule requests for the next model forward pass.

        Returns:
            SchedulerOutput containing scheduling decisions
        """
        return self._scheduler.schedule()

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> Dict[int, "EngineCoreOutputs"]:
        """
        Update scheduler state after model processing.

        Args:
            scheduler_output: Output from the schedule() method
            model_runner_output: Output from the model runner

        Returns:
            Dictionary mapping request IDs to engine core outputs
        """
        return self._scheduler.update_from_output(scheduler_output, model_runner_output)

    def update_draft_token_ids(
        self,
        draft_token_ids: "DraftTokenIds",
    ) -> None:
        """
        Update draft token IDs for scheduled requests.

        Args:
            draft_token_ids: Draft token IDs to update
        """
        self._scheduler.update_draft_token_ids(draft_token_ids)

    def add_request(self, request: "Request") -> None:
        """
        Add a new request to the scheduler.

        Args:
            request: Request object to add to the scheduler
        """
        self._scheduler.add_request(request)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: "RequestStatus",
    ) -> None:
        """
        Mark requests as finished.

        Args:
            request_ids: Request ID(s) to mark as finished
            finished_status: The finish status for the requests
        """
        self._scheduler.finish_requests(request_ids, finished_status)

    def get_num_unfinished_requests(self) -> int:
        """
        Get the number of unfinished requests.

        Returns:
            Number of unfinished requests in the scheduler
        """
        return self._scheduler.get_num_unfinished_requests()

    def has_finished_requests(self) -> bool:
        """
        Check if there are any finished requests.

        Returns:
            True if there are finished requests, False otherwise
        """
        return self._scheduler.has_finished_requests()

    def reset_prefix_cache(self) -> bool:
        """
        Reset the prefix cache.

        Returns:
            True if cache was reset successfully
        """
        return self._scheduler.reset_prefix_cache()

    def get_request_counts(self) -> Tuple[int, int]:
        """
        Get counts of requests in different states.

        Returns:
            Tuple of (waiting_count, running_count)
        """
        return self._scheduler.get_request_counts()

    def make_stats(self) -> Optional["SchedulerStats"]:
        """
        Generate statistics about the scheduler's current state.

        Returns:
            SchedulerStats object or None
        """
        return self._scheduler.make_stats()

    def shutdown(self) -> None:
        """
        Shutdown the scheduler and clean up resources.
        """
        self._scheduler.shutdown()
