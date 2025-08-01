# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Singleton DistributedRuntime utility for VLLM integration.

This module provides a singleton DistributedRuntime instance to prevent
the "Worker already initialized" error when multiple KV connectors are created.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from dynamo.runtime import DistributedRuntime

# Global singleton instance and lock to ensure thread safety
_global_drt: Optional[DistributedRuntime] = None
_drt_lock = threading.Lock()
_logger = logging.getLogger(__name__)


def get_or_create_distributed_runtime(
    event_loop=None, is_static: bool = False
) -> DistributedRuntime:
    """
    Get or create a singleton DistributedRuntime instance.

    This prevents the "Worker already initialized" error by ensuring
    only one DistributedRuntime is created per process. If the underlying
    Rust Worker is already initialized from a previous call, this function
    will handle it gracefully.

    Args:
        event_loop: The event loop to use (usually None for default)
        is_static: Whether to use static mode (False for dynamic discovery)

    Returns:
        The singleton DistributedRuntime instance
    """
    global _global_drt

    with _drt_lock:
        if _global_drt is None:
            try:
                _global_drt = DistributedRuntime(
                    event_loop=event_loop, is_static=is_static
                )
                _logger.debug("Created new DistributedRuntime singleton instance")
            except Exception as e:
                # Check if this is the "Worker already initialized" error
                if "Worker already initialized" in str(e):
                    _logger.warning(
                        "Rust Worker already initialized - this suggests a DistributedRuntime "
                        "was created outside of the singleton pattern. This may cause issues."
                    )
                    # In this case, we can't create a new DistributedRuntime because the
                    # underlying Rust Worker already exists. This is a design limitation.
                    raise RuntimeError(
                        "Cannot create DistributedRuntime: underlying Rust Worker already "
                        "initialized. Ensure all DistributedRuntime instances use the "
                        "get_or_create_distributed_runtime() singleton function."
                    ) from e
                else:
                    # Some other error occurred
                    raise
        else:
            _logger.debug("Returning existing DistributedRuntime singleton instance")

        return _global_drt


def reset_distributed_runtime() -> None:
    """
    Reset the singleton DistributedRuntime instance.

    WARNING: This only resets the Python singleton. If the underlying Rust
    Worker has already been initialized, subsequent calls to create a new
    DistributedRuntime will fail. This function is mainly useful for testing
    scenarios where you want to simulate multiple initialization attempts.

    Note: This doesn't shutdown the existing runtime, that should be done
    separately if needed.
    """
    global _global_drt

    with _drt_lock:
        if _global_drt is not None:
            _logger.warning(
                "Resetting DistributedRuntime singleton. Note: underlying Rust Worker "
                "may still be initialized and prevent creation of new instances."
            )
        _global_drt = None


def has_distributed_runtime() -> bool:
    """
    Check if a DistributedRuntime singleton instance exists.

    Returns:
        True if a singleton instance exists, False otherwise
    """
    with _drt_lock:
        return _global_drt is not None
