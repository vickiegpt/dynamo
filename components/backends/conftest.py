# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for backend unit tests.

Handles conditional test collection to prevent import errors when backend
frameworks are not installed in the current container.
"""

import importlib.util

# Map test file patterns to required framework modules
BACKEND_REQUIREMENTS = {
    "test_vllm": "vllm",
    "test_sglang": "sglang",
    "test_trtllm": "tensorrt_llm",
}


def pytest_ignore_collect(collection_path, config):
    """Skip collecting backend test files if their framework isn't installed."""
    path_str = str(collection_path)

    for file_pattern, required_module in BACKEND_REQUIREMENTS.items():
        if file_pattern in path_str:
            if importlib.util.find_spec(required_module) is None:
                return True  # Module not available, skip this file

    return False  # Don't ignore this file
