# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo backend tests.

Handles conditional test collection to prevent import errors when backend
frameworks are not installed in the current container.
"""

import importlib.util


def pytest_ignore_collect(collection_path, config):
    """Skip collecting backend test files if their framework isn't installed."""
    path_str = str(collection_path)

    backend_requirements = {
        "/vllm/": "vllm",
        "/sglang/": "sglang",
        "/trtllm/": "tensorrt_llm",
    }

    for path_pattern, required_module in backend_requirements.items():
        if path_pattern in path_str and "/tests/" in path_str:
            if importlib.util.find_spec(required_module) is None:
                return True  # Module not available, skip this file

    return False  # Don't ignore this file
