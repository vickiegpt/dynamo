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

import os
import shutil
import subprocess
import time
from typing import Dict, Any, Optional

import pytest

from tests.utils import managed_process, find_free_port


# Dictionary to hold cached model instances
_MODEL_CACHE = {}


def _ensure_killed(pattern: str):
    subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    time.sleep(0.1)


@pytest.fixture(scope="session")
def etcd_server():
    _ensure_killed("etcd")
    # Hardcoded port for ETCD
    client_port = 2379
    peer_port = 2380
    # client_port = find_free_port()
    # peer_port   = find_free_port()

    client_url = f"http://127.0.0.1:{client_port}"
    peer_url   = f"http://127.0.0.1:{peer_port}"

    etcd_env = os.environ.copy()
    etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
    shutil.rmtree("/tmp/etcd-test-data", ignore_errors=True)

    cmd = [
        "etcd",
        "--listen-client-urls", client_url,
        "--advertise-client-urls", client_url,
        "--listen-peer-urls", peer_url,
        "--initial-advertise-peer-urls", peer_url,
        "--initial-cluster",              f"default={peer_url}",
        "--initial-cluster-state",        "new",
        "--data-dir",                     "/tmp/etcd-test-data",
    ]

    with managed_process(cmd, env=etcd_env, check_ports=[client_port], output=True):
        yield client_url


@pytest.fixture(scope="session")
def nats_server():
    _ensure_killed("nats-server")
    # Hardcoded ports for NATS
    client_port = 4222
    http_port = 8222
    # client_port = find_free_port()
    # http_port   = find_free_port()

    shutil.rmtree("/tmp/nats/jetstream", ignore_errors=True)
    os.makedirs("/tmp/nats/jetstream", exist_ok=True)

    cmd = [
        "nats-server",
        "-js",
        "--trace",
        "--store_dir", "/tmp/nats/jetstream",
        "--port",        str(client_port),
        "--http_port",   str(http_port),
    ]

    with managed_process(cmd, check_ports=[client_port, http_port], output=True):
        time.sleep(0.2)
        yield {
            "client_url":   f"nats://127.0.0.1:{client_port}",
            "http_monitor": f"http://127.0.0.1:{http_port}",
        }


@pytest.fixture(scope="session")
def model_loader():
    """
    Session-scoped fixture for loading and caching models.
    
    This fixture provides a function that loads models only once per test session,
    then returns the cached instances for subsequent requests, improving test performance.
    """
    def _load_model(model_name: str, backend: str = "vllm", **kwargs) -> Any:
        """
        Load a model or return a cached instance if available.
        
        Args:
            model_name: Name or path of the model to load
            backend: Backend to use for model loading (vllm, transformers, etc.)
            **kwargs: Additional arguments to pass to the model loader
            
        Returns:
            The loaded model instance
        """
        cache_key = f"{model_name}_{backend}_{sorted(kwargs.items())}"
        
        if cache_key in _MODEL_CACHE:
            print(f"Using cached model: {model_name}")
            return _MODEL_CACHE[cache_key]
        
        print(f"Loading model: {model_name} with backend {backend}")
        # Implement actual model loading logic based on backend
        # This is a placeholder - replace with actual implementation
        if backend == "vllm":
            try:
                # Import here to avoid dependency issues if not testing with vllm
                from vllm import LLM
                model = LLM(model=model_name, **kwargs)
                _MODEL_CACHE[cache_key] = model
                return model
            except ImportError:
                pytest.skip("vllm not installed, skipping test")
        elif backend == "transformers":
            try:
                # Import here to avoid dependency issues
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                _MODEL_CACHE[cache_key] = (model, tokenizer)
                return model, tokenizer
            except ImportError:
                pytest.skip("transformers not installed, skipping test")
        
        # Add support for other backends as needed
        raise ValueError(f"Unsupported backend: {backend}")
    
    return _load_model


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup: kill stray daemons and reap zombies."""
    # Clear model cache
    for key, model in list(_MODEL_CACHE.items()):
        try:
            # Attempt to clean up model resources if possible
            if hasattr(model, 'unload') and callable(model.unload):
                model.unload()
            # Remove from cache
            del _MODEL_CACHE[key]
        except Exception as e:
            print(f"Error cleaning up model {key}: {e}")
    
    subprocess.run(["pkill", "-9", "-f", "etcd|nats-server"], check=False)
    import os
    from time import sleep
    sleep(0.1)
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
        except ChildProcessError:
            break
