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
import subprocess
import time
from contextlib import contextmanager

import pytest
import requests


@contextmanager
def managed_process(
    command, env=None, check_ports=[], timeout=600, cwd=None, output=False
):
    print(" ".join(command))
    _input = subprocess.DEVNULL
    _output = subprocess.DEVNULL
    if output:
        _input = None
        _output = None
    proc = subprocess.Popen(
        command,
        env=env or os.environ.copy(),
        cwd=cwd,
        stdin=_input,
        stdout=_output,
        stderr=_output,
    )
    start_time = time.time()

    # Wait for ports to become available
    while time.time() - start_time < timeout:
        if all(is_port_open(p) for p in check_ports):
            break
        time.sleep(0.1)
    else:
        proc.terminate()
        raise TimeoutError(f"Ports {check_ports} not ready in time")

    try:
        yield proc
    finally:
        proc.terminate()
        proc.wait()


def is_port_open(port):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture(scope="session")
def etcd_server():
    etcd_env = os.environ.copy()
    etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
    with managed_process(
        [
            "etcd",
            "--listen-client-urls",
            "http://0.0.0.0:2379",
            "--advertise-client-urls",
            "http://0.0.0.0:2379",
        ],
        env=etcd_env,
        check_ports=[2379],
    ) as proc:
        yield proc


@pytest.fixture(scope="session")
def nats_server():
    with managed_process(["nats-server", "-js", "--trace"], check_ports=[4222]) as proc:
        yield proc


@pytest.fixture(
    params=[
        "agg",
        "agg_router",
    ]
)  #'disagg_router','disagg'])
def deployment_config(request):
    config_map = {
        "agg": ("graphs.agg:Frontend", "configs/agg.yaml"),
        "disagg": ("graphs.disagg:Frontend", "configs/disagg.yaml"),
        "agg_router": ("graphs.agg_router:Frontend", "configs/agg_router.yaml"),
        "disagg_router": (
            "graphs.disagg_router:Frontend",
            "configs/disagg_router.yaml",
        ),
    }
    return config_map[request.param]


@contextmanager
def dynamo_serve_process(graph_module, config_path):
    with managed_process(
        ["dynamo", "serve", graph_module, "-f", config_path],
        check_ports=[8000],  # Frontend port
        timeout=600,
        cwd="/workspace/examples/llm",
        output=True,
    ) as proc:
        yield proc


def test_deployment(etcd_server, nats_server, deployment_config):
    graph_module, config_path = deployment_config

    with dynamo_serve_process(graph_module, config_path):
        # Test OpenAI-compatible endpoint
        url = "http://localhost:8000/v1/chat/completions"
        payload = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the concept of quantum entanglement in 20 words",
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        # Retry to account for service startup time
        for _ in range(40):
            try:
                response = requests.post(url, json=payload, timeout=600)
                if response.status_code == 200:
                    time.sleep(0.5)
                    continue
                time.sleep(5)
            except requests.ConnectionError:
                time.sleep(1)
        else:
            pytest.fail("Service failed to start within timeout")

        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "message" in result["choices"][0]
        assert len(result["choices"][0]["message"]["content"]) > 10
        print(result["choices"][0]["message"]["content"])


@pytest.fixture(scope="session", autouse=True)
def cleanup_processes():
    yield
    subprocess.run(["pkill", "-f", "nats-server|etcd"], check=False)
