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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest


@contextmanager
def managed_process(
    command, env=None, check_ports=[], timeout=600, cwd=None, output=False
):
    cmd_string = " ".join(command)
    cmd_name = command[0]
    print()
    print("*" * 80)
    print(f"* Starting: {cmd_name}")
    print(f"* \t {cmd_string}")
    print("*" * 80)
    print()

    _input: Optional[int] = subprocess.DEVNULL
    _output: Optional[int] = subprocess.DEVNULL
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
        raise TimeoutError(
            f"{cmd_name} failed to start. Ports {check_ports} not ready in time"
        )

    try:
        yield proc
    finally:
        print(f"Terminating {cmd_name}")
        proc.terminate()
        proc.wait()


def is_port_open(port):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture()
def etcd_server(port=2379):
    port_string = str(port)
    etcd_env = os.environ.copy()
    etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
    shutil.rmtree("/tmp/etcd-test-data", ignore_errors=True)
    with managed_process(
        [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{port_string}",
            "--advertise-client-urls",
            f"http://0.0.0.0:{port_string}",
            "--data-dir",
            "/tmp/etcd-test-data",
        ],
        env=etcd_env,
        check_ports=[port],
    ) as proc:
        yield proc


@pytest.fixture()
def nats_server(port=4222):
    shutil.rmtree("/tmp/nats/jetstream", ignore_errors=True)
    with managed_process(
        ["nats-server", "-js", "--trace", "--store_dir", "/tmp/nats/jetstream"],
        check_ports=[port],
    ) as proc:
        yield proc


@pytest.fixture(scope="function", autouse=True)
def runtime_processes(etcd_server, nats_server):
    yield
    subprocess.run(["pkill", "-f", "nats-server|etcd"], check=False)


@pytest.fixture(params=[])
def prompt(request):
    pass


@pytest.fixture(
    params=["agg", "agg_router", "disagg_skeleton"]
)  #'disagg_router','disagg'])
def deployment_config(request):
    config_map = {
        "agg": ("graphs.agg:Frontend", "configs/agg.yaml", "/workspace/examples/llm"),
        "disagg": (
            "graphs.disagg:Frontend",
            "configs/disagg.yaml",
            "/workspace/examples/llm",
        ),
        "agg_router": (
            "graphs.agg_router:Frontend",
            "configs/agg_router.yaml",
            "/workspace/examples/llm",
        ),
        "disagg_router": (
            "graphs.disagg_router:Frontend",
            "configs/disagg_router.yaml",
            "/workspae/examples/llm",
        ),
        "disagg_skeleton": (
            "components.graph:Frontend",
            "",
            "/workspace/examples/hello_world/disagg_skeleton",
        ),
    }
    return config_map[request.param]


@dataclass
class DeploymentGraph:
    module: str
    config: str
    directory: str
    endpoint: str
    response_handler: Callable[[Any], str]


@contextmanager
def dynamo_serve_process(graph: DeploymentGraph, port=8000, timeout=600):
    command = ["dynamo", "serve", graph.module]
    if graph.config:
        command.extend(["-f", graph.config])

    with managed_process(
        command,
        check_ports=[port],  # Frontend port
        timeout=timeout,
        cwd=graph.directory,
        output=True,
    ) as proc:
        yield proc


def completions_response_handler(response):
    if response.status_code != 200:
        return ""
    result = response.json()
    assert "choices" in result
    assert len(result["choices"]) > 0
    assert "message" in result["choices"][0]
    assert "content" in result["choices"][0]["message"]
    return result["choices"][0]["message"]["content"]


@pytest.fixture(
    params=["agg", "agg_router", "disagg_skeleton"]
)  #'disagg_router','disagg'])
def deployment_graph(request):
    graphs = {
        "agg": DeploymentGraph(
            "graphs.agg:Frontend",
            "configs/agg.yaml",
            "/workspace/examples/llm",
            "v1/chat/completions",
            completions_response_handler,
        )
    }
    return graphs[request.param]


def test_deployment(deployment_graph: deployment_graph):
    with dynamo_serve_process(deployment_graph):
        pass


# def test_deployment(etcd_server, nats_server, deployment_config):
#     graph_module, config_path, cwd = deployment_config

#     with dynamo_serve_process(graph_module, config_path, cwd):
#         # Test OpenAI-compatible endpoint
#         url = "http://localhost:8000/v1/chat/completions"
#         payload = {
#             "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.",
#                 }
#             ],
#             "max_tokens": 500,
#             "temperature": 0.1,
#         }

#         # Retry to account for service startup time
#         for _ in range(4):
#             try:
#                 response = requests.post(url, json=payload, timeout=600)
#                 if response.status_code == 200:
#                     time.sleep(0.5)
#                     continue
#                 time.sleep(5)
#             except requests.ConnectionError:
#                 time.sleep(1)
#         else:
#             if response.status_code != 200:
#                 pytest.fail("Service failed to start within timeout")

#         assert response.status_code == 200
#         result = response.json()
#         assert "choices" in result
#         assert len(result["choices"]) > 0
#         assert "message" in result["choices"][0]
#         assert len(result["choices"][0]["message"]["content"]) > 10
#         print(result["choices"][0]["message"]["content"])


# @pytest.fixture(scope="function", autouse=True)
# def cleanup_processes():
#     yield
#     subprocess.run(["pkill", "-f", "nats-server|etcd"], check=False)
