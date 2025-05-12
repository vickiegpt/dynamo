# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import pytest
import requests

pytestmark = pytest.mark.e2e 

# Utility to manage subprocesses
@contextmanager
def managed_process(
    command, env=None, check_ports=None, timeout=600, cwd=None, output=False
):
    print(f"Running command: {' '.join(command)} in {cwd or os.getcwd()}")

    stdin: Union[int, None] = subprocess.DEVNULL
    stdout: Union[int, None] = subprocess.DEVNULL
    stderr: Union[int, None] = subprocess.DEVNULL
    if output:
        stdin = None
        stdout = None
        stderr = None

    proc = subprocess.Popen(
        command,
        env=env or os.environ.copy(),
        cwd=cwd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )
    start_time = time.time()

    # Wait for ports to be available
    if check_ports:
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

def is_port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

#Fixtures for external services; should we use the docker compose instead? 
@pytest.fixture()
def etcd_server():
    etcd_env = os.environ.copy()
    etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
    shutil.rmtree("/tmp/etcd-test-data", ignore_errors=True)
    with managed_process(
        [
            "etcd",
            "--listen-client-urls", "http://0.0.0.0:2379",
            "--advertise-client-urls", "http://0.0.0.0:2379",
            "--data-dir", "/tmp/etcd-test-data",
        ],
        env=etcd_env,
        check_ports=[2379],
        output=True,
    ) as proc:
        yield proc

@pytest.fixture()
def nats_server():
    shutil.rmtree("/tmp/nats/jetstream", ignore_errors=True)
    with managed_process(
        ["nats-server", "-js", "--trace", "--store_dir", "/tmp/nats/jetstream"],
        check_ports=[4222],
        output=True,
    ) as proc:
        yield proc

# Deployment configurations
@pytest.fixture(
    params=["agg", "agg_router", "disagg_skeleton"]
)
def deployment_config(request):
    config_map = {
        "agg": (
            "graphs.agg:Frontend",
            "configs/agg.yaml",
            "/workspace/examples/llm",
        ),
        "agg_router": (
            "graphs.agg_router:Frontend",
            "configs/agg_router.yaml",
            "/workspace/examples/llm",
        ),
        "disagg_skeleton": (
            "components.graph:Frontend",
            "",
            "/workspace/examples/hello_world/disagg_skeleton",
        ),
    }
    return config_map[request.param]

#Dynamo serve wrapper
@contextmanager
def dynamo_serve_process(graph_module: str, config_path: str, cwd: str):
    command = ["dynamo", "serve", graph_module]
    if config_path:
        full_cfg = os.path.join(cwd, config_path)
        if not os.path.isfile(full_cfg):
            raise FileNotFoundError(f"Config file not found: {full_cfg}")
        command.extend(["-f", full_cfg])

    with managed_process(
        command,
        check_ports=[8000],
        timeout=600,
        cwd=cwd,
        output=True,
    ) as proc:
        yield proc

# Add more tests with different prompts. Multi modal prompt support? 
def test_deployment(etcd_server, nats_server, deployment_config):
    graph_module, config_path, cwd = deployment_config

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "messages": [
            {"role": "user", "content": (
                "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, "
                "lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria "
                "was buried beneath the shifting sands of time, lost to the world for centuries. You are "
                "an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled "
                "upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has "
                "the potential to reshape the very fabric of reality. Your journey will take you through "
                "treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: "
                "Character Background: Develop a detailed background for your character. Describe their "
                "motivations for seeking out Aeloria, their skills and weaknesses, and any personal "
                "connections to the ancient city or its legends. Are they driven by a quest for knowledge, "
                "a search for lost familt clue is hidden."
            )}
        ],
        "stream": False,
        "max_tokens": 30,
        "temperature": 0.1,
    }

    with dynamo_serve_process(graph_module, config_path, cwd):
        url = "http://localhost:8000/v1/chat/completions"
        response: Optional[requests.Response] = None

        for _ in range(10):
            try:
                response = requests.post(url, json=payload, timeout=60)
                if response and response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        else:
            pytest.fail("Service did not respond with HTTP 200 within timeout")

        assert response is not None
        assert response.status_code == 200

        result = response.json()
        assert "choices" in result and len(result["choices"]) > 0
        choice = result["choices"][0]
        assert "message" in choice
        content: str = choice["message"]["content"]
        assert isinstance(content, str) and len(content) > 10
        print("Model response:", content)

#  Cleanup fixture 
# TODO use contaxt manager, this can be improved. 
@pytest.fixture(scope="function", autouse=True)
def cleanup_processes():
    yield
    subprocess.run(["pkill", "-f", "nats-server"], check=False)
    subprocess.run(["pkill", "-f", "etcd"], check=False)
