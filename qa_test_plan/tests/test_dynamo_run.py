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


import subprocess
import time

# test_dynamo_run.py
import pytest
import requests


class DynamoRunProcess:
    def __init__(self, model_path, backend, port):
        self.port = port
        self.model_path = model_path
        self.backend = backend
        self.process = None
        self.url = f"http://localhost:{self.port}"

    def start(self):
        cmd = [
            "dynamo-run",
            "in=http",
            f"out={self.backend}",
            str(self.model_path),
            "--model-name",
            "test-model",
            "--http-port",
            str(self.port),
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self._wait_for_ready()

    def _wait_for_ready(self, timeout=30):
        start_time = time.time()
        while True:
            try:
                response = requests.get(f"{self.url}/health", timeout=1)
                if response.status_code == 200:
                    return
            except Exception:
                pass

            if time.time() - start_time > timeout:
                raise TimeoutError("Server failed to start")
            time.sleep(0.5)

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


@pytest.fixture(scope="module")
def dynamo_server(tmp_path_factory, backend):
    model_path = tmp_path_factory.mktemp("model")
    port = 8888  # In real usage should use random port

    # For EchoEngine, create dummy model files
    if backend == "echo_engine":
        (model_path / "config.json").write_text('{"model_type":"gpt2"}')

    server = DynamoRunProcess(model_path=model_path, backend=backend, port=port)
    server.start()
    yield server
    server.stop()


@pytest.mark.parametrize("backend", ["vllm", "mistralrs"])
def test_basic_chat_completion(dynamo_server):
    # Test OpenAI-compatible endpoint
    url = f"{dynamo_server.url}/v1/chat/completions"
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "2+2="}],
        "max_tokens": 20,
        "temperature": 0,
    }

    response = requests.post(url, json=payload)
    assert response.status_code == 200

    # Verify response structure
    completion = response.json()
    assert "choices" in completion
    assert len(completion["choices"]) > 0
    assert "message" in completion["choices"][0]
    assert "content" in completion["choices"][0]["message"]


@pytest.mark.parametrize("backend", ["echo_engine"])
def test_streaming(dynamo_server):
    url = f"{dynamo_server.url}/v1/chat/completions"
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Repeat 'test'"}],
        "stream": True,
        "max_tokens": 10,
    }

    with requests.post(url, json=payload, stream=True) as response:
        chunks = []
        for line in response.iter_lines():
            if line:
                chunks.append(line.decode())

        assert len(chunks) >= 3  # At least content + stop reason + empty
        assert all("test" in chunk for chunk in chunks if '"content":' in chunk)
