import subprocess
import time
import requests
from contextlib import contextmanager
import pytest
import os


@contextmanager
def run_dynamo_hello_world():
    # Launch `dynamo serve hello_world:Frontend`
    proc = subprocess.Popen(
        ["dynamo", "serve", "hello_world:Frontend"],
        cwd="../../../examples/hello_world",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        for _ in range(60): 
            try:
                response = requests.post(
                    "http://localhost:8000/generate",
                    timeout=1
                )
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise TimeoutError("Dynamo service failed to start on port 8000")

        yield
    finally:
        proc.terminate()
        proc.wait()


def test_hello_world_service():
    with run_dynamo_hello_world():
        payload = {"text": "test"}
        headers = {
            "accept": "text/event-stream",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "http://localhost:8000/generate",
            json=payload,
            headers=headers,
            timeout=10,
        )

        assert response.status_code == 200
        assert response.text.strip() != ""
        print("Response:", response.text)