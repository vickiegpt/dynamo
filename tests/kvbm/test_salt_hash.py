# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
from dataclasses import dataclass, field

import pytest
import requests

from tests.serve.common import params_with_model_mark, run_serve_deployment
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    chat_payload,
    chat_payload_default,
    completion_payload_default,
    metric_payload_default,
)
from tests.utils.constants import TINY_LLAMA_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess, DynamoFrontendProcess
from tests.utils.payloads import check_models_api, completions_response_handler

logger = logging.getLogger(__name__)

PROMPT = "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."

class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend"""

    def __init__(self, request, worker_id: str, engine_config_file: str):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model-path",
            TINY_LLAMA_MODEL_NAME,
            "--served-model-name",
            TINY_LLAMA_MODEL_NAME,
            "--extra-engine-args",
            engine_config_file
        ]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = "9345"
        env["DYN_KVBM_CPU_CACHE_GB"] = "20"
        env["DYN_KVBM_DISK_CACHE_GB"] = "60"
        env["DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS"] = "1200"


        # TODO: Have the managed process take a command name explicitly to distinguish
        #       between processes started with the same command.
        log_dir = f"{request.node.name}_{worker_id}"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            # Directory doesn't exist, which is fine
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
                ("http://localhost:9345/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            log_dir=log_dir,
        )

    def get_pid(self) -> int | None:
        """Get the PID of the worker process"""
        return self.proc.pid if hasattr(self, "proc") and self.proc else None

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(
                    f"{self.__class__.__name__} {{ name: {self.worker_id} }} status is ready"
                )
                return True
            logger.warning(
                f"{self.__class__.__name__} {{ name: {self.worker_id} }} status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(
                f"{self.__class__.__name__} {{ name: {self.worker_id} }} health response is not valid JSON"
            )
        return False

def send_completion_request(
    model_name: str, prompt: str, max_tokens: int, timeout: int = 120
) -> requests.Response:
    """Send a completion request to the frontend"""
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    headers = {"Content-Type": "application/json"}

    logger.info(
        f"Sending completion request with prompt: '{prompt[:50]}...' and max_tokens: {max_tokens}"
    )

    try:
        response = requests.post(
            "http://localhost:8000/v1/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        logger.info(f"Received response with status code: {response.status_code}")
        return response
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed with error: {e}")
        raise


@pytest.mark.trtllm
@pytest.mark.kvbm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(TINY_LLAMA_MODEL_NAME)
def test_e2e_salt_hash(request, runtime_services):
    """
    End-to-end test of the salt hash feature in KVBM.

    TODO: write more test description
    """

    logger.info("Starting frontend...")
    with DynamoFrontendProcess(request):
        logger.info("Frontend started.")

        engine_config_file = "tests/kvbm/simple_kvbm_config.yaml"
        logger.info("Starting trtllm worker...")
        with DynamoWorkerProcess(request, "decode", engine_config_file) as worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            print(f"{send_completion_request(TINY_LLAMA_MODEL_NAME, PROMPT, 100, timeout=10)}")