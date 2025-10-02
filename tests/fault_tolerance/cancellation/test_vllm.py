# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil

import pytest

from tests.fault_tolerance.cancellation.utils import (
    DynamoFrontendProcess,
    poll_for_pattern,
    read_log_content,
    read_streaming_responses,
    send_cancellable_request,
    strip_ansi_codes,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api

logger = logging.getLogger(__name__)


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend"""

    def __init__(self, request, is_prefill: bool = False):
        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.45",
            "--max-model-len",
            "8192",
            "--migration-limit",
            "3",
        ]

        health_check_urls = [
            (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
            (f"http://localhost:{FRONTEND_PORT}/health", check_health_generate),
        ]

        # Set port based on worker type
        port = "8082" if is_prefill else "8081"

        # Add prefill worker flag if needed
        if is_prefill:
            command.append("--is-prefill-worker")
            health_check_urls = [(f"http://localhost:{port}/health", self.is_ready)]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = port

        # Set log directory based on worker type
        worker_type = "prefill_worker" if is_prefill else "worker"
        log_dir = f"{request.node.name}_{worker_type}"

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
            health_check_urls=health_check_urls,
            timeout=300,
            display_output=True,
            terminate_existing=False,
            # Ensure any orphaned vLLM engine cores or child helpers are cleaned up
            stragglers=[
                "VLLM::EngineCore",
            ],
            straggler_commands=[
                "-m dynamo.vllm",
            ],
            log_dir=log_dir,
        )

        self.is_prefill = is_prefill

    def get_pid(self):
        """Get the PID of the worker process"""
        return self.proc.pid if self.proc else None

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        try:
            data = response.json()
            if data.get("status") == "ready":
                worker_type = "Prefill worker" if self.is_prefill else "Worker"
                logger.info(f"{worker_type} status is ready")
                return True
            worker_type = "Prefill worker" if self.is_prefill else "Worker"
            logger.warning(f"{worker_type} status is not ready: {data.get('status')}")
        except ValueError:
            worker_type = "Prefill worker" if self.is_prefill else "Worker"
            logger.warning(f"{worker_type} health response is not valid JSON")
        return False


def verify_request_cancelled(
    frontend_process: DynamoFrontendProcess,
    worker_process: DynamoWorkerProcess,
    prefill_worker_process: DynamoWorkerProcess | None = None,
    frontend_log_offset: int = 0,
    worker_log_offset: int = 0,
    assert_cancel_at_prefill: bool = False,
) -> tuple[int, int]:
    """Verify that the worker and frontend logs contain cancellation messages

    Returns:
        tuple: (new_worker_log_length, new_frontend_log_length)
    """

    # Check worker log for cancellation pattern
    worker_log_content = read_log_content(worker_process._log_path)
    new_worker_content = worker_log_content[worker_log_offset:]

    # Find the LAST occurrence of "New Request ID: <id>" line (health checks may log earlier ones)
    request_id = None
    for line in reversed(new_worker_content.split("\n")):
        # Strip ANSI codes and whitespace for pattern matching
        clean_line = strip_ansi_codes(line).strip()
        if "New Request ID: " in clean_line:
            # Extract ID from the last delimiter occurrence on the line
            parts = clean_line.rsplit("New Request ID: ", 1)
            if len(parts) > 1:
                request_id = parts[-1].strip()
                break
    if request_id is None:
        pytest.fail("Could not find 'New Request ID: <id>' pattern in worker log")

    # Check if the same request ID was cancelled
    has_worker_cancellation = False
    cancellation_pattern = (
        f"Aborted Remote Prefill Request ID: {request_id}"
        if assert_cancel_at_prefill
        else f"Aborted Request ID: {request_id}"
    )
    for line in new_worker_content.split("\n"):
        # Strip ANSI codes and whitespace for pattern matching
        clean_line = strip_ansi_codes(line).strip()
        if clean_line.endswith(cancellation_pattern):
            has_worker_cancellation = True
            break
    if not has_worker_cancellation:
        pytest.fail(f"Could not find '{cancellation_pattern}' pattern in worker log")

    # Check prefill worker log if provided
    if prefill_worker_process is not None:
        prefill_worker_log_content = read_log_content(prefill_worker_process._log_path)

        # Check if the same request ID was remote prefilled
        has_remote_prefill = False
        remote_prefill_pattern = f"New Prefill Request ID: {request_id}"
        for line in prefill_worker_log_content.split("\n"):
            clean_line = strip_ansi_codes(line).strip()
            if clean_line.endswith(remote_prefill_pattern):
                has_remote_prefill = True
                break
        if not has_remote_prefill:
            pytest.fail(
                f"Could not find '{remote_prefill_pattern}' pattern in prefill worker log"
            )

        # Check for remote prefill cancellation
        if assert_cancel_at_prefill:
            has_prefill_cancellation = False
            prefill_cancellation_pattern = f"Aborted Prefill Request ID: {request_id}"
            for line in prefill_worker_log_content.split("\n"):
                clean_line = strip_ansi_codes(line).strip()
                if clean_line.endswith(prefill_cancellation_pattern):
                    has_prefill_cancellation = True
                    break
            if not has_prefill_cancellation:
                pytest.fail(
                    f"Could not find '{prefill_cancellation_pattern}' pattern in prefill worker log"
                )

    # Check frontend log for cancellation issued pattern
    frontend_log_content = read_log_content(frontend_process._log_path)
    new_frontend_content = frontend_log_content[frontend_log_offset:]

    has_kill_message = False
    kill_message = "issued control message Kill to sender"
    for line in new_frontend_content.split("\n"):
        # Strip ANSI codes and whitespace for pattern matching
        clean_line = strip_ansi_codes(line).strip()
        if clean_line.endswith(kill_message):
            has_kill_message = True
            break
    if not has_kill_message:
        pytest.fail("Could not find cancellation issued in frontend log")

    return len(frontend_log_content), len(worker_log_content)


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_vllm_aggregated(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation functionality in aggregated mode.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the worker side in aggregated (single worker) mode. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start a single worker
        logger.info("Starting worker...")
        worker = DynamoWorkerProcess(request)

        with worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            # Step 3: Test request cancellation with polling approach
            frontend_log_offset, worker_log_offset = 0, 0

            test_scenarios = [
                ("completion", "Completion request cancellation"),
                ("chat_completion", "Chat completion request cancellation"),
                (
                    "chat_completion_stream",
                    "Chat completion stream request cancellation",
                ),
            ]

            for request_type, description in test_scenarios:
                logger.info(f"Testing {description.lower()}...")

                # Send the request (non-blocking)
                cancellable_req = send_cancellable_request(request_type)

                # Poll for "New Request ID" pattern
                request_id, worker_log_offset = poll_for_pattern(
                    process=worker,
                    pattern="New Request ID: ",
                    log_offset=worker_log_offset,
                    match_type="contains",
                )

                # For streaming, read 5 responses before cancelling
                if request_type == "chat_completion_stream":
                    read_streaming_responses(cancellable_req, expected_count=5)

                # Now cancel the request
                cancellable_req.cancel()
                logger.info(f"Cancelled request ID: {request_id}")

                # Poll for "Aborted Request ID" with matching ID
                _, worker_log_offset = poll_for_pattern(
                    process=worker,
                    pattern=f"Aborted Request ID: {request_id}",
                    log_offset=worker_log_offset,
                )

                # Verify frontend log has kill message
                _, frontend_log_offset = poll_for_pattern(
                    process=frontend,
                    pattern="issued control message Kill to sender",
                    log_offset=frontend_log_offset,
                )

                logger.info(f"{description} detected successfully")


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_vllm_decode_cancel(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation during decode phase.

    This test verifies that when a request is cancelled by the client during the decode phase,
    the system properly handles the cancellation and cleans up resources
    on the decode worker side in a disaggregated setup.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker
        logger.info("Starting prefill worker...")
        prefill_worker = DynamoWorkerProcess(request, is_prefill=True)

        with prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker
            logger.info("Starting decode worker...")
            decode_worker = DynamoWorkerProcess(request, is_prefill=False)

            with decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # Step 4: Test request cancellation for streaming scenario
                logger.info(
                    "Testing chat completion stream request cancellation in decode worker (decode phase)..."
                )

                # Send streaming request (non-blocking)
                cancellable_req = send_cancellable_request("chat_completion_stream")

                # Poll for "New Request ID" pattern in decode worker
                request_id, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern="New Request ID: ",
                    match_type="contains",
                )

                # Verify same request ID reached prefill worker (as "New Prefill Request ID")
                _, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern=f"New Prefill Request ID: {request_id}",
                )

                # Read 5 streaming responses (decode phase)
                read_streaming_responses(cancellable_req, expected_count=5)

                # Now cancel the request
                cancellable_req.cancel()
                logger.info(f"Cancelled request ID: {request_id}")

                # Poll for "Aborted Request ID" in decode worker
                _, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern=f"Aborted Request ID: {request_id}",
                    log_offset=decode_log_offset,
                )

                # Verify frontend log has kill message
                _, frontend_log_offset = poll_for_pattern(
                    process=frontend,
                    pattern="issued control message Kill to sender",
                )

                logger.info(
                    "Chat completion stream cancellation in decode phase detected successfully"
                )


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_vllm_remote_prefill_cancel(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation during remote prefill phase.

    This test verifies that when a request is cancelled by the client during the remote prefill phase,
    the system properly handles the cancellation and cleans up resources
    on both the decode and prefill workers in a disaggregated setup.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker
        logger.info("Starting prefill worker...")
        prefill_worker = DynamoWorkerProcess(request, is_prefill=True)

        with prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker
            logger.info("Starting decode worker...")
            decode_worker = DynamoWorkerProcess(request, is_prefill=False)

            with decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # Step 4: Test request cancellation during remote prefill phase
                logger.info(
                    "Testing completion request cancellation during remote prefill phase..."
                )

                # Send request with long prompt (non-blocking)
                cancellable_req = send_cancellable_request(
                    "completion", use_long_prompt=True
                )

                # Poll for "New Request ID" pattern in decode worker
                request_id, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern="New Request ID: ",
                    match_type="contains",
                )

                # Poll for same request ID in prefill worker (as "New Prefill Request ID")
                _, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern=f"New Prefill Request ID: {request_id}",
                )

                # Cancel during prefill phase
                cancellable_req.cancel()
                logger.info(f"Cancelled request ID: {request_id} during remote prefill")

                # Poll for "Aborted Prefill Request ID" in prefill worker first (where cancellation happens)
                _, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern=f"Aborted Prefill Request ID: {request_id}",
                    log_offset=prefill_log_offset,
                )

                # Then poll for "Aborted Remote Prefill Request ID" in decode worker
                _, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern=f"Aborted Remote Prefill Request ID: {request_id}",
                    log_offset=decode_log_offset,
                )

                # Verify frontend log has kill message
                _, frontend_log_offset = poll_for_pattern(
                    process=frontend,
                    pattern="issued control message Kill to sender",
                )

                logger.info(
                    "Completion request cancellation during remote prefill phase detected successfully"
                )
