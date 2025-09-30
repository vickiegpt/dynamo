#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinism test for KVBM in disaggregated mode.

To make sure KVBM's accuracy, this test suite checks if the model produces
deterministic outputs when same requests are served 1) without KVBM onboarded KV
blocks and 2) with KVBM onboarded KV blocks, when given the same inputs with
fixed seed and temperature=0.

The expected results should be at least 90% match between the two cases.
Compared to aggregated mode, disaggregated mode has some known randomness.
Example reference: https://github.com/vllm-project/vllm/issues/7779#issuecomment-2304967870
"""

import importlib.util
import logging
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, TextIO

import pytest
import requests

from .common import KVBM_PYTEST_MARKS, DeterminismTester
from .common import TestDeterminism as BaseTestDeterminism

# Test markers to align with repository conventions
# Todo: enable the rest when kvbm is built in the ci
pytestmark = KVBM_PYTEST_MARKS + [pytest.mark.gpu_2]


# NOTE: disagg only supports vllm now, no trtllm support
class ServerType(str, Enum):
    vllm = "vllm"


class LLMServerManager:
    """Manages LLM server lifecycle for determinism testing."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        port: Optional[int] = None,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
        log_dir: Optional[Path] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        self.server_type = server_type
        self.port = port or int(os.environ.get("KVBM_SERVER_PORT", "8000"))
        self.base_url = base_url or f"http://localhost:{self.port}"
        self.process_frontend: Optional[subprocess.Popen] = None
        self.process_prefiller: Optional[subprocess.Popen] = None
        self.process_decoder: Optional[subprocess.Popen] = None
        self.cpu_cache_blocks = cpu_cache_blocks
        self.gpu_cache_blocks = gpu_cache_blocks

        # Prepare logging
        self.log_dir = log_dir or Path(".")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_str = (
            f"cpu{cpu_cache_blocks or 'default'}_gpu{gpu_cache_blocks or 'default'}"
        )

        self.prefiller_log_file = (
            self.log_dir / f"{self.server_type}_prefiller_{config_str}_{timestamp}.log"
        )
        self.prefiller_stdout_file: Optional[TextIO] = None
        self.prefiller_stderr_file: Optional[TextIO] = None

        self.decoder_log_file = (
            self.log_dir / f"{self.server_type}_decoder_{timestamp}.log"
        )
        self.decoder_stdout_file: Optional[TextIO] = None
        self.decoder_stderr_file: Optional[TextIO] = None

        # Environment for the process
        self.env = os.environ.copy()
        self.env.update(
            {
                "RUST_BACKTRACE": "1",
                # DynamoConnector connection settings
                "NATS_SERVER": "nats://localhost:4222",
                "ETCD_ENDPOINTS": "http://localhost:2379",
            }
        )

        # CPU cache blocks override via env
        if cpu_cache_blocks is not None:
            self.env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] = str(cpu_cache_blocks)

        self._set_up_dynamo_config()

        if self.server_type == ServerType.vllm:
            self._set_up_vllm_config(gpu_cache_blocks)
        else:
            raise ValueError(
                f"{self.server_type} is not supported yet in the KVBM test suite"
            )

    def _set_up_dynamo_config(self, router_mode: str = "kv"):
        self.dynamo_frontend_cmd = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--router-mode",
            router_mode,
            "--http-port",
            str(self.port),
        ]

    def _set_up_vllm_config(self, gpu_cache_blocks):
        self.env["VLLM_SERVER_DEV_MODE"] = "1"

        # Construct decoder command
        self.decoder_cmd = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            os.environ.get("KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            "--block-size",
            "16",
            "--max-seq-len",
            "8000",  # required to fit on L4 GPU when using 8b model
            "--connector",
            "nixl",
        ]

        # Construct prefiller command
        self.prefiller_cmd = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            os.environ.get("KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            "--is-prefill-worker",
            "--block-size",
            "16",
            "--max-seq-len",
            "8000",  # required to fit on L4 GPU when using 8b model
            "--connector",
            "kvbm",
            "nixl",
        ]

        # GPU blocks override
        if gpu_cache_blocks is not None:
            self.decoder_cmd.extend(
                ["--num-gpu-blocks-override", str(gpu_cache_blocks)]
            )
            self.prefiller_cmd.extend(
                ["--num-gpu-blocks-override", str(gpu_cache_blocks)]
            )

    def start_server(self, timeout: int = 300) -> bool:
        """Start LLM server and wait for readiness."""
        if self.is_server_running():
            self.stop_server()
            time.sleep(5)

        # Open log files
        self.prefiller_stdout_file = open(
            self.prefiller_log_file.with_suffix(".stdout.log"), "w"
        )
        self.prefiller_stderr_file = open(
            self.prefiller_log_file.with_suffix(".stderr.log"), "w"
        )
        if self.prefiller_stdout_file is not None:
            self.prefiller_stdout_file.write(
                f"=== {self.server_type} Prefiller Started at {datetime.now()} ===\nCommand: {' '.join(self.prefiller_cmd)}\n"
            )
            self.prefiller_stdout_file.flush()

        self.decoder_stdout_file = open(
            self.decoder_log_file.with_suffix(".stdout.log"), "w"
        )
        self.decoder_stderr_file = open(
            self.decoder_log_file.with_suffix(".stderr.log"), "w"
        )
        if self.decoder_stdout_file is not None:
            self.decoder_stdout_file.write(
                f"=== {self.server_type} Decoder Started at {datetime.now()} ===\nCommand: {' '.join(self.decoder_cmd)}\n"
            )
            self.decoder_stdout_file.flush()

        # Create separate environment configs for different processes
        decoder_env = self.env.copy()
        decoder_env["CUDA_VISIBLE_DEVICES"] = "0"

        prefiller_env = self.env.copy()
        prefiller_env["CUDA_VISIBLE_DEVICES"] = "1"

        # Launch frontend first
        self.process_frontend = subprocess.Popen(
            self.dynamo_frontend_cmd,
            env=self.env,
            preexec_fn=os.setsid,
        )
        print(f"Frontend process started with PID: {self.process_frontend.pid}")

        # Give frontend time to start up
        time.sleep(5)

        # Launch decoder
        self.process_decoder = subprocess.Popen(
            self.decoder_cmd,
            stdout=self.decoder_stdout_file,
            stderr=self.decoder_stderr_file,
            env=decoder_env,
            preexec_fn=os.setsid,
        )
        print(f"Decoder process started with PID: {self.process_decoder.pid}")

        # Give decoder time to start up
        time.sleep(5)

        # Launch prefiller
        self.process_prefiller = subprocess.Popen(
            self.prefiller_cmd,
            stdout=self.prefiller_stdout_file,
            stderr=self.prefiller_stderr_file,
            env=prefiller_env,
            preexec_fn=os.setsid,
        )
        print(f"Prefiller process started with PID: {self.process_prefiller.pid}")

        # Give prefiller time to start up
        time.sleep(5)

        # Wait for health
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.is_server_running():
                    return True
                if (
                    self.process_frontend.poll() is not None
                    or self.process_prefiller.poll() is not None
                    or self.process_decoder.poll() is not None
                ):
                    self._close_log_files()
                    return False
            except Exception as e:
                print(f"Error checking server status: {e}")

            print("Waiting for server to start up:")
            print(f"timeout: {timeout}, elapsed: {int(time.time() - start_time)}")
            time.sleep(5)

        # Timeout
        self.stop_server()
        return False

    def stop_server(self):
        """Stop LLM server and close logs."""
        if self.process_frontend:
            try:
                os.killpg(os.getpgid(self.process_frontend.pid), signal.SIGTERM)
                try:
                    self.process_frontend.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.process_frontend.pid), signal.SIGKILL)
                    self.process_frontend.wait()
            except (ProcessLookupError, OSError):
                pass
            finally:
                self.process_frontend = None
        if self.process_prefiller:
            try:
                os.killpg(os.getpgid(self.process_prefiller.pid), signal.SIGTERM)
                try:
                    self.process_prefiller.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.process_prefiller.pid), signal.SIGKILL)
                    self.process_prefiller.wait()
            except (ProcessLookupError, OSError):
                pass
            finally:
                self.process_prefiller = None
        if self.process_decoder:
            try:
                os.killpg(os.getpgid(self.process_decoder.pid), signal.SIGTERM)
                try:
                    self.process_decoder.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.process_decoder.pid), signal.SIGKILL)
                    self.process_decoder.wait()
            except (ProcessLookupError, OSError):
                pass
            finally:
                self.process_decoder = None
        self._close_log_files()

    def _close_log_files(self):
        if self.prefiller_stdout_file:
            self.prefiller_stdout_file.write(
                f"\n=== Prefiller Stopped at {datetime.now()} ===\n"
            )
            self.prefiller_stdout_file.close()
            self.prefiller_stdout_file = None
        if self.prefiller_stderr_file:
            self.prefiller_stderr_file.close()
            self.prefiller_stderr_file = None

        if self.decoder_stdout_file:
            self.decoder_stdout_file.write(
                f"\n=== Decoder Stopped at {datetime.now()} ===\n"
            )
            self.decoder_stdout_file.close()
            self.decoder_stdout_file = None
        if self.decoder_stderr_file:
            self.decoder_stderr_file.close()
            self.decoder_stderr_file = None

    def is_server_running(self) -> bool:
        try:
            # First check basic health
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                return False

            # Then check if the model endpoint is ready with a simple test request
            test_payload = {
                "model": os.environ.get(
                    "KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
                ),
                "messages": [{"role": "user", "content": "test"}],
                "max_completion_tokens": 1,
                "temperature": 0,
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=test_payload,
                timeout=10,
            )
            return response.status_code == 200

        except requests.exceptions.RequestException:
            return False


class DisaggDeterminismTester(DeterminismTester):
    """Disaggregated architecture specific determinism tester."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        super().__init__(base_url, model_id, server_type)

    def reset_prefix_cache(self):
        """Reset the prefix cache."""
        print("Resetting prefix cache...")
        # 150 shakespeare requests (each request is 200 words, and roughly 17 blocks) could evict 150 * 17 = 2550 blocks
        shakespeare_count = 150
        for seq_idx in range(1, shakespeare_count + 1):
            start_word = (seq_idx - 1) * self.word_count
            content = self.get_shakespeare_content(start_word)

            if content:
                print(
                    f"Resetting Shakespeare sequence {seq_idx} (words {start_word}-{start_word + self.word_count - 1})..."
                )
                try:
                    self.make_request(content)
                except Exception as e:
                    print(f"Resetting request failed: {e}")
        print("Cache reset done")


@pytest.fixture(scope="function")
def llm_server(request, runtime_services):
    """Start and stop a LLM server for each test with optional cache block overrides.

    To parametrize, use:
      @pytest.mark.parametrize("llm_server", [{"cpu_blocks": 10000, "gpu_blocks": 1000}], indirect=True)
    """
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    cpu_blocks = getattr(request, "param", {}).get("cpu_blocks", None)
    gpu_blocks = getattr(request, "param", {}).get("gpu_blocks", None)
    port = getattr(request, "param", {}).get("port", None)

    # Put logs in the per-test directory set up by tests/conftest.py
    log_dir = Path(request.node.name)

    if importlib.util.find_spec("vllm") is not None:
        server_type = ServerType.vllm
    else:
        raise Exception("vllm module is not available in the current environment.")

    server_manager = LLMServerManager(
        port=port,
        cpu_cache_blocks=cpu_blocks,
        gpu_cache_blocks=gpu_blocks,
        log_dir=log_dir,
        server_type=server_type,
    )

    start_timeout = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "300"))
    if not server_manager.start_server(timeout=start_timeout):
        pytest.fail(
            f"Failed to start {server_type} server (cpu_blocks={cpu_blocks}, gpu_blocks={gpu_blocks}, port={server_manager.port})"
        )

    yield server_manager

    server_manager.stop_server()


@pytest.fixture(scope="function")
def tester(llm_server):
    """Create determinism tester bound to the running server's base URL."""
    t = DisaggDeterminismTester(
        base_url=llm_server.base_url, server_type=llm_server.server_type
    )
    t.download_shakespeare_text()
    return t


class TestDeterminism(BaseTestDeterminism):
    """Test class for determinism validation."""

    @pytest.mark.parametrize(
        "llm_server",
        [
            {
                "cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000")),
                "gpu_blocks": int(os.environ.get("KVBM_GPU_BLOCKS", "1000")),
            },
        ],
        indirect=True,
    )
    def test_determinism_with_cache_reset(self, tester, llm_server, runtime_services):
        """Test determinism across cache reset: run test with warmup, reset cache, run again without warmup."""
        # Call the base class implementation
        super().test_determinism_with_cache_reset(tester, llm_server, runtime_services)

    @pytest.mark.parametrize(
        "llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "20000"))},
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "num_concurrent",
        [int(x) for x in os.environ.get("KVBM_CONCURRENT_REQUESTS", "3").split(",")],
    )
    @pytest.mark.parametrize(
        "max_tokens",
        [int(x) for x in os.environ.get("KVBM_MAX_TOKENS", "10").split(",")],
    )
    @pytest.mark.parametrize(
        "num_prompts",
        [int(x) for x in os.environ.get("KVBM_IFEVAL_PROMPTS", "120").split(",")],
    )
    @pytest.mark.skip(reason="Flaky test: DIS-665")
    def test_concurrent_determinism_with_ifeval(
        self,
        tester,
        llm_server,
        runtime_services,
        num_concurrent,
        max_tokens,
        num_prompts,
    ):
        """Simple concurrent determinism test: send IFEval prompts concurrently, with cache reset."""
        print("\n" + "=" * 70)
        print("CONCURRENT DETERMINISM TEST WITH IFEVAL")
        print("=" * 70)

        # Override max_tokens for this test iteration
        original_max_tokens = os.environ.get("KVBM_MAX_TOKENS")
        os.environ["KVBM_MAX_TOKENS"] = str(max_tokens)
        print(
            f"Using KVBM_MAX_TOKENS={max_tokens} (parametrized, original: {original_max_tokens or '48'})"
        )

        # Configuration comes from parametrize
        print(
            f"Configuration: {num_concurrent} concurrent requests, {max_tokens} max tokens"
        )

        # Load IFEval prompts
        ifeval_prompts = tester.download_ifeval_dataset()
        if not ifeval_prompts:
            pytest.skip("IFEval dataset not available")

        # Use parametrized number of IFEval prompts
        test_prompts = ifeval_prompts[:num_prompts]
        print(
            f"Using {len(test_prompts)} IFEval prompts for concurrent testing (parametrized: {num_prompts})"
        )
        print(f"Concurrency level: {num_concurrent} simultaneous requests")

        # Show sample prompts
        print("\nSample prompts:")
        for i, prompt in enumerate(test_prompts[:3]):
            print(f"  {i+1}. {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        if len(test_prompts) > 3:
            print(f"  ... and {len(test_prompts) - 3} more")

        def run_concurrent_test(phase_name, do_warmup=False):
            """Run one phase of concurrent testing."""
            print(f"\n=== {phase_name} ===")

            if do_warmup:
                # KV Cache warmup - send ALL test prompts to compute KV caches
                print(
                    f"Warming up KV caches with all {len(test_prompts)} test prompts..."
                )
                warmup_failed = 0

                for i, prompt in enumerate(test_prompts):
                    if (
                        i % 5 == 0 or i == len(test_prompts) - 1
                    ):  # Progress every 5 prompts
                        print(f"  Warmup progress: {i+1}/{len(test_prompts)}")

                    try:
                        tester.make_request(prompt)
                    except Exception as e:
                        warmup_failed += 1
                        if warmup_failed <= 3:  # Show first few failures
                            print(f"    Warmup failed for prompt {i}: {e}")

                if warmup_failed > 0:
                    print(
                        f"Warmup completed with {warmup_failed} failures out of {len(test_prompts)} prompts"
                    )
                else:
                    print(
                        f"Warmup completed successfully - all {len(test_prompts)} KV caches computed"
                    )

                # Wait for 10 seconds to make sure all transfers are complete
                time.sleep(10)
            else:
                print("Skipping warmup (already done in previous phase)")

            # Run concurrent requests
            print(
                f"Sending {len(test_prompts)} requests with {num_concurrent} max concurrent..."
            )
            start_time = time.time()

            def make_request_wrapper(prompt_and_idx):
                idx, prompt = prompt_and_idx
                try:
                    response = tester.make_request(prompt)
                    return {
                        "idx": idx,
                        "prompt": prompt,
                        "response": response,
                        "success": True,
                    }
                except Exception as e:
                    return {
                        "idx": idx,
                        "prompt": prompt,
                        "error": str(e),
                        "success": False,
                    }

            # Execute all requests concurrently
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                results = list(
                    executor.map(make_request_wrapper, enumerate(test_prompts))
                )

            elapsed = time.time() - start_time
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            print(
                f"Completed in {elapsed:.2f}s - Success: {len(successful)}, Failed: {len(failed)}"
            )

            if failed:
                for fail in failed[:3]:  # Show first few failures
                    print(f"  Failed: {fail['error']}")

            return successful

        # Phase 1: Before cache reset
        results_before = run_concurrent_test(
            "PHASE 1: BEFORE CACHE RESET", do_warmup=True
        )

        # Reset cache
        print("\n" + "=" * 50)
        print("RESETTING CACHE")
        print("=" * 50)
        tester.reset_prefix_cache()

        # Phase 2: After cache reset
        results_after = run_concurrent_test("PHASE 2: AFTER CACHE RESET")

        # Compare results between phases
        print("\n" + "=" * 70)
        print("DETERMINISM ANALYSIS")
        print("=" * 70)

        # Create lookup for before results
        before_responses = {r["idx"]: r["response"] for r in results_before}
        after_responses = {r["idx"]: r["response"] for r in results_after}

        deterministic_count = 0
        total_compared = 0

        for idx in before_responses:
            if idx in after_responses:
                total_compared += 1
                before_resp = before_responses[idx]
                after_resp = after_responses[idx]

                if before_resp == after_resp:
                    deterministic_count += 1
                    print(f"   Prompt {idx}: DETERMINISTIC")
                else:
                    print(f"   Prompt {idx}: NON-DETERMINISTIC")
                    print(f"     Before: {before_resp}")
                    print(f"     After:  {after_resp}")

        # Final assessment
        success_rate = deterministic_count / total_compared if total_compared > 0 else 0
        print("\n=== FINAL RESULT ===")
        print(f"Prompts compared: {total_compared}")
        print(f"Deterministic: {deterministic_count}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Concurrent requests: {num_concurrent}")

        # Restore original max_tokens setting
        if original_max_tokens is not None:
            os.environ["KVBM_MAX_TOKENS"] = original_max_tokens
        else:
            os.environ.pop("KVBM_MAX_TOKENS", None)

        assert (
            success_rate == 1.0
        ), f"Determinism failed: {deterministic_count}/{total_compared} prompts deterministic"


if __name__ == "__main__":
    # Allow running as script
    pytest.main([__file__, "-v", "-s"])
