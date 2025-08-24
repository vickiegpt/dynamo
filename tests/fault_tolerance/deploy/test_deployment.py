# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import multiprocessing
from contextlib import contextmanager
from multiprocessing import Process

import psutil
import pytest

from tests.fault_tolerance.deploy.client import client
from tests.fault_tolerance.deploy.parse_results import main as parse_results
from tests.utils.deployment_graph import Payload
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

multiprocessing.set_start_method("spawn")


# Initial payload used for testing
# initial deployment readiness.

text_prompt = "Tell me a short joke about AI."

text_payload = Payload(
    payload_chat={
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {
                "role": "user",
                "content": text_prompt,  # Shorter prompt
            }
        ],
        "max_tokens": 150,
        "temperature": 0.1,
        #        "seed": 10,
        "ignore_eos": True,
        "min_tokens": 150,
        "stream": False,
    },
    expected_log=[],
    expected_response=["AI"],
)

# Each Deployment Graph contains
# the dynamo serve module and configuration as well
# as the endpoint for interaction


deployment_specs = {
    "agg-tp-1-dp-1": (
        DeploymentSpec("/workspace/components/backends/vllm/deploy/agg.yaml")
    )
}


# Each failure scenaro contains a list of failure injections
# Each failure injection has a time in seconds after the pervious injection and
# a list of failures to inject including the number of failures for each type.
# Failures are currently process termination.
#
# Example:
#
#   "prefill_worker": [[30, [("dynamo_prefillworker", 1)]]],
#
# terminates 1 prefill worker after 30 seconds

failure_scenarios = {
    "decode_worker": [[30, [("dynamo_vllmworker", 1)]]],
    "prefill_worker": [[30, [("dynamo_prefillworker", 1)]]],
    "frontend": [[30, [("dynamo_frontend", 1)]]],
    "processor": [[30, [("dynamo_processor", 1)]]],
    "vllm_worker": [[30, [("vllm_worker", 1)]]],
    "none": [],
}


@pytest.fixture(params=list(failure_scenarios.keys()))
def failures(request):
    return failure_scenarios[request.param]


@pytest.fixture(params=list(deployment_specs.keys()))
def deployment_spec_test(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    return deployment_specs[request.param]


# from tests.fault_tolerance.scenarios import (  # noqa: F401
#    deployment_graph_test,
#    failures,
# )


def _list_vllm_worker_processes():
    processes = []
    for ps_process in psutil.process_iter(["name", "cmdline"]):
        try:
            if "from multiprocessing.spawn import spawn_main;" in " ".join(
                ps_process.cmdline()
            ):
                processes.append(ps_process.pid)
        except Exception:
            pass
    return processes


@contextmanager
def _clients(
    logger,
    num_clients,
    request,
    deployment_spec,
    payload,
    requests_per_client,
    input_token_length,
    output_token_length,
    max_retries,
):
    procs = []
    for i in range(num_clients):
        procs.append(
            Process(
                target=client,
                args=(
                    deployment_spec,
                    payload,
                    request.node.name,
                    i,
                    requests_per_client,
                    input_token_length,
                    output_token_length,
                    max_retries,
                ),
            )
        )
        procs[-1].start()
    yield procs

    for proc in procs:
        logger.debug(f"{proc} waiting for join")
        proc.join()
        logger.debug(f"{proc} joined")


# def _inject_failures(failures, logger):  # noqa: F811
#     circus_controller = CircusController.from_state_file("dynamo")

#     for failure_time, component in failures:
#         time.sleep(failure_time)
#         for component_name, number in component:
#             logger.info(f"Injecting failure for: {component_name}")

#             if "dynamo" in component_name:
#                 result = circus_controller.client.call(
#                     {"command": "list", "properties": {"name": f"{component_name}"}}
#                 )
#                 if result["status"] == "error":
#                     logger.warning(f"component {component_name} not found {result}")
#                     continue

#                 num_processes = len(result["pids"])
#                 if number is None:
#                     number = num_processes
#                 for x in range(number):
#                     pid = result["pids"][x % num_processes]
#                     logger.info(f"Terminating {component_name} Pid {pid}")
#                     terminate_process_tree(pid, logger, immediate_kill=True)
#             elif "vllm" in component_name:
#                 vllm_processes = _list_vllm_worker_processes()
#                 num_processes = len(vllm_processes)
#                 if number is None:
#                     number = len(vllm_processes)
#                 for x in range(number):
#                     pid = vllm_processes[x % num_processes]
#                     terminate_process_tree(pid, logger, immediate_kill=True)

#     circus_controller.close()


global_result_list = []


@pytest.fixture(autouse=True)
def results_table(request):
    yield
    parse_results(logs_dir=None, log_paths=[request.node.name], tablefmt="fancy")
    global_result_list.append(request.node.name)


@pytest.fixture(autouse=True, scope="session")
def results_summary():
    yield
    parse_results(logs_dir=None, log_paths=global_result_list, tablefmt="fancy")


@pytest.mark.e2e
@pytest.mark.slow
async def test_fault_scenario(
    deployment_spec_test,  # noqa: F811
    request,
    image,
    namespace,
    num_clients,
    requests_per_client,
    #    failures,  # noqa: F811
    input_token_length,
    output_token_length,
    max_retries,
):
    """
    Test dynamo serve deployments with injected failures
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)

    deployment_spec_test.disable_grove()

    deployment_spec_test.name = "fault-tolerance-test"

    deployment_spec_test.set_image(image)

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=deployment_spec_test,
    ):
        await asyncio.sleep(10)

        print(multiprocessing.get_start_method())

        await asyncio.sleep(120)

        with _clients(
            logger,
            num_clients,
            request,
            deployment_spec_test,
            text_payload,
            requests_per_client,
            input_token_length,
            output_token_length,
            max_retries,
        ):
            pass


#             _inject_failures(failures, logger)


#        await asyncio.sleep(3000)
