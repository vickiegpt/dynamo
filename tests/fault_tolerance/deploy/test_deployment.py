# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
import time
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Process

import pytest

from tests.fault_tolerance.deploy.client import client
from tests.fault_tolerance.deploy.parse_results import main as parse_results
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

multiprocessing.set_start_method("spawn")


# Each Deployment Graph contains
# the dynamo serve module and configuration as well
# as the endpoint for interaction

deployment_specs = {
    "agg-tp-1-dp-1": (
        DeploymentSpec("/workspace/components/backends/vllm/deploy/agg.yaml")
    ),
    "disagg-tp-1-dp-1": (
        DeploymentSpec("/workspace/components/backends/vllm/deploy/disagg.yaml")
    ),
}

deployment_specs["agg-tp-1-dp-2"] = DeploymentSpec(
    "/workspace/components/backends/vllm/deploy/agg.yaml"
)
deployment_specs["agg-tp-1-dp-2"]["Frontend"].replicas = 2
deployment_specs["agg-tp-1-dp-2"]["VllmDecodeWorker"].replicas = 2

deployment_specs["disagg-tp-1-dp-2"] = DeploymentSpec(
    "/workspace/components/backends/vllm/deploy/disagg.yaml"
)
deployment_specs["disagg-tp-1-dp-2"]["Frontend"].replicas = 2
deployment_specs["disagg-tp-1-dp-2"]["VllmDecodeWorker"].replicas = 2
deployment_specs["disagg-tp-1-dp-2"]["VllmPrefillWorker"].replicas = 2


@dataclass
class Failure:
    time: int
    pod_name: str
    command: str
    signal: str = "SIGINT"
    replicas: int = 1


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
    "frontend": [Failure(10, "Frontend", "dynamo.frontend")],
    "frontend_pod": [Failure(10, "Frontend", "delete_pod")],
    "decode_worker": [Failure(10, "VllmDecodeWorker", "dynamo.vllm")],
    "decode_worker_pod": [Failure(10, "VllmDecodeWorker", "delete pod")],
    "prefill_worker": [Failure(10, "VllmPrefillWorker", "dynamo.vllm")],
    "prefill_worker_pod": [Failure(10, "VllmPrefillWorker", "delete_pod")],
    "vllm_decode_engine_core": [
        Failure(10, "VllmDecodeWorker", "VLLM::EngineCore", "SIGKILL")
    ],
    "vllm_prefill_engine_core": [
        Failure(10, "VllmPrefillWorker", "VLLM::EngineCore", "SIGKILL")
    ],
    "none": [],
}


@pytest.fixture(params=failure_scenarios.keys())
def failures(request):
    return failure_scenarios[request.param]


@pytest.fixture(params=list(deployment_specs.keys()))
def deployment_spec(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    return deployment_specs[request.param]


@contextmanager
def _clients(
    logger,
    num_clients,
    request,
    deployment_spec,
    namespace,
    model,
    requests_per_client,
    input_token_length,
    output_token_length,
    max_retries,
    max_request_rate,
):
    procs = []
    for i in range(num_clients):
        procs.append(
            Process(
                target=client,
                args=(
                    deployment_spec,
                    namespace,
                    model,
                    request.node.name,
                    i,
                    requests_per_client,
                    input_token_length,
                    output_token_length,
                    max_retries,
                    max_request_rate,
                ),
            )
        )
        procs[-1].start()
    yield procs

    for proc in procs:
        logger.debug(f"{proc} waiting for join")
        proc.join()
        logger.debug(f"{proc} joined")


def _inject_failures(failures, logger, deployment: ManagedDeployment):  # noqa: F811
    for failure in failures:
        time.sleep(failure.time)

        pods = deployment.get_pods(failure.pod_name)[failure.pod_name]

        num_pods = len(pods)

        replicas = failure.replicas

        if not replicas:
            replicas = num_pods

        logger.info(f"Injecting failure for: {failure}")

        for x in range(replicas):
            pod = pods[x % num_pods]

            if failure.command == "delete_pod":
                deployment.get_pod_logs(failure.pod_name, pod, ".before_delete")
                pod.delete(force=True)
            else:
                processes = deployment.get_processes(pod)
                for process in processes:
                    if failure.command in process.command:
                        logger.info(
                            f"Terminating {failure.pod_name} Pid {process.pid} Command {process.command}"
                        )
                        process.kill(failure.signal)
                        process.wait()


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
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_fault_scenario(
    deployment_spec,  # noqa: F811
    request,
    image,
    namespace,
    num_clients,
    requests_per_client,
    model,
    failures,  # noqa: F811
    input_token_length,
    output_token_length,
    max_retries,
    max_request_rate,
):
    """
    Test dynamo serve deployments with injected failures
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)

    deployment_spec.disable_grove()

    deployment_spec.name = "fault-tolerance-test"

    if image:
        deployment_spec.set_image(image)

    if model:
        deployment_spec.set_model(model)
    else:
        model = deployment_spec["VllmDecodeWorker"].model

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
    ) as deployment:
        with _clients(
            logger,
            num_clients,
            request,
            deployment_spec,
            namespace,
            model,
            requests_per_client,
            input_token_length,
            output_token_length,
            max_retries,
            max_request_rate,
        ):
            _inject_failures(failures, logger, deployment)
