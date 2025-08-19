# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass

import kubernetes
import psutil
import yaml
from kubernetes_asyncio import client, config

benchmark_utils_path = os.path.join(
    os.environ.get("DYNAMO_HOME", "/workspace"), "benchmarks/profiler/utils"
)

sys.path.append(benchmark_utils_path)


def terminate_process(process, logger=logging.getLogger(), immediate_kill=False):
    try:
        logger.info("Terminating PID: %s name: %s", process.pid, process.name())
        if immediate_kill:
            logger.info("Sending Kill: %s %s", process.pid, process.name())
            process.kill()
        else:
            process.terminate()
    except psutil.AccessDenied:
        logger.warning("Access denied for PID %s", process.pid)
    except psutil.NoSuchProcess:
        logger.warning("PID %s no longer exists", process.pid)


def terminate_process_tree(
    pid, logger=logging.getLogger(), immediate_kill=False, timeout=10
):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            terminate_process(child, logger, immediate_kill)

        terminate_process(parent, logger, immediate_kill)

        for child in parent.children(recursive=True):
            try:
                child.wait(timeout)
            except psutil.TimeoutExpired:
                terminate_process(child, logger, immediate_kill=True)
        try:
            parent.wait(timeout)
        except psutil.TimeoutExpired:
            terminate_process(parent, logger, immediate_kill=True)

    except psutil.NoSuchProcess:
        # Process already terminated
        pass


class ServiceSpec:
    """Wrapper around a single service in the deployment spec."""

    def __init__(self, service_name: str, service_spec: dict):
        self._name = service_name
        self._spec = service_spec

    @property
    def name(self) -> str:
        """The service name (read-only)"""
        return self._name

    # ----- Image -----
    @property
    def image(self) -> str:
        """Container image for the service"""
        try:
            return self._spec["extraPodSpec"]["mainContainer"]["image"]
        except KeyError:
            return None

    @image.setter
    def image(self, value: str):
        if "extraPodSpec" not in self._spec:
            self._spec["extraPodSpec"] = {"mainContainer": {}}
        if "mainContainer" not in self._spec["extraPodSpec"]:
            self._spec["extraPodSpec"]["mainContainer"] = {}
        self._spec["extraPodSpec"]["mainContainer"]["image"] = value

    # ----- Replicas -----
    @property
    def replicas(self) -> int:
        return self._spec.get("replicas", 0)

    @replicas.setter
    def replicas(self, value: int):
        self._spec["replicas"] = value

    # ----- GPUs -----
    @property
    def gpus(self) -> int:
        try:
            return int(self._spec["resources"]["limits"]["gpu"])
        except KeyError:
            return 0

    @gpus.setter
    def gpus(self, value: int):
        if "resources" not in self._spec:
            self._spec["resources"] = {}
        if "limits" not in self._spec["resources"]:
            self._spec["resources"]["limits"] = {}
        self._spec["resources"]["limits"]["gpu"] = str(value)


class DeploymentSpec:
    def __init__(self, base: str):
        """Load the deployment YAML file"""
        with open(base, "r") as f:
            self._deployment_spec = yaml.safe_load(f)

    @property
    def name(self) -> str:
        """Deployment name"""
        return self._deployment_spec["metadata"]["name"]

    @name.setter
    def name(self, value: str):
        self._deployment_spec["metadata"]["name"] = value

    @property
    def namespace(self) -> str:
        """Deployment name"""
        return self._deployment_spec["metadata"]["namespace"]

    @namespace.setter
    def namespace(self, value: str):
        self._deployment_spec["metadata"]["namespace"] = value

    def disable_grove(self):
        if "annotations" not in self._deployment_spec["metadata"]:
            self._deployment_spec["metadata"]["annotations"] = {}
        self._deployment_spec["metadata"]["annotations"][
            "nvidia.com/enable-grove"
        ] = "false"

    def set_image(self, image, service_name=None):
        if service_name is None:
            services = self.services
        else:
            services = [self[service_name]]
        for service in services:
            service.image = image

    @property
    def services(self) -> list:
        """List of ServiceSpec objects"""
        return [
            ServiceSpec(svc, spec)
            for svc, spec in self._deployment_spec["spec"]["services"].items()
        ]

    def __getitem__(self, service_name: str) -> ServiceSpec:
        """Allow dict-like access: d['Frontend']"""
        return ServiceSpec(
            service_name, self._deployment_spec["spec"]["services"][service_name]
        )

    def spec(self):
        return self._deployment_spec

    def save(self, out_file: str):
        """Save updated deployment to file"""
        with open(out_file, "w") as f:
            yaml.safe_dump(self._deployment_spec, f, default_flow_style=False)


@dataclass
class ManagedDeployment:
    log_dir: str
    deployment_spec: DeploymentSpec
    namespace: str

    _custom_api = None
    _core_api = None
    _in_cluster = False
    _logger = logging.getLogger()

    async def _init_kubernetes(self):
        """Initialize kubernetes client"""
        try:
            # Try in-cluster config first (for pods with service accounts)
            await config.load_incluster_config()
            self._in_cluster = True
        except Exception:
            # Fallback to kube config file (for local development)
            await config.load_kube_config()

        k8s_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(k8s_client)
        self._core_api = client.CoreV1Api(k8s_client)

    async def _wait_for_ready(self, timeout: int = 1800):
        """
        Wait for the custom resource to be ready.

        Args:
            timeout: Maximum time to wait in seconds, default to 30 mins (image pulling can take a while)
        """
        start_time = time.time()
        # TODO: A little brittle, also should output intermediate status every so often.
        while (time.time() - start_time) < timeout:
            try:
                status = await self._custom_api.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self.deployment_name,
                )
                # Check both conditions:
                # 1. Ready condition is True
                # 2. State is successful
                status_obj = status.get("status", {})
                conditions = status_obj.get("conditions", [])
                current_state = status_obj.get("state", "unknown")

                self._logger.info(f"Current deployment state: {current_state}")
                self._logger.info(f"Current conditions: {conditions}")
                self._logger.info(
                    f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s"
                )

                ready_condition = False
                for condition in conditions:
                    if (
                        condition.get("type") == "Ready"
                        and condition.get("status") == "True"
                    ):
                        ready_condition = True
                        break

                state_successful = status_obj.get("state") == "successful"

                if ready_condition and state_successful:
                    self._logger.info(
                        "Deployment is ready: Ready condition is True and state is successful"
                    )
                    return True
                else:
                    self._logger.info(
                        f"Deployment not ready yet - Ready condition: {ready_condition}, State successful: {state_successful}"
                    )

            except kubernetes.client.rest.ApiException as e:
                self._logger.info(
                    f"API Exception while checking deployment status: {e}"
                )
                self._logger.info(f"Status code: {e.status}, Reason: {e.reason}")
            except Exception as e:
                self._logger.info(
                    f"Unexpected exception while checking deployment status: {e}"
                )
            await asyncio.sleep(20)
        raise TimeoutError("Deployment failed to become ready within timeout")

    async def _create_deployment(self):
        """
        Create a DynamoGraphDeployment from either a dict or yaml file path.

        Args:
            deployment: Either a dict containing the deployment spec or a path to a yaml file
        """
        await self._init_kubernetes()

        # Extract component names

        self._services = self.deployment_spec.services

        self.deployment_spec.namespace = self.namespace
        self.deployment_name = self.deployment_spec.name

        print(self.deployment_spec.spec())

        for k, v in self.deployment_spec.spec().items():
            if k == "extraPodSpec":
                print(v)

        try:
            await self._custom_api.create_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                body=self.deployment_spec.spec(),
            )
            self._logger.info(f"Successfully created deployment {self.deployment_name}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.info(f"Deployment {self.deployment_name} already exists")
            else:
                self._logger.info(
                    f"Failed to create deployment {self.deployment_name}: {e}"
                )
                raise

    async def _get_deployment_logs(self):
        """
        Get logs from all pods in the deployment, organized by component.
        """
        # Create logs directory
        base_dir = os.path.join(self.log_dir, self.deployment_name)
        os.makedirs(base_dir, exist_ok=True)

        for component in self.deployment_spec.services:
            component_dir = os.path.join(base_dir, component.name)
            os.makedirs(component_dir, exist_ok=True)

            # List pods for this component using the selector label
            # nvidia.com/selector: deployment-name-component
            label_selector = (
                f"nvidia.com/selector={self.deployment_name}-{component.name.lower()}"
            )

            pods = await self._core_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=label_selector
            )

            # Get logs for each pod
            for i, pod in enumerate(pods.items):
                try:
                    logs = await self._core_api.read_namespaced_pod_log(
                        name=pod.metadata.name, namespace=self.namespace
                    )
                    with open(os.path.join(component_dir, f"{i}.log"), "w") as f:
                        f.write(logs)
                except kubernetes.client.rest.ApiException as e:
                    print(f"Error getting logs for pod {pod.metadata.name}: {e}")

    async def _delete_deployment(self):
        """
        Delete the DynamoGraphDeployment CR.
        """
        try:
            await self._custom_api.delete_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                name=self.deployment_name,
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                raise

    async def __aenter__(self):
        try:
            self._logger = logging.getLogger(self.__class__.__name__)
            await self._init_kubernetes()
            await self._create_deployment()
            await self._wait_for_ready()
        except:
            await self._delete_deployment()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._get_deployment_logs()
        finally:
            await self._delete_deployment()


async def main():
    LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,  # ISO 8601 UTC format
    )

    deployment_spec = DeploymentSpec(
        "/workspace/components/backends/vllm/deploy/agg.yaml"
    )

    deployment_spec.disable_grove()

    print(deployment_spec._deployment_spec)

    deployment_spec.name = "foo"

    deployment_spec.set_image(
        "gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/dynamo:nnshah1-vllm-latest"
    )

    async with ManagedDeployment(
        namespace="nnshah1-test", log_dir=".", deployment_spec=deployment_spec
    ):
        pass
    #        time.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
