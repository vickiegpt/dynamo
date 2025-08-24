# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import kubernetes
import psutil
import yaml
from kr8s.objects import Pod as kr8s_Pod
from kubernetes_asyncio import client, config


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
    def image(self) -> Optional[str]:
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
    def __init__(self, base: str, endpoint="/v1/chat/completions", port=8000):
        """Load the deployment YAML file"""
        with open(base, "r") as f:
            self._deployment_spec = yaml.safe_load(f)
        self._endpoint = endpoint
        self._port = port

    @property
    def name(self) -> str:
        """Deployment name"""
        return self._deployment_spec["metadata"]["name"]

    @property
    def port(self) -> int:
        """Deployment name"""
        return self._port

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @name.setter
    def name(self, value: str):
        self._deployment_spec["metadata"]["name"] = value

    @property
    def namespace(self) -> str:
        """Deployment namespace"""
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

    def set_image(self, image: str, service_name: Optional[str] = None):
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
    frontend_service_name: Optional[str] = "frontend"

    _custom_api = None
    _core_api = None
    _in_cluster = False
    _logger = logging.getLogger()
    _port_forward = None
    _deployment_name = None

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
        self._apps_v1 = client.AppsV1Api()

    async def _wait_for_pods(self, label, expected, timeout=60):
        for _ in range(timeout):
            pods = await self._core_api.list_namespaced_pod(
                self.namespace, label_selector=label
            )
            running = sum(
                1
                for pod in pods.items
                if any(
                    cond.type == "Ready" and cond.status == "True"
                    for cond in (pod.status.conditions or [])
                )
            )
            if running == expected:
                return True
            await asyncio.sleep(1)
        raise Exception(f"Didn't Reach Expected Pod Count {label}=={expected}")

    async def _scale_statfulset(self, name, label, replicas):
        body = {"spec": {"replicas": replicas}}
        await self._apps_v1.patch_namespaced_stateful_set_scale(
            name, self.namespace, body
        )
        await self._wait_for_pods(label, replicas)

    async def _restart_stateful(self, name, label):
        self._logger.info(f"Restarting {name} {label}")

        await self._scale_statfulset(name, label, 0)
        nats_pvc = await self._core_api.list_namespaced_persistent_volume_claim(
            self.namespace, label_selector=label
        )
        for pvc in nats_pvc.items:
            await self._core_api.delete_namespaced_persistent_volume_claim(
                pvc.metadata.name, self.namespace
            )

        await self._scale_statfulset(name, label, 1)

        self._logger.info(f"Restarted {name} {label}")

    async def _wait_for_ready(self, timeout: int = 1800):
        """
        Wait for the custom resource to be ready.

        Args:
            timeout: Maximum time to wait in seconds, default to 30 mins (image pulling can take a while)
        """
        start_time = time.time()
        # TODO: A little brittle, also should output intermediate status every so often.

        self._logger.info("Waiting for Deployment {self._deployment_name}")

        while (time.time() - start_time) < timeout:
            try:
                status = await self._custom_api.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
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

    async def _restart_nats(self):
        NATS_STS_NAME = "dynamo-platform-nats"
        NATS_LABEL = "app.kubernetes.io/component=nats"

        await self._restart_stateful(NATS_STS_NAME, NATS_LABEL)

    async def _restart_etcd(self):
        ETCD_STS_NAME = "dynamo-platform-etcd"
        ETCD_LABEL = "app.kubernetes.io/component=etcd"

        await self._restart_stateful(ETCD_STS_NAME, ETCD_LABEL)

    async def _create_deployment(self):
        """
        Create a DynamoGraphDeployment from either a dict or yaml file path.

        Args:
            deployment: Either a dict containing the deployment spec or a path to a yaml file
        """

        # Extract service names

        self._services = self.deployment_spec.services

        self._logger.info(
            f"Starting Deployment {self._deployment_name} with spec {self.deployment_spec}"
        )

        try:
            await self._custom_api.create_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                body=self.deployment_spec.spec(),
            )
            self._logger.info(f"Deployment Started {self._deployment_name}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.info(f"Deployment {self._deployment_name} already exists")
            else:
                self._logger.info(
                    f"Failed to create deployment {self._deployment_name}: {e}"
                )
                raise

    async def _get_deployment_logs(self):
        """
        Get logs from all pods in the deployment, organized by component.
        """
        # Create logs directory
        base_dir = self.log_dir
        os.makedirs(base_dir, exist_ok=True)

        for component in self.deployment_spec.services:
            component_dir = os.path.join(base_dir, component.name)
            os.makedirs(component_dir, exist_ok=True)

            # List pods for this component using the selector label
            # nvidia.com/selector: deployment-name-component
            label_selector = (
                f"nvidia.com/selector={self._deployment_name}-{component.name.lower()}"
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
            if self._deployment_name:
                await self._custom_api.delete_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
        except client.exceptions.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                raise

    def _start_port_forward(self):
        label_selector = f"nvidia.com/selector={self._deployment_name}-{self.frontend_service_name.lower()}"

        frontend_service_pod = kr8s_Pod.get(
            label_selector=label_selector, namespace=self.namespace
        )

        self._port_forward = frontend_service_pod.portforward(
            remote_port=self.deployment_spec.port,
            local_port=self.deployment_spec.port,
            address="0.0.0.0",
        )
        self._port_forward.start()

    def _stop_port_forward(self):
        print("stopping!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self._port_forward:
            self._port_forward.stop()

    async def _cleanup(self):
        try:
            await self._get_deployment_logs()
        finally:
            print("stopping!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self._stop_port_forward()
            await self._delete_deployment()

    async def __aenter__(self):
        try:
            self._logger = logging.getLogger(self.__class__.__name__)
            self.deployment_spec.namespace = self.namespace
            self._deployment_name = self.deployment_spec.name
            await self._init_kubernetes()
            await self._delete_deployment()
            await self._restart_etcd()
            await self._restart_nats()
            await self._create_deployment()
            await self._wait_for_ready()
            self._start_port_forward()

        except:
            await self._cleanup()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()


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
