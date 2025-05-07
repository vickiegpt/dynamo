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

from typing import Optional

from kubernetes import client, config


class KubernetesAPI:
    def __init__(self):
        # Load kubernetes configuration
        config.load_incluster_config()  # for in-cluster deployment

        self.custom_api = client.CustomObjectsApi()
        self.current_namespace = self._get_current_namespace()

    def _get_current_namespace(self) -> str:
        """Get the current namespace if running inside a k8s cluster"""
        try:
            with open(
                "/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r"
            ) as f:
                return f.read().strip()
        except FileNotFoundError:
            # Fallback to 'default' if not running in k8s
            return "default"

    async def get_graph_deployment(
        self, component_name: str, dynamo_namespace: str
    ) -> Optional[dict]:
        """Get DynamoGraphDeployment custom resource"""
        try:
            # Use label selector to find the deployment
            label_selector = f"nvidia.com/dynamo-namespace={dynamo_namespace}"
            response = self.custom_api.list_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.current_namespace,
                plural="dynamographdeployments",
                label_selector=label_selector,
            )

            items = response.get("items", [])
            if not items:
                return None
            if len(items) > 1:
                raise ValueError(
                    f"Multiple deployments found for component {component_name} in dynamo namespace {dynamo_namespace}. "
                    "Expected exactly one deployment."
                )
            return items[0]

        except client.ApiException as e:
            if e.status == 404:
                return None
            raise

    async def update_component_replicas(
        self, graph_deployment_name: str, component_name: str, replicas: int
    ) -> None:
        """Update the replicas count for a component in a DynamoGraphDeployment"""
        patch = {"spec": {"services": {component_name: {"replicas": replicas}}}}
        self.custom_api.patch_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=self.current_namespace,
            plural="dynamographdeployments",
            name=graph_deployment_name,
            body=patch,
        )
