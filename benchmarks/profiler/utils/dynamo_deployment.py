#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "PyYAML",
#   "aiofiles",
#   "kubernetes-asyncio",
# ]
# ///

import asyncio
import json
import os
import random
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
import aiofiles
import kubernetes_asyncio as kubernetes
from kubernetes_asyncio import client, config
from contextlib import asynccontextmanager

# Example chat completion request for testing deployments
EXAMPLE_CHAT_REQUEST = {
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
        {
            "role": "user",
            "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
        }
    ],
    "stream": False,
    "max_tokens": 30
}

class DynamoDeploymentClient:
    def __init__(self, namespace: str, deployment_name: str = "vllm-v1-agg", base_log_dir: Optional[str] = None):
        """
        Initialize the client with the namespace and deployment name.
        
        Args:
            namespace: The Kubernetes namespace
            deployment_name: Name of the deployment, defaults to vllm-v1-agg
            base_log_dir: Base directory for storing logs, defaults to ./logs if not specified
        """
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.components = []  # Will store component names from CR
        self.deployment_spec = None  # Will store the full deployment spec
        self.base_log_dir = Path(base_log_dir) if base_log_dir else Path("logs")
        
    async def _init_kubernetes(self):
        """Initialize kubernetes client"""
        await config.load_kube_config()
        self.k8s_client = client.ApiClient()
        self.custom_api = client.CustomObjectsApi(self.k8s_client)
        self.core_api = client.CoreV1Api(self.k8s_client)

    async def create_deployment(self, deployment: Union[dict, str]):
        """
        Create a DynamoGraphDeployment from either a dict or yaml file path.
        
        Args:
            deployment: Either a dict containing the deployment spec or a path to a yaml file
        """
        await self._init_kubernetes()
        
        if isinstance(deployment, str):
            # Load from yaml file
            async with aiofiles.open(deployment, 'r') as f:
                content = await f.read()
                self.deployment_spec = yaml.safe_load(content)
        else:
            self.deployment_spec = deployment
            
        # Extract component names
        self.components = [svc.lower() for svc in self.deployment_spec['spec']['services'].keys()]
        
        # Ensure name and namespace are set correctly
        self.deployment_spec['metadata']['name'] = self.deployment_name
        self.deployment_spec['metadata']['namespace'] = self.namespace
        
        try:
            await self.custom_api.create_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                body=self.deployment_spec
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                print(f"Deployment {self.deployment_name} already exists")
            else:
                raise

    async def wait_for_deployment_ready(self, timeout: int = 300):
        """
        Wait for the custom resource to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                status = await self.custom_api.get_namespaced_custom_object_status(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self.deployment_name
                )
                if status.get('status', {}).get('ready', False):
                    return True
            except kubernetes.client.rest.ApiException:
                pass
            await asyncio.sleep(5)
        raise TimeoutError("Deployment failed to become ready within timeout")

    @asynccontextmanager
    async def port_forward(self, port: Optional[int] = None):
        """
        Port forward the frontend service to local machine.
        
        Args:
            port: Local port to use. If None, uses a random port.
            
        Yields:
            The local port number being used
        """
        if port is None:
            port = random.randint(49152, 65535)
            
        service_name = f"{self.deployment_name}-frontend"
        cmd = f"kubectl port-forward service/{service_name} {port}:8000 -n {self.namespace}"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            # Wait briefly to ensure port-forward is established
            await asyncio.sleep(2)
            yield port
        finally:
            process.terminate()
            await process.wait()

    async def check_chat_completion(self):
        """
        Test the deployment with a chat completion request.
        """
        async with self.port_forward() as port:
            url = f"http://localhost:{port}/v1/chat/completions"
            
            cmd = f"""curl -X POST {url} \\
                     -H "Content-Type: application/json" \\
                     -d '{json.dumps(EXAMPLE_CHAT_REQUEST)}'"""
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return stdout.decode()

    async def get_deployment_logs(self):
        """
        Get logs from all pods in the deployment, organized by component.
        """
        # Create logs directory
        base_dir = self.base_log_dir / self.deployment_name
        base_dir.mkdir(parents=True, exist_ok=True)
        
        for component in self.components:
            component_dir = base_dir / component
            component_dir.mkdir(exist_ok=True)
            
            # List pods for this component
            label_selector = f"app={self.deployment_name}-{component}"
            pods = await self.core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector
            )
            
            # Get logs for each pod
            for i, pod in enumerate(pods.items):
                try:
                    logs = await self.core_api.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace=self.namespace
                    )
                    async with aiofiles.open(component_dir / f"replica_{i}.log", 'w') as f:
                        await f.write(logs)
                except kubernetes.client.rest.ApiException as e:
                    print(f"Error getting logs for pod {pod.metadata.name}: {e}")

    async def delete_deployment(self):
        """
        Delete the DynamoGraphDeployment CR.
        """
        try:
            await self.custom_api.delete_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                name=self.deployment_name
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                raise

async def main():
    # Example usage with custom log directory
    client = DynamoDeploymentClient(
        namespace="default",
        base_log_dir="/tmp/dynamo_logs"  # Example custom log directory
    )
    
    try:
        # Create deployment from yaml file
        await client.create_deployment("examples/vllm/deploy/agg.yaml")
        
        # Wait for deployment to be ready
        print("Waiting for deployment to be ready...")
        await client.wait_for_deployment_ready()
        print("Deployment is ready!")
        
        # Test chat completion
        print("Testing chat completion...")
        response = await client.check_chat_completion()
        print(f"Chat completion response: {response}")
        
        # Get logs
        print("Getting deployment logs...")
        await client.get_deployment_logs()
        print(f"Logs have been saved to {client.base_log_dir / client.deployment_name}!")
        
    finally:
        # Cleanup
        print("Cleaning up deployment...")
        await client.delete_deployment()
        print("Deployment deleted!")

if __name__ == "__main__":
    asyncio.run(main()) 