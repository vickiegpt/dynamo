from .planner_connector import PlannerConnector
from .kube import KubernetesAPI

class KubernetesConnector(PlannerConnector):
    def __init__(self):
        self.kube_api = KubernetesAPI()

    async def add_component(self, component_name: str):
        """Add a component by increasing its replica count to 1"""
        deployment = await self.kube_api.get_graph_deployment(component_name)
        if deployment is None:
            raise ValueError(f"Graph not found for component {component_name} in dynamo namespace {self.namespace}")
        # get current replicas or 1 if not found
        current_replicas = self._get_current_replicas(deployment, component_name)
        await self.kube_api.update_graph_replicas(self._get_graph_deployment_name(deployment), component_name, current_replicas + 1)

    async def remove_component(self, component_name: str):
        """Remove a component by setting its replica count to 0"""
        deployment = await self.kube_api.get_graph_deployment(component_name, self.namespace)
        if deployment is None:
            raise ValueError(f"Graph {component_name} not found for namespace {self.namespace}")
        # get current replicas or 1 if not found
        current_replicas = self._get_current_replicas(deployment, component_name)
        if current_replicas > 0:
            await self.kube_api.update_graph_replicas(self._get_graph_deployment_name(deployment), component_name, current_replicas - 1)
    
    def _get_current_replicas(self, deployment: dict, component_name: str) -> int:
        """Get the current replicas for a component in a graph deployment"""
        return deployment['spec']['services'][component_name]['replicas'] if 'replicas' in deployment['spec']['services'][component_name] else 1

    def _get_graph_deployment_name(self, deployment: dict) -> str:
        """Get the name of the graph deployment"""
        return deployment['metadata']['name']