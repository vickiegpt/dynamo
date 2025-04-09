from abc import ABC, abstractmethod


class PlannerConnector(ABC):
    @abstractmethod
    async def get_component_replicas(self, component_name):
        """Get current number of replicas for a component"""
        pass

    @abstractmethod
    async def scale_component(self, component_name, replicas):
        """Scale a component to specified number of replicas"""
        pass

    @abstractmethod
    async def get_resource_usage(self, component_name):
        """Get resource usage for a component"""
        pass

    @abstractmethod
    async def get_system_topology(self):
        """Get system topology (components and their relationships)"""
        pass
