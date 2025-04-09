import logging

from planner_connector import PlannerConnector

from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_logger

configure_logger()
logger = logging.getLogger(__name__)


class LocalConnector(PlannerConnector):
    def __init__(self, namespace, runtime: DistributedRuntime):
        self.namespace = namespace
        self.runtime = runtime

    async def get_component_replicas(self, component_name):
        # Use local process counting or dynamo runtime API
        pass

    async def scale_component(self, component_name, replicas):
        # Use dynamo serve APIs to start/stop processes
        # This would tap into serving.py functionality
        pass

    async def get_resource_usage(self, component_name):
        # Query metrics from local components
        pass

    async def get_system_topology(self):
        # Get topology from local dynamo components
        pass
