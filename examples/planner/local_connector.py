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
