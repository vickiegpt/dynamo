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

from dynamo.runtime import DistributedRuntime, dynamo_worker, dynamo_endpoint, Component
from pydantic import BaseModel
from typing import List, Any, Dict
import asyncio
import logging
import inspect
logger = logging.getLogger(__name__)

from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk.cli.utils import configure_target_environment
from dynamo.sdk.core.runner import TargetEnum
from dynamo.sdk.core.protocol.interface import DynamoTransport

# Use Dynamo target (this is the only supported one)
configure_target_environment(TargetEnum.DYNAMO)


class DynamoContext(BaseModel):
    """Context object for the service"""
    class Config:
        arbitrary_types_allowed = True
    runtime: DistributedRuntime
    component: Component
    endpoints: Dict[str, Any]
    name: str
    namespace: str

async def serve(service, *args, **kwargs):
    # 1. Create a runtime/dyn worker if one is not passed
    # 2. Init the context
    # 3. Init the inner class injecting the context
    # 4. Run async init to do any async setup
    # 5. Serve the decorated endpoints of the component

    @dynamo_worker()
    async def worker(runtime: DistributedRuntime):
        # Create the service
        namespace, name = service.dynamo_address()
        component = runtime.namespace(namespace).component(name)
        logger.info(f"Registering component {namespace}/{name}")
        await component.create_service()

        # 2. Declare the endpoints on the component that use DynamoTransport.DEFAULT (NATS based)
        drt_endpoints = [ep for ep in service.get_dynamo_endpoints().values() if DynamoTransport.DEFAULT in ep.transports]
        endpoints = {ep.name: component.endpoint(ep.name) for ep in drt_endpoints}

        # 3. init a pydantic model with the runtime, component, endpoints, name, namespace
        dynamo_context = DynamoContext(
            runtime=runtime,
            component=component,
            endpoints=endpoints,
            name=name,
            namespace=namespace,
        )

        # 4. Init the inner class injecting the context and other args and kwargs passed by the user
        if _should_inject_dynamo_context(service):
            inner_instance = service.inner(dynamo_context, *args, **kwargs)
        else:
            inner_instance = service.inner(*args, **kwargs)

        # 5. Get and run async init if it exists
        async_init = get_async_init(inner_instance)
        if async_init:
            logger.info(f"Running async init for {inner_instance.__class__.__name__}")
            await async_init()
    
        # 6. Finally serve each endpoint
        handlers = get_ep_handlers(drt_endpoints, inner_instance)
        ep_2_handler = {endpoints[ep_name]: handlers[ep_name] for ep_name in endpoints}
        tasks = [ep.serve_endpoint(handler) for ep, handler in ep_2_handler.items()]
        await asyncio.gather(*tasks)
    
    await worker()


def get_async_init(instance):
    """Return the decorated async init method for the class"""
    for name, member in vars(instance.__class__).items():
        if callable(member) and getattr(member, "__dynamo_startup_hook__", False):
            return getattr(instance, name)
    return None

def get_ep_handlers(endpoints, inner_instance):
    """Get the endpoint handlers for the service"""
    ep_handlers = {}
    for endpoint in endpoints:
        # Binding the instance to the methods of the class
        bound_method = endpoint.func.__get__(inner_instance)
        ep_handlers[endpoint.name] = dynamo_endpoint(endpoint.request_type, Any)(bound_method)
    return ep_handlers

def _should_inject_dynamo_context(service):
    """Helper function to determine if dynamo_context should be injected. It is only injected if the inner class has a constructor that takes dynamo_context as the first argument.
    
    Args:
        service: The service instance
        
    Returns:
        bool: True if dynamo_context should be injected as first arg, False otherwise
    """
    sig = inspect.signature(service.inner.__init__)
    params = list(sig.parameters.keys())
    return len(params) > 1 and params[1] == "dynamo_context"