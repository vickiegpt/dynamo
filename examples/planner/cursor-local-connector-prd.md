# Product Requirements Document: LocalConnector for Dynamo Planner

## Overview

The LocalConnector is a critical component of the Dynamo Planner system that enables automatic scaling of Dynamo components in a local development environment. This document outlines the detailed requirements, APIs, and implementation strategy for the LocalConnector.

## Objectives

1. Create a clean, user-friendly API for managing local Dynamo components
2. Integrate with existing Dynamo serving infrastructure
3. Support automatic resource allocation for components
4. Provide comprehensive metrics collection
5. Enable scale up/down operations for components

## Target Users

- Dynamo developers working in local environments
- Data scientists testing scaling behaviors
- DevOps engineers validating scaling policies before production deployment

## Functional Requirements

### Core Functionality

1. **Component Discovery**
   - Discover all components in a specified namespace
   - Retrieve component metadata (name, endpoint, status)

2. **Component Scaling**
   - Scale components up by adding new instances
   - Scale components down by removing instances
   - Support granular scaling (add/remove specific number of replicas)

3. **Resource Management**
   - Allocate appropriate CPU/memory/GPU resources
   - Track resource usage per component
   - Prevent resource overallocation

4. **Metrics Collection**
   - Gather system metrics (CPU, memory, GPU usage)
   - Collect component-specific metrics
   - Monitor queue sizes and backpressure

5. **Process Management**
   - Start/stop component processes
   - Monitor process health
   - Restart failed processes if needed

### User-Friendly APIs

1. **Component Operations**
   - List all components with status
   - Get detailed component information
   - Restart components
   - View component logs

2. **Metrics Access**
   - Get comprehensive metrics for a component
   - Monitor prefill queue metrics
   - Track system-wide resource usage

3. **System Topology**
   - View component relationships
   - Understand data flow between components

## Technical Requirements

### Integration Points

1. **Circus Integration**
   - Use Circus for process management
   - Manage watchers for each component

2. **Dynamo Runtime**
   - Connect to DistributedRuntime
   - Query component state via etcd

3. **Resource Allocation**
   - Leverage ResourceAllocator for GPU assignment
   - Track port and socket usage

4. **Service Discovery**
   - Import services from Bento packages
   - Map components to service definitions

### Implementation Details

1. **Initialization and Cleanup**
   - Support async context manager pattern
   - Provide factory method for easy creation
   - Ensure proper resource cleanup

2. **Process Management**
   - Create and manage Unix domain sockets
   - Handle process lifecycle events
   - Capture process logs

3. **Error Handling**
   - Gracefully handle process failures
   - Provide detailed error information
   - Support automatic retry for critical operations

## API Design

### Class Overview

```python
class LocalConnector(PlannerConnector):
    # Core methods for scaling
    async def get_component_replicas(self, component_name: str) -> int
    async def scale_component(self, component_name: str, replicas: int) -> bool

    # Resource and metrics methods
    async def get_resource_usage(self, component_name: str) -> dict
    async def get_system_topology(self) -> dict

    # User-friendly APIs
    async def list_components(self) -> List[dict]
    async def get_component_logs(self, component_name: str, lines: int = 100) -> List[str]
    async def restart_component(self, component_name: str) -> bool
    async def get_component_metrics(self, component_name: str) -> dict
    async def get_prefill_queue_metrics(self) -> dict

    # Lifecycle methods
    async def initialize(self)
    async def shutdown(self)

    # Factory and context manager methods
    @classmethod
    async def create(cls, namespace: str, bento_identifier: str = ".", working_dir: Optional[str] = None)
    async def __aenter__(self)
    async def __aexit__(self, exc_type, exc_val, exc_tb)
```

## Implementation Phases

### Phase 1: Core Infrastructure

1. Implement basic connector structure
2. Integrate with Circus for process management
3. Connect to Dynamo runtime
4. Implement component discovery

### Phase 2: Scaling Operations

1. Implement scale up operation
2. Implement scale down operation
3. Add resource allocation logic
4. Handle process lifecycle events

### Phase 3: Metrics and User APIs

1. Implement metrics collection
2. Add user-friendly APIs
3. Support component logs
4. Implement topology discovery

### Phase 4: Testing and Optimization

1. Test with different component types
2. Optimize resource allocation
3. Add comprehensive error handling
4. Performance testing

## Discussion Points

As we begin implementation, let's discuss the following:

1. **Resource Allocation Strategy**
   - How should GPU resources be divided among components?
   - Should we implement dynamic resource allocation based on load?

2. **Metrics Collection**
   - What specific metrics are most important to collect?
   - How frequently should metrics be collected?

3. **Component Dependencies**
   - How should we handle scaling interdependent components?
   - Should we implement dependency-aware scaling?

4. **Failure Handling**
   - How should component failures be handled?
   - What recovery mechanisms should be implemented?

5. **Logging and Observability**
   - What level of logging is appropriate?
   - How can we make the connector observable for debugging?

## Next Steps

1. Review the proposed API design and provide feedback
2. Discuss the resource allocation strategy
3. Define metrics collection requirements in more detail
4. Outline testing strategy for the connector
5. Prioritize implementation phases

Let's continue the discussion to refine this PRD and move toward implementation.
