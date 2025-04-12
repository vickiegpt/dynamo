# ComputeGroups: Resource Abstraction for Dynamo Planner

## Overview

ComputeGroups is an abstraction over physical compute resources (primarily GPUs) that provides planner with a unified interface for resource management across different environments (local development and Kubernetes). It enables efficient allocation, tracking, and management of compute resources needed by Dynamo components.

## Problem Statement

Currently, resource management is tightly coupled with the specific environment:
- Local environment: Uses state files and circus controller for resource tracking
- Kubernetes: Uses K8s API for resource management

This creates inconsistencies in how resources are allocated, monitored, and scaled. We need a unified abstraction that works across environments while leveraging the strengths of each platform.

## Core Concepts

### ComputeGroup

A logical grouping of physical resources that can be allocated as a unit. A ComputeGroup may represent:
- A single GPU
- Multiple GPUs (potentially with topology awareness)
- A fractional GPU
- CPU-only compute resources

### ComputeNode

A physical or virtual machine that hosts compute resources. In local environments, this is the local machine. In Kubernetes, this could be a node in the cluster.

### ResourceState

The current allocation state of resources, including which components are using which resources and how much capacity is available.

## Core Requirements

### 1. Resource Discovery and Tracking

- Automatically discover available compute resources
- Track resource capabilities (GPU model, memory, etc.)
- Monitor resource utilization and health
- Support for topology-aware resource grouping (e.g., NVLink connected GPUs)

### 2. Resource Allocation

- Allocate resources to components based on requirements
- Support different allocation strategies (binpack, spread, etc.)
- Handle resource allocation atomicity and conflicts
- Support fractional resource allocation where applicable

### 3. Resource Lifecycle Management

- Initialize and cleanup resources
- Handle resource draining for graceful component migration
- Support resource reservation for planned scaling
- Manage resource state persistence

### 4. Environment Abstraction

- Provide consistent API across local and Kubernetes environments
- Abstract environment-specific details while utilizing native capabilities
- Support environment-specific extensions when needed

## API Design

```python
class ComputeGroup:
    """Represents a logical grouping of compute resources"""
    
    @property
    def id(self) -> str:
        """Unique identifier for this compute group"""
        
    @property
    def resources(self) -> Dict[str, Any]:
        """Resources in this compute group (types, quantities)"""
        
    @property
    def topology(self) -> Dict[str, Any]:
        """Topology information for the resources in this group"""
        
    @property
    def utilization(self) -> Dict[str, float]:
        """Current utilization metrics for resources in this group"""
        
    @property
    def allocation_state(self) -> str:
        """Current allocation state (free, allocated, draining, etc.)"""
        
    @property
    def component(self) -> Optional[str]:
        """Component currently using this compute group, if any"""


class ComputeNode:
    """Represents a physical or virtual machine with compute resources"""
    
    @property
    def id(self) -> str:
        """Unique identifier for this compute node"""
        
    @property
    def compute_groups(self) -> List[ComputeGroup]:
        """Compute groups available on this node"""
        
    @property
    def resources(self) -> Dict[str, Any]:
        """Total resources available on this node"""
        
    @property
    def utilization(self) -> Dict[str, float]:
        """Current utilization metrics for this node"""


class ComputeGroupManager:
    """Manages compute groups across the system"""
    
    async def list_compute_nodes(self) -> List[ComputeNode]:
        """List all compute nodes in the system"""
        
    async def list_compute_groups(self, filters: Dict[str, Any] = None) -> List[ComputeGroup]:
        """List compute groups, optionally filtered by criteria"""
        
    async def get_compute_group(self, group_id: str) -> ComputeGroup:
        """Get a specific compute group by ID"""
        
    async def create_compute_group(self, resources: Dict[str, Any], node_id: Optional[str] = None) -> ComputeGroup:
        """Create a new compute group with specified resources"""
        
    async def delete_compute_group(self, group_id: str) -> bool:
        """Delete a compute group"""
        
    async def allocate_compute_group(self, component_name: str, resource_requirements: Dict[str, Any], 
                                     constraints: Dict[str, Any] = None) -> ComputeGroup:
        """Allocate a compute group to a component with specific requirements"""
        
    async def deallocate_compute_group(self, group_id: str) -> bool:
        """Deallocate a compute group, making it available for reallocation"""
        
    async def drain_compute_group(self, group_id: str) -> bool:
        """Mark a compute group as draining, preventing new allocations"""
        
    async def get_allocation_summary(self) -> Dict[str, Any]:
        """Get a summary of current resource allocations"""
        
    async def get_component_allocations(self, component_name: str) -> List[ComputeGroup]:
        """Get all compute groups allocated to a specific component"""


class ComputeResourceProvider(ABC):
    """Abstract base class for environment-specific resource providers"""
    
    @abstractmethod
    async def discover_resources(self) -> Dict[str, Any]:
        """Discover available compute resources in the environment"""
        
    @abstractmethod
    async def allocate_resources(self, component_name: str, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources to a component"""
        
    @abstractmethod
    async def deallocate_resources(self, component_name: str, resource_ids: List[str]) -> bool:
        """Deallocate resources from a component"""
        
    @abstractmethod
    async def get_resource_state(self) -> Dict[str, Any]:
        """Get current state of all resources"""


class LocalComputeResourceProvider(ComputeResourceProvider):
    """Implementation for local development environment"""
    # Implementation details specific to local environment


class KubernetesComputeResourceProvider(ComputeResourceProvider):
    """Implementation for Kubernetes environment"""
    # Implementation details specific to Kubernetes
```

## Resource Allocation Workflow

1. Component registration: Components register with the planner, specifying resource requirements
2. Resource discovery: ComputeGroupManager discovers available resources through the provider
3. Resource matching: Manager finds suitable compute groups based on requirements
4. Allocation: Resources are allocated to the component
5. State update: Allocation state is updated
6. Monitoring: Resources are continuously monitored for utilization and health
7. Deallocation: Resources are released when no longer needed

## Environment-Specific Considerations

### Local Environment

- Resource discovery: Use system APIs to discover local GPUs
- State management: Persist allocation state to local file system
- Process management: Use circus controller for process management
- Resource isolation: Use CUDA_VISIBLE_DEVICES for GPU isolation

### Kubernetes Environment

- Resource discovery: Use Kubernetes API for node and resource discovery
- State management: Use Kubernetes custom resources or ConfigMaps
- Process management: Use Kubernetes deployments and pods
- Resource isolation: Use Kubernetes device plugins and resource limits

## Advanced Features

### Topology-Aware Allocation

- Group GPUs based on PCIe topology or NVLink connectivity
- Prioritize allocation of GPUs with high-bandwidth connectivity
- Support for NUMA-aware allocations

### Resource Reservation

- Allow compute groups to be reserved for future allocation
- Support for planned scaling operations
- Handle reservation timeouts and conflicts

### Resource Pooling

- Create logical pools of resources with different characteristics
- Allow components to request resources from specific pools
- Enable pool quota management and priority allocation

### Smart Allocation Strategies

- Bin-packing: Minimize resource fragmentation
- Spreading: Distribute workload across nodes
- Priority-based: Allocate based on component priority
- Affinity/anti-affinity: Place related components together/apart

## Implementation Phases

### Phase 1: Core Abstraction

1. Define core interfaces and data models
2. Implement basic resource discovery for local environment
3. Implement basic allocation/deallocation APIs
4. Create simple state persistence mechanism

### Phase 2: Enhanced Local Support

1. Add topology awareness for local GPU discovery
2. Implement monitoring and health checking
3. Enhance allocation strategies
4. Add support for fractional GPU allocation

### Phase 3: Kubernetes Integration

1. Implement Kubernetes resource provider
2. Add support for Kubernetes-specific features
3. Ensure consistent behavior across environments
4. Implement cross-environment testing

### Phase 4: Advanced Features

1. Add topology-aware allocation
2. Implement resource reservation
3. Add resource pooling
4. Enhance allocation strategies

## Success Criteria

1. Unified API works consistently across local and Kubernetes environments
2. Resource utilization is optimized for different workloads
3. Component scaling operations handle resource allocation correctly
4. System recovers gracefully from failures
5. Resource state is accurately tracked and persisted
6. Performance overhead of abstraction is minimal 