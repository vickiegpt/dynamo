# Dynamo SDK Service Providers

This directory contains service provider implementations for the Dynamo SDK.

## Available Providers

### BentoMLDeploymentTarget

The BentoML provider uses BentoML for service management and deployment. It's the default provider used by the Dynamo SDK for service orchestration and deployment.

### CircusDeploymentTarget

The Circus provider uses [Circus](https://circus.readthedocs.io/) for process management and service orchestration. Circus is a process and socket manager that can be used to monitor and control processes and sockets.

## Circus Provider Features

The CircusDeploymentTarget implements the DeploymentTarget interface and provides the following functionality:

- Process management using Circus's Arbiter and Watcher classes
- Socket-based communication between services
- Service dependency management
- Endpoint registration and routing

### Key Components

- **Arbiter**: Responsible for managing all watchers, ensuring processes run correctly
- **Watcher**: Manages the service processes (workers)
- **CircusSocket**: Communication mechanism between services
- **CircusService**: Implementation of the ServiceInterface using Circus
- **CircusDependency**: Implementation of DependencyInterface for Circus services

### Usage Example

```python
from dynamo.sdk.core.providers.circus import CircusDeploymentTarget
from dynamo.sdk.core.service.interface import ServiceConfig, DynamoConfig

# Create service provider
provider = CircusDeploymentTarget()

# Create services
backend_config = ServiceConfig({"workers": 1})
backend_dynamo = DynamoConfig(enabled=True, namespace="inference")

# Create service with the provider
backend_service = provider.create_service(
    BackendClass,
    backend_config,
    backend_dynamo,
    args=["-m", "backend_worker"],
    env_vars={"PYTHONPATH": os.getcwd()},
)

# Run all services with the circus arbiter
with provider.run_services() as arbiter:
    # Services are now running
    pass  # Your application logic here
```

See the full example in `examples/circus_example.py`.

## Extending with New Providers

To create a new service provider, implement the DeploymentTarget interface from `dynamo.sdk.core.service.interface`. The key methods to implement are:

- `create_service()`: Create a service instance
- `create_dependency()`: Create a dependency on a service

You'll also need to implement the ServiceInterface and DependencyInterface for your provider-specific implementation.
