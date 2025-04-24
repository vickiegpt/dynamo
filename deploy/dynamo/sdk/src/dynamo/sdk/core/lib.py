from typing import Any, Dict, Optional, Type, TypeVar, Union

from dynamo.sdk.core.service.dependency import DependencyInterface
from dynamo.sdk.core.service.interface import (
    DynamoConfig,
    ServiceConfig,
    ServiceInterface,
    ServiceProvider,
)

T = TypeVar("T", bound=object)

#  Note: global service provider.
# this should be set to a concrete implementation of the ServiceProvider interface
# TODO: bis: refactor to use dependency injection
_service_provider: ServiceProvider


def set_service_provider(provider: ServiceProvider) -> None:
    """Set the global service provider implementation"""
    global _service_provider
    _service_provider = provider


def get_service_provider() -> ServiceProvider:
    """Get the current service provider implementation"""
    return _service_provider


def service(
    inner: Optional[Type[T]] = None,
    /,
    *,
    dynamo: Optional[Union[Dict[str, Any], DynamoConfig]] = None,
    **kwargs: Any,
) -> Any:
    """Service decorator that's adapter-agnostic"""
    config = ServiceConfig(kwargs)

    # Parse dict into DynamoConfig object
    dynamo_config: Optional[DynamoConfig] = None
    if dynamo is not None:
        if isinstance(dynamo, dict):
            dynamo_config = DynamoConfig(**dynamo)
        else:
            dynamo_config = dynamo

    def decorator(inner: Type[T]) -> ServiceInterface[T]:
        provider = get_service_provider()
        return provider.create_service(
            service_cls=inner, config=config, dynamo_config=dynamo_config, **kwargs
        )

    return decorator(inner) if inner is not None else decorator


def depends(
    on: Optional[ServiceInterface[T]] = None, **kwargs: Any
) -> DependencyInterface[T]:
    """Create a dependency using the current service provider"""
    provider = get_service_provider()
    return provider.create_dependency(on=on, **kwargs)
