from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar

T = TypeVar("T", bound=object)


class ServiceConfig(Dict[str, Any]):
    """Base service configuration that can be extended by adapters"""

    pass


class EndpointInterface(ABC):
    """Generic interface for service endpoints"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this endpoint"""
        pass

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the endpoint implementation"""
        pass


class ServiceInterface(Generic[T], ABC):
    """Generic interface for service implementations"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the service name"""
        pass

    @property
    @abstractmethod
    def config(self) -> ServiceConfig:
        """Get the service configuration"""
        pass

    @property
    @abstractmethod
    def inner(self) -> Type[T]:
        """Get the inner service implementation class"""
        pass

    @abstractmethod
    def get_endpoints(self) -> Dict[str, EndpointInterface]:
        """Get all registered endpoints"""
        pass

    @abstractmethod
    def get_endpoint(self, name: str) -> EndpointInterface:
        """Get a specific endpoint by name"""
        pass

    @abstractmethod
    def list_endpoints(self) -> List[str]:
        """List names of all registered endpoints"""
        pass

    @abstractmethod
    def link(self, next_service: "ServiceInterface") -> "ServiceInterface":
        """Link this service to another service, creating a pipeline"""
        pass

    @abstractmethod
    def remove_unused_edges(self, used_edges: Set["ServiceInterface"]) -> None:
        """Remove unused dependencies"""
        pass


class DynamoConfig:
    """Configuration for Dynamo components"""

    def __init__(
        self,
        enabled: bool = False,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ):
        self.enabled = enabled
        self.name = name
        self.namespace = namespace
        # Store any additional configuration options
        for key, value in kwargs.items():
            setattr(self, key, value)


class ServiceProvider(ABC):
    """Interface for service provider implementations"""

    @abstractmethod
    def create_service(
        self,
        service_cls: Type[T],
        config: ServiceConfig,
        dynamo_config: Optional[DynamoConfig] = None,
        **kwargs,
    ) -> ServiceInterface[T]:
        """Create a service instance"""
        pass

    @abstractmethod
    def create_dependency(
        self, on: Optional[ServiceInterface[T]] = None, **kwargs
    ) -> "DependencyInterface[T]":
        """Create a dependency on a service"""
        pass
