from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from dynamo.sdk.core.service.interface import ServiceInterface

T = TypeVar("T", bound=object)


class DependencyInterface(Generic[T], ABC):
    """Generic interface for service dependencies"""

    @property
    @abstractmethod
    def on(self) -> Optional[ServiceInterface[T]]:
        """Get the service this dependency is on"""
        pass

    @abstractmethod
    def get(self, *args: Any, **kwargs: Any) -> Any:
        """Get the dependency client"""
        pass

    @abstractmethod
    async def get_endpoint(self, name: str) -> Any:
        """Get a specific endpoint from the service"""
        pass
