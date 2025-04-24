from functools import wraps
from typing import Any, Callable, Optional, TypeVar, get_type_hints

from dynamo.sdk.core.service.interface import EndpointInterface

T = TypeVar("T")


class Endpoint(EndpointInterface):
    """Base class for service endpoints"""

    def __init__(self, func: Callable, name: Optional[str] = None, **kwargs):
        self.func = func
        self._name = name or func.__name__
        # Extract request type from hints
        hints = get_type_hints(func)
        args = list(hints.items())

        # Skip self/cls argument if present
        if args and args[0][0] in ("self", "cls"):
            args = args[1:]

        # Get request type from first arg if available
        self.request_type = args[0][1] if args else None
        wraps(func)(self)

        # Store additional metadata
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def name(self) -> str:
        return self._name

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.func(*args, **kwargs)


def endpoint(name: Optional[str] = None, **kwargs) -> Callable[[Callable], Endpoint]:
    """Decorator for service endpoints."""

    def decorator(func: Callable) -> Endpoint:
        return Endpoint(func, name, **kwargs)

    return decorator
