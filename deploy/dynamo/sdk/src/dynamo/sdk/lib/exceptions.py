from __future__ import annotations

from http import HTTPStatus


class DynamoException(Exception):
    """Base class for all Dynamo SDK errors."""

    error_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_mapping = {}

    def __init_subclass__(cls) -> None:
        if "error_code" in cls.__dict__:
            cls.error_mapping[cls.error_code] = cls

    def __init__(self, message: str, error_code: HTTPStatus | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.error_code
