"""Base router interface."""

from __future__ import annotations

from typing import Any, Dict


class BaseRouter:
    """Base routing interface."""

    def route(self, ticket: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return routing decision metadata."""
        raise NotImplementedError("route() must be implemented by subclasses")


__all__ = ["BaseRouter"]
