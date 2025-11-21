"""Null memory backend."""

from typing import Any


class NullMemory:
    """Memory backend that stores nothing."""

    def fetch(self, _: Any = None) -> str:
        """Return empty context."""
        return ""

    def store(self, _: Any) -> None:
        """Discard inputs."""
        return None
