"""Null memory backend."""

from typing import Any

from RouterGym.memory.base import MemoryBase


class NoneMemory(MemoryBase):
    """Memory backend that stores nothing."""

    def add(self, _: Any) -> None:
        """Discard inputs."""
        return None

    def get_context(self) -> str:
        """Return empty context."""
        return ""


__all__ = ["NoneMemory"]
