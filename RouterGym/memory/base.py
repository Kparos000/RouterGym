"""Base memory interface."""

from __future__ import annotations

from typing import Any


class MemoryBase:
    """Base memory abstract class."""

    def add(self, item: Any) -> None:
        """Add an item to memory."""
        raise NotImplementedError

    def get_context(self) -> str:
        """Return a context string for prompting."""
        raise NotImplementedError


__all__ = ["MemoryBase"]
