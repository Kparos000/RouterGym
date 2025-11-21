"""Ticket utility abstractions for model prompts."""

from __future__ import annotations

from typing import Any, Dict


class Ticket:
    """Simple ticket container with prompt formatting."""

    def __init__(self, id: Any, text: str, category: str | None = None, metadata: Dict[str, Any] | None = None) -> None:
        self.id = id
        self.text = text
        self.category = category
        self.metadata = metadata or {}

    def to_prompt(self) -> str:
        """Format ticket content for model consumption."""
        header = f"[Ticket {self.id}]"
        category_line = f"Category: {self.category}" if self.category else ""
        meta_line = f"Metadata: {self.metadata}" if self.metadata else ""
        body = f"Text: {self.text}"
        parts = [header, category_line, meta_line, body]
        return "\n".join([p for p in parts if p])


__all__ = ["Ticket"]
