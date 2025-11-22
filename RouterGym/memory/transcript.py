"""Transcript memory backend."""

from typing import List

from RouterGym.memory.base import MemoryBase


class TranscriptMemory(MemoryBase):
    """Keeps a running transcript of turns."""

    def __init__(self, k: int = 3) -> None:
        self.messages: List[str] = []
        self.k = k

    def add(self, message: str) -> None:
        """Append a message to the transcript."""
        self.messages.append(message)

    def get_context(self) -> str:
        """Return the last k messages concatenated."""
        tail = self.messages[-self.k :] if self.k else self.messages
        return "\n".join(tail)


__all__ = ["TranscriptMemory"]
