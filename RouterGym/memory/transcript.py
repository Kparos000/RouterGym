"""Transcript memory backend."""

from typing import List


class TranscriptMemory:
    """Keeps a running transcript of turns."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def add(self, message: str) -> None:
        """Append a message to the transcript."""
        self.messages.append(message)

    def get_context(self) -> str:
        """Return the concatenated transcript."""
        return "\n".join(self.messages)
