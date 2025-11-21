"""Salience-gated RAG memory backend."""

from typing import List, Tuple


class SalienceMemory:
    """Adds salience scoring to retrieved context."""

    def __init__(self) -> None:
        self.docs: List[str] = []

    def upsert(self, text: str) -> None:
        """Store a document with implicit salience scoring."""
        self.docs.append(text)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k documents with placeholder salience scores."""
        return [(doc, 0.0) for doc in self.docs[:k]]
