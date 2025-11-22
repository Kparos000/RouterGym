"""Salience-gated RAG memory backend."""

from typing import List, Tuple

from RouterGym.memory.base import MemoryBase


class SalienceGatedMemory(MemoryBase):
    """Adds simple gating before using salient snippets."""

    def __init__(self, top_k: int = 3, max_items: int = 5) -> None:
        self.docs: List[Tuple[str, float]] = []
        self.top_k = top_k
        self.max_items = max_items

    def _score(self, text: str) -> float:
        """Simple salience heuristic: length and unique tokens."""
        tokens = text.split()
        unique = len(set(tokens))
        return unique + 0.1 * len(tokens)

    def add(self, text: str) -> None:
        """Store text with salience score, keep top-N."""
        score = self._score(text)
        self.docs.append((text, score))
        self.docs = sorted(self.docs, key=lambda x: x[1], reverse=True)[: self.max_items]

    def get_context(self) -> str:
        """Return top-k salient messages."""
        top = self.docs[: self.top_k]
        return "\n".join([msg for msg, _ in top])


__all__ = ["SalienceGatedMemory"]
