"""Salience-gated RAG memory backend."""

from typing import List

from RouterGym.memory.base import MemoryBase
from RouterGym.memory.rag import RAGMemory


class SalienceGatedMemory(MemoryBase):
    """Adds simple gating before using RAG retrieval."""

    def __init__(self, top_k: int = 3) -> None:
        self.docs: List[str] = []
        self.rag = RAGMemory(top_k=top_k)

    def should_retrieve(self, ticket_text: str) -> bool:
        """Decide whether to retrieve; scaffold uses length heuristic."""
        return len(ticket_text) > 20

    def add(self, text: str) -> None:
        """Store text for potential retrieval."""
        self.docs.append(text)
        self.rag.add(text)

    def get_context(self) -> str:
        """Return context, gated by salience heuristic."""
        ticket_text = self.docs[-1] if self.docs else ""
        if self.should_retrieve(ticket_text):
            return self.rag.get_context()
        return f"Recent note: {ticket_text}" if ticket_text else ""
