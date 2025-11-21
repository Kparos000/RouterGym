"""RAG memory backend."""

from typing import List, Tuple


class RAGMemory:
    """Retrieval-augmented memory placeholder."""

    def __init__(self) -> None:
        self.docs: List[str] = []

    def upsert(self, text: str) -> None:
        """Store a document."""
        self.docs.append(text)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k documents with dummy scores."""
        return [(doc, 0.0) for doc in self.docs[:k]]
