"""RAG memory backend."""

from typing import List, Tuple

from RouterGym.memory.base import MemoryBase
from RouterGym.data import kb_loader


class RAGMemory(MemoryBase):
    """Retrieval-augmented memory using KB retriever."""

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k
        self.docs: List[str] = []

    def add(self, text: str) -> None:
        """Store a document."""
        self.docs.append(text)

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """Return top-k KB snippets."""
        try:
            results = kb_loader.retrieve(query, top_k=self.top_k)
            return [(r["chunk"], r.get("score", 0.0)) for r in results]
        except Exception:
            return []

    def get_context(self) -> str:
        """Return formatted KB references."""
        snippets = []
        # Use the most recent doc, if any, otherwise no query context
        query = self.docs[-1] if self.docs else ""
        for idx, (chunk, _) in enumerate(self.retrieve(query), start=1):
            snippets.append(f"### KB Reference {idx}:\n> {chunk.strip()}")
        return "\n\n".join(snippets)
