"""RAG memory backend."""

from typing import List, Tuple

import numpy as np

from RouterGym.memory.base import MemoryBase
from RouterGym.data.policy_kb import kb_loader

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


class RAGMemory(MemoryBase):
    """Retrieval-augmented memory using KB retriever."""

    def __init__(self, top_k: int = 3, embed_model: str = "all-MiniLM-L6-v2") -> None:
        self.top_k = top_k
        self.docs: List[str] = []
        self.kb = kb_loader.load_kb()
        self.embedder = SentenceTransformer(embed_model) if SentenceTransformer is not None else None
        self.doc_texts = list(self.kb.values()) if isinstance(self.kb, dict) else []
        self.doc_embeddings = self._embed(self.doc_texts)

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.embedder is None or not texts:
            return np.zeros((0, 0), dtype="float32")
        return np.array(self.embedder.encode(texts), dtype="float32")

    def add(self, text: str) -> None:
        """Store a document."""
        self.docs.append(text)

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """Return top-k KB snippets via cosine similarity."""
        if not query:
            return []

        # Prefer live KB retrieval (allows monkeypatching in tests)
        live = kb_loader.retrieve(query, top_k=self.top_k)
        if live:
            return [(r.get("chunk") or r.get("text", ""), float(r.get("score", 0.0))) for r in live]

        if self.embedder is None or self.doc_embeddings.size == 0:
            return []

        query_vec = np.array(self.embedder.encode([query]), dtype="float32")
        doc_norm = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-9
        query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9
        sims = (self.doc_embeddings @ query_vec.T) / (doc_norm * query_norm)
        sims = sims.flatten()
        top_idx = sims.argsort()[-self.top_k :][::-1]
        ranked: List[Tuple[str, float]] = []
        for idx in top_idx:
            ranked.append((self.doc_texts[idx], float(sims[idx])))
        return ranked

    def get_context(self) -> str:
        """Return formatted KB references."""
        snippets = []
        query = self.docs[-1] if self.docs else ""
        for idx, (chunk, _) in enumerate(self.retrieve(query), start=1):
            if not chunk:
                continue
            snippets.append(f"### KB Reference {idx}:\n> {chunk.strip()}")
        return "\n\n".join(snippets)


__all__ = ["RAGMemory"]
