"""RAG memory backend."""

from typing import Any, Dict, List, Tuple

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
        self.embed_model = embed_model
        self.docs: List[str] = []
        self.kb = kb_loader.load_kb()
        self.doc_keys, self.doc_texts = self._collect_docs(self.kb)
        self.kb_hash = self._kb_hash_safe()
        self.embedder = SentenceTransformer(embed_model) if SentenceTransformer is not None else None
        cached = self._load_cached_embeddings()
        if cached is not None:
            self.doc_embeddings = cached
        else:
            self.doc_embeddings = self._embed(self.doc_texts)
            self._maybe_cache_embeddings(self.doc_embeddings)

    def _collect_docs(self, kb: Dict[str, str] | Any) -> Tuple[List[str], List[str]]:
        if isinstance(kb, dict):
            items = sorted(kb.items())
            keys = [path for path, _ in items]
            texts = [text for _, text in items]
            return keys, texts
        return [], []

    def _kb_hash_safe(self) -> str:
        try:
            return kb_loader.kb_hash()
        except Exception:
            return ""

    def _cache_key(self) -> str:
        return f"{self.embed_model}:{self.kb_hash}"

    def _load_cached_embeddings(self) -> np.ndarray | None:
        if self.embedder is None or not self.kb_hash or not self.doc_texts:
            return None
        try:
            cache = kb_loader.load_cached_embeddings()
            entry = cache.get(self._cache_key(), {})
        except Exception:
            return None
        if not isinstance(entry, dict):
            return None
        if entry.get("kb_hash") != self.kb_hash or entry.get("model") != self.embed_model:
            return None
        if entry.get("kb_order") and entry["kb_order"] != self.doc_keys:
            return None
        embeddings = entry.get("embeddings")
        if embeddings is None:
            return None
        arr = np.array(embeddings, dtype="float32")
        if arr.shape[0] != len(self.doc_texts):
            return None
        return arr

    def _maybe_cache_embeddings(self, embeddings: np.ndarray) -> None:
        if self.embedder is None or not self.kb_hash or embeddings.size == 0:
            return
        try:
            cache = kb_loader.load_cached_embeddings()
            cache[self._cache_key()] = {
                "kb_hash": self.kb_hash,
                "kb_order": self.doc_keys,
                "model": self.embed_model,
                "embeddings": embeddings,
            }
            kb_loader.save_cached_embeddings(cache)
        except Exception:
            return

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
