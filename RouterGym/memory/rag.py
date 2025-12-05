"""RAG memory backend."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from RouterGym.memory.base import MemoryBase, MemoryRetrieval
from RouterGym.data.policy_kb import kb_loader

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


class RAGMemory(MemoryBase):
    """Retrieval-augmented memory using KB retriever."""

    def __init__(self, top_k: int = 3, embed_model: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.top_k = top_k
        self.embed_model = embed_model
        self.docs: List[str] = []
        self.kb = kb_loader.load_kb()
        self.doc_keys, self.doc_texts = self._collect_docs(self.kb)
        self.kb_hash = self._kb_hash_safe()
        self.embedder = self._maybe_load_embedder(embed_model)
        self._latest_context = ""
        cached = self._load_cached_embeddings()
        if cached is not None:
            self.doc_embeddings = cached
        else:
            self.doc_embeddings = self._embed(self.doc_texts)
            self._maybe_cache_embeddings(self.doc_embeddings)

    def _maybe_load_embedder(self, model_name: str) -> Any | None:
        """Return a SentenceTransformer instance when available."""
        if SentenceTransformer is None:
            return None
        # Disable hf_transfer to avoid hard dependency during tests/offline runs.
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") not in {"0", "false", "False"}:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        try:
            return SentenceTransformer(model_name)
        except Exception:
            return None

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

    def load(self, ticket: Dict[str, Any]) -> None:
        super().load(ticket)

    def update(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not text:
            return
        self.docs.append(text)
        if len(self.docs) > 100:
            self.docs = self.docs[-100:]

    def retrieve(self, query: Optional[str] = None) -> MemoryRetrieval:
        query = query or (self.docs[-1] if self.docs else "")
        snippets, backend = self._retrieve_snippets(query)
        context = self._format_snippets(snippets)
        self._latest_context = context
        scores = [score for _, score in snippets]
        relevance = float(sum(scores) / len(scores)) if scores else 0.0
        metadata = {
            "mode": "rag",
            "kb_hash": self.kb_hash,
            "backend": backend,
            "top_k": self.top_k,
            "snippets": [
                {"text": chunk, "score": score}
                for chunk, score in snippets
            ],
            "query": query,
        }
        token_cost = self._estimate_tokens(query) + self._estimate_tokens(context)
        return MemoryRetrieval(
            retrieved_context=context,
            retrieval_metadata=metadata,
            retrieval_cost_tokens=token_cost,
            relevance_score=relevance,
        )

    def summarize(self) -> str:
        return self._latest_context

    def _retrieve_snippets(self, query: str) -> Tuple[List[Tuple[str, float]], str]:
        if not query:
            return [], "none"
        live = kb_loader.retrieve(query, top_k=self.top_k)
        if live:
            return [
                (r.get("chunk") or r.get("text", ""), float(r.get("score", 0.0)))
                for r in live
            ], "kb_retriever"

        if self.embedder is not None and self.doc_embeddings.size > 0:
            query_vec = np.array(self.embedder.encode([query]), dtype="float32")
            doc_norm = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-9
            query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9
            sims = (self.doc_embeddings @ query_vec.T) / (doc_norm * query_norm)
            sims = sims.flatten()
            top_idx = sims.argsort()[-self.top_k :][::-1]
            ranked = [(self.doc_texts[idx], float(sims[idx])) for idx in top_idx]
            return ranked, "embedding"

        lexical = self._lexical_rank(query)
        return lexical, "lexical"

    def _lexical_rank(self, query: str) -> List[Tuple[str, float]]:
        q_tokens = set(query.lower().split())
        ranked: List[Tuple[str, float]] = []
        for text in self.doc_texts[:500]:
            tokens = set(text.lower().split())
            if not tokens:
                continue
            overlap = len(tokens & q_tokens) / len(tokens)
            if overlap > 0:
                ranked.append((text, overlap))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: self.top_k]

    def _format_snippets(self, snippets: List[Tuple[str, float]]) -> str:
        formatted = []
        for idx, (chunk, score) in enumerate(snippets, start=1):
            if not chunk:
                continue
            formatted.append(f"### KB Reference {idx} (score={score:.2f}):\n> {chunk.strip()}")
        return "\n\n".join(formatted)


__all__ = ["RAGMemory"]
