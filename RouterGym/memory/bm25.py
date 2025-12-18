"""BM25 lexical memory backend (rag_bm25)."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from RouterGym.data.policy_kb import kb_loader
from RouterGym.memory.base import MemoryBase, MemoryRetrieval

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _normalize_text(text: Any) -> str:
    """
    Normalize arbitrary input into a clean string for BM25.

    Accepts Any to be defensive, but always returns a stripped str.
    """
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


class BM25Memory(MemoryBase):
    """Lightweight BM25 retriever over the policy KB."""

    def __init__(self, top_k: int = 3, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.index = kb_loader.load_kb_index()
        self.doc_keys, self.doc_texts, self.doc_meta = self._collect_docs(self.index)
        self.doc_tokens: List[List[str]] = [self._tokenize(text) for text in self.doc_texts]
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: List[Counter[str]] = []
        self._latest_context = ""
        self._build_index()

    def _collect_docs(self, index: Any) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        if isinstance(index, list):
            keys: List[str] = []
            texts: List[str] = []
            meta: List[Dict[str, Any]] = []
            for entry in index:
                if not isinstance(entry, dict):
                    continue
                keys.append(str(entry.get("id", entry.get("path", ""))))
                texts.append(_normalize_text(entry.get("content", "")))
                meta.append(entry)
            return keys, texts, meta
        if isinstance(index, dict):
            # Fallback for legacy dict path->text
            items = sorted(index.items())
            keys = [path for path, _ in items]
            texts = [_normalize_text(text) for _, text in items]
            meta = [{"id": key, "content": text, "path": key, "title": key, "category": ""} for key, text in items]
            return keys, texts, meta
        return [], [], []

    def _tokenize(self, text: Any) -> List[str]:
        normalized = _normalize_text(text)
        return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(normalized)]

    def _build_index(self) -> None:
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq = Counter()
        for tf in self.term_freqs:
            self.doc_freq.update(tf.keys())
        self.avgdl = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        self.N = len(self.doc_tokens)

    def load(self, ticket: Dict[str, Any]) -> None:
        super().load(ticket)

    def update(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Allow incremental additions (stored as lightweight documents)."""
        if not text:
            return
        normalized = _normalize_text(text)
        meta = metadata or {}
        tokens = self._tokenize(normalized)
        key = meta.get("id", f"dynamic:{len(self.doc_keys)}")
        self.doc_keys.append(str(key))
        self.doc_texts.append(normalized)
        self.doc_meta.append(
            {
                "id": str(key),
                "content": normalized,
                "path": meta.get("path", str(key)),
                "title": meta.get("title", str(key)),
                "category": meta.get("category", ""),
                "tags": meta.get("tags", []),
            }
        )
        self.doc_tokens.append(tokens)
        tf = Counter(tokens)
        self.term_freqs.append(tf)
        self.doc_freq.update(tf.keys())
        self.N = len(self.doc_tokens)
        self.avgdl = sum(len(toks) for toks in self.doc_tokens) / max(self.N, 1)

    def retrieve(self, query: Optional[str] = None) -> MemoryRetrieval:
        t_start = time.perf_counter()
        query_text = _normalize_text(query or "")
        q_tokens = self._tokenize(query_text)
        if not q_tokens or not self.doc_tokens:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return MemoryRetrieval(
                retrieved_context="",
                retrieval_metadata={"mode": "rag_bm25", "query": query_text, "snippets": []},
                retrieval_cost_tokens=self._estimate_tokens(query_text),
                relevance_score=0.0,
                retrieval_latency_ms=latency_ms,
                retrieved_context_length=0,
            )

        scores = []
        for idx, tf in enumerate(self.term_freqs):
            doc_len = len(self.doc_tokens[idx])
            score = 0.0
            for token in q_tokens:
                df = self.doc_freq.get(token, 0)
                if df == 0:
                    continue
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                freq = tf.get(token, 0)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += idf * ((freq * (self.k1 + 1)) / denom)
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        top = scores[: self.top_k]
        max_score = top[0][1] if top else 0.0
        snippets = [
            {
                "policy_id": self.doc_meta[idx].get("id", self.doc_keys[idx]),
                "category": self.doc_meta[idx].get("category", ""),
                "title": self.doc_meta[idx].get("title", ""),
                "tags": self.doc_meta[idx].get("tags", []),
                "score": sc,
                "source": self.doc_meta[idx].get("path", self.doc_keys[idx]),
                "text": self.doc_texts[idx],
            }
            for idx, sc in top
        ]
        context_parts = []
        for i, snippet in enumerate(snippets, start=1):
            text = _normalize_text(snippet["text"])
            pid = snippet.get("policy_id", "")
            context_parts.append(
                f"### KB (BM25) {i} [{pid}] (score={snippet['score']:.2f}):\n> {text}"
            )
        context = "\n\n".join(context_parts)
        latency_ms = (time.perf_counter() - t_start) * 1000
        relevance = float(max_score) if max_score else 0.0
        self._latest_context = context
        metadata = {
            "mode": "rag_bm25",
            "query": query_text,
            "top_k": self.top_k,
            "snippets": snippets,
            "retrieval_latency_ms": latency_ms,
        }
        token_cost = self._estimate_tokens(query_text) + self._estimate_tokens(context)
        return MemoryRetrieval(
            retrieved_context=context,
            retrieval_metadata=metadata,
            retrieval_cost_tokens=token_cost,
            relevance_score=relevance,
            retrieval_latency_ms=latency_ms,
            retrieved_context_length=len(context),
        )

    def summarize(self) -> str:
        return self._latest_context


__all__ = ["BM25Memory"]
