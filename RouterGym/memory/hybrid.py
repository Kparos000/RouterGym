"""Hybrid hierarchical memory combining BM25, dense RAG, salience, and transcript context."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from RouterGym.memory.base import MemoryBase, MemoryRetrieval
from RouterGym.memory.bm25 import BM25Memory
from RouterGym.memory.rag import RAGMemory
from RouterGym.memory.salience import SalienceGatedMemory
from RouterGym.memory.transcript import TranscriptMemory


class HybridRAGMemory(MemoryBase):
    """Hybrid retrieval with lexical + dense fusion and transcript hierarchy."""

    def __init__(self, top_k: int = 3, alpha: float = 0.6) -> None:
        super().__init__()
        self.top_k = top_k
        self.alpha = alpha
        self.bm25 = BM25Memory(top_k=top_k)
        self.dense = RAGMemory(top_k=top_k)
        self.salience = SalienceGatedMemory(top_k=top_k, max_items=8)
        self.transcript = TranscriptMemory(k=3)
        self._latest_context = ""

    def load(self, ticket: Dict[str, Any]) -> None:
        super().load(ticket)
        self.bm25.load(ticket)
        self.dense.load(ticket)
        self.salience.load(ticket)
        self.transcript.load(ticket)

    def update(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.transcript.update(text)
        self.salience.update(text)
        self.bm25.update(text)
        self.dense.update(text)

    def retrieve(self, query: Optional[str] = None) -> MemoryRetrieval:
        start = time.perf_counter()
        query_text = query or (self.transcript.messages[-1] if self.transcript.messages else "")

        transcript_hits = self._transcript_hits(query_text)
        bm25_result = self.bm25.retrieve(query_text)
        dense_result = self.dense.retrieve(query_text)
        fused_snippets = self._fuse_snippets(
            bm25_result.retrieval_metadata.get("snippets", []),
            dense_result.retrieval_metadata.get("snippets", []),
        )

        context_parts = []
        if transcript_hits:
            context_parts.append(self._format_snippets(transcript_hits, prefix="Transcript"))
        if fused_snippets:
            context_parts.append(self._format_snippets(fused_snippets, prefix="KB Hybrid"))
        context = "\n\n".join([part for part in context_parts if part])

        latency_ms = max(
            (time.perf_counter() - start) * 1000,
            bm25_result.retrieval_latency_ms,
            dense_result.retrieval_latency_ms,
        )
        relevance = (
            fused_snippets[0]["score"]
            if fused_snippets
            else (transcript_hits[0][1] if transcript_hits else 0.0)
        )
        metadata = {
            "mode": "rag_hybrid",
            "alpha": self.alpha,
            "query": query_text,
            "fused_snippets": fused_snippets,
            "transcript_hits": [{"text": t, "score": s} for t, s in transcript_hits],
            "bm25": bm25_result.retrieval_metadata,
            "dense": dense_result.retrieval_metadata,
        }
        token_cost = self._estimate_tokens(query_text) + self._estimate_tokens(context)
        self._latest_context = context
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _transcript_hits(self, query: str) -> List[Tuple[str, float]]:
        """Score transcript snippets by overlap + salience."""
        if not query:
            return []
        q_tokens = set(query.lower().split())
        hits: List[Tuple[str, float]] = []
        for text, sal_score in self.salience.docs:
            tokens = set(text.lower().split())
            if not tokens:
                continue
            overlap = len(tokens & q_tokens) / max(len(q_tokens), 1)
            score = sal_score * (1.0 + overlap)
            hits.append((text, score))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[: self.top_k]

    def _fuse_snippets(
        self, bm25_snips: List[Dict[str, Any]], dense_snips: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not bm25_snips and not dense_snips:
            return []
        max_bm25 = max((s.get("score", 0.0) for s in bm25_snips), default=0.0)
        max_dense = max((s.get("score", 0.0) for s in dense_snips), default=0.0)
        fused: Dict[str, Dict[str, Any]] = {}

        def add_snip(snippet: Dict[str, Any], kind: str) -> None:
            text = snippet.get("text", "")
            if not text:
                return
            entry = fused.setdefault(text, {"text": text, "bm25": 0.0, "dense": 0.0})
            entry[kind] = float(snippet.get("score", 0.0))

        for snip in bm25_snips:
            add_snip(snip, "bm25")
        for snip in dense_snips:
            add_snip(snip, "dense")

        fused_list: List[Dict[str, Any]] = []
        for entry in fused.values():
            bm25_norm = entry["bm25"] / max_bm25 if max_bm25 else 0.0
            dense_norm = entry["dense"] / max_dense if max_dense else 0.0
            salience_boost = 1.0 + 0.1 * self._salience_score(entry["text"])
            score = (self.alpha * bm25_norm + (1 - self.alpha) * dense_norm) * salience_boost
            fused_list.append({"text": entry["text"], "score": score})

        fused_list.sort(key=lambda item: item["score"], reverse=True)
        return fused_list[: self.top_k]

    def _format_snippets(self, snippets: List[Any], prefix: str) -> str:
        formatted = []
        for idx, snippet in enumerate(snippets, start=1):
            text = snippet[0] if isinstance(snippet, tuple) else snippet.get("text", "")
            score = snippet[1] if isinstance(snippet, tuple) else snippet.get("score", 0.0)
            if not text:
                continue
            formatted.append(f"### {prefix} {idx} (score={score:.2f}):\n> {text.strip()}")
        return "\n\n".join(formatted)

    def _salience_score(self, text: str) -> float:
        tokens = [tok for tok in text.split() if tok]
        if not tokens:
            return 0.0
        unique = len(set(tokens))
        return min(unique / max(len(tokens), 1), 1.0)


__all__ = ["HybridRAGMemory"]
