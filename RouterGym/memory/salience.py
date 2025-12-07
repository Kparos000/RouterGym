"""Salience-gated transcript memory backend."""

from __future__ import annotations

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from RouterGym.memory.base import MemoryBase, MemoryRetrieval


class SalienceGatedMemory(MemoryBase):
    """Ranks transcript snippets by salience before reuse."""

    def __init__(self, top_k: int = 3, max_items: int = 5, decay: float = 0.9) -> None:
        super().__init__()
        self.docs: List[Tuple[str, float]] = []
        self.top_k = top_k
        self.max_items = max_items
        self.decay = decay
        self.token_freq: Counter[str] = Counter()
        self._last_tokens: List[str] = []
        self.turn = 0

    def load(self, ticket: Dict[str, str]) -> None:
        super().load(ticket)

    def update(self, text: str, metadata: Optional[Dict[str, str]] = None) -> None:
        if not text:
            return
        self.turn += 1
        tokens = self._tokenize(text)
        score = self._score(tokens, text)
        self.docs.append((text.strip(), score))
        self.docs = sorted(self.docs, key=lambda item: item[1], reverse=True)[: self.max_items]
        self.token_freq.update(tokens)
        self._last_tokens = tokens

    def retrieve(self, query: Optional[str] = None) -> MemoryRetrieval:
        t_start = time.perf_counter()
        context = self.summarize()
        scores = [score for _, score in self.docs[: self.top_k]]
        max_score = max(scores) if scores else 1.0
        relevance = float(sum(scores) / (len(scores) * max_score)) if scores else 0.0
        metadata = {
            "mode": "salience",
            "top_k": self.top_k,
            "max_items": self.max_items,
            "scores": scores,
            "query": query or "",
        }
        return MemoryRetrieval(
            retrieved_context=context,
            retrieval_metadata=metadata,
            retrieval_cost_tokens=self._estimate_tokens(context),
            relevance_score=relevance,
            retrieval_latency_ms=(time.perf_counter() - t_start) * 1000,
            retrieved_context_length=len(context),
        )

    def summarize(self) -> str:
        top = [msg for msg, _ in self.docs[: self.top_k]]
        return "\n".join(top)

    def _tokenize(self, text: str) -> List[str]:
        return [tok.lower() for tok in text.split() if tok.strip()]

    def _score(self, tokens: List[str], text: str) -> float:
        if not tokens:
            return 0.0
        rarity = sum(1.0 / (1.0 + self.token_freq[token]) for token in tokens) / len(tokens)
        sentence_weight = 1.0 + 0.15 * max(text.count("."), text.count("\n"), 1)
        overlap = len(set(tokens) & set(self._last_tokens)) / max(len(tokens), 1)
        continuity = 1.0 + (1.0 - overlap) * 0.5
        length_weight = min(len(tokens) / 20.0, 2.0)
        decay_factor = self.decay ** self.turn
        return (rarity * sentence_weight * continuity + length_weight) * decay_factor


__all__ = ["SalienceGatedMemory"]
