"""Base memory interface with unified retrieval contract."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


TOKENS_PER_CHAR = 0.25


@dataclass(slots=True)
class MemoryRetrieval:
    """Structured memory retrieval payload."""

    retrieved_context: str = ""
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_cost_tokens: int = 0
    relevance_score: float = 0.0
    retrieval_latency_ms: float = 0.0
    retrieved_context_length: int = 0

    def __post_init__(self) -> None:
        if not self.retrieved_context_length:
            self.retrieved_context_length = len(self.retrieved_context)

    @property
    def memory_cost_tokens(self) -> int:
        """Alias for compatibility."""
        return self.retrieval_cost_tokens

    @property
    def memory_relevance_score(self) -> float:
        """Alias for compatibility."""
        return self.relevance_score

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "retrieved_context": self.retrieved_context,
            "retrieval_metadata": self.retrieval_metadata,
            "retrieval_cost_tokens": self.retrieval_cost_tokens,
            "memory_cost_tokens": self.retrieval_cost_tokens,
            "relevance_score": self.relevance_score,
            "memory_relevance_score": self.relevance_score,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "retrieved_context_length": self.retrieved_context_length,
        }


class MemoryBase:
    """Base memory abstract class providing a unified interface."""

    def __init__(self) -> None:
        self._tokens_per_char = TOKENS_PER_CHAR

    # ------------------------------------------------------------------
    # Unified API expected by downstream consumers
    # ------------------------------------------------------------------
    def load(self, ticket: Any) -> None:
        """Prime memory with the current ticket."""
        text = self._extract_text(ticket)
        if text:
            self.update(text)

    def retrieve(self, query: Optional[str] = None) -> MemoryRetrieval:
        """Retrieve context plus metadata for a query."""
        t_start = time.perf_counter()
        context = self.summarize()
        metadata = {"mode": self.__class__.__name__.lower(), "query": query or ""}
        return MemoryRetrieval(
            retrieved_context=context,
            retrieval_metadata=metadata,
            retrieval_cost_tokens=self._estimate_tokens(context),
            relevance_score=0.0,
            retrieval_latency_ms=(time.perf_counter() - t_start) * 1000,
        )

    def update(self, item: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist information back into memory."""
        raise NotImplementedError

    def summarize(self) -> str:
        """Return the current memory summary used for prompting."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Backwards-compatible helpers
    # ------------------------------------------------------------------
    def add(self, item: Any) -> None:
        """Backward compatible alias for update()."""
        self.update(item)

    def get_context(self) -> str:
        """Backward compatible alias for summarize()."""
        return self.summarize()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _estimate_tokens(self, text: str) -> int:
        return max(int(len(text) * self._tokens_per_char), 0)

    def _extract_text(self, item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("text", ""))
        return str(item)


__all__ = ["MemoryBase", "MemoryRetrieval"]
