"""Null memory backend."""

import time
from typing import Any, Dict, Optional

from RouterGym.memory.base import MemoryBase, MemoryRetrieval


class NoneMemory(MemoryBase):
    """Memory backend that stores nothing."""

    def __init__(self) -> None:
        super().__init__()

    def load(self, ticket: Any) -> None:  # pragma: no cover - trivial
        return None

    def retrieve(self, query: Optional[str] = None) -> MemoryRetrieval:
        t_start = time.perf_counter()
        return MemoryRetrieval(
            retrieved_context="",
            retrieval_metadata={"mode": "none", "query": query or ""},
            retrieval_cost_tokens=0,
            relevance_score=0.0,
            retrieval_latency_ms=(time.perf_counter() - t_start) * 1000,
            retrieved_context_length=0,
        )

    def update(self, item: Any, metadata: Optional[Dict[str, Any]] = None) -> None:  # pragma: no cover - trivial
        return None

    def summarize(self) -> str:
        return ""


__all__ = ["NoneMemory"]
