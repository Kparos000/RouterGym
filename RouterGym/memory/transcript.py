"""Transcript memory backend."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from RouterGym.memory.base import MemoryBase, MemoryRetrieval


class TranscriptMemory(MemoryBase):
    """Keeps a running transcript of turns."""

    def __init__(self, k: int = 3, max_messages: int = 50) -> None:
        super().__init__()
        self.messages: List[str] = []
        self.k = k
        self.max_messages = max_messages

    def load(self, ticket: Dict[str, str]) -> None:
        super().load(ticket)

    def update(self, message: str, metadata: Optional[Dict[str, str]] = None) -> None:
        if not message:
            return
        self.messages.append(str(message).strip())
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def retrieve(self, query: Optional[str] = None) -> MemoryRetrieval:
        t_start = time.perf_counter()
        context = self.summarize()
        tail_count = len(self.messages[-self.k :]) if self.k else len(self.messages)
        relevance = min(tail_count / max(self.k or 1, 1), 1.0)
        return MemoryRetrieval(
            retrieved_context=context,
            retrieval_metadata={
                "mode": "transcript",
                "messages_available": len(self.messages),
                "window": self.k,
            },
            retrieval_cost_tokens=self._estimate_tokens(context),
            relevance_score=relevance,
            retrieval_latency_ms=(time.perf_counter() - t_start) * 1000,
            retrieved_context_length=len(context),
        )

    def summarize(self) -> str:
        tail = self.messages[-self.k :] if self.k else self.messages
        return "\n".join(tail)


__all__ = ["TranscriptMemory"]
