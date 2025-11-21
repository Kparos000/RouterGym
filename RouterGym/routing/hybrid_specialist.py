"""Hybrid specialist routing policy."""

from typing import Any, Dict, Optional


class HybridSpecialistRouter:
    """Route to specialized SLMs by domain; fallback to LLM as needed."""

    def route(self, prompt: str, kb_retriever: Optional[Any] = None) -> Dict[str, Any]:
        """Return routing decision metadata (kb_retriever included for pipeline parity)."""
        return {"strategy": "hybrid_specialist", "prompt": prompt, "kb_attached": kb_retriever is not None}
