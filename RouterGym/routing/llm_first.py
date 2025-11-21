"""LLM-first routing policy."""

from typing import Any, Dict, Optional


class LLMFirstRouter:
    """Always prefer LLM, with optional downshift hooks."""

    def route(self, prompt: str, kb_retriever: Optional[Any] = None) -> Dict[str, Any]:
        """Return routing decision metadata (kb_retriever included for pipeline parity)."""
        return {"strategy": "llm_first", "prompt": prompt, "kb_attached": kb_retriever is not None}
