"""LLM-first routing policy."""

from typing import Any, Dict


class LLMFirstRouter:
    """Always prefer LLM, with optional downshift hooks."""

    def route(self, prompt: str) -> Dict[str, Any]:
        """Return routing decision metadata."""
        return {"strategy": "llm_first", "prompt": prompt}
