"""Hybrid specialist routing policy."""

from typing import Any, Dict


class HybridSpecialistRouter:
    """Route to specialized SLMs by domain; fallback to LLM as needed."""

    def route(self, prompt: str) -> Dict[str, Any]:
        """Return routing decision metadata."""
        return {"strategy": "hybrid_specialist", "prompt": prompt}
