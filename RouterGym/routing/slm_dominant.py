"""SLM-dominant routing policy."""

from typing import Any, Dict, Optional


class SLMDominantRouter:
    """Prefer SLM routes and escalate on contract or confidence failures."""

    def route(self, prompt: str, kb_retriever: Optional[Any] = None) -> Dict[str, Any]:
        """Return routing decision metadata (kb_retriever included for pipeline parity)."""
        return {"strategy": "slm_dominant", "prompt": prompt, "kb_attached": kb_retriever is not None}
