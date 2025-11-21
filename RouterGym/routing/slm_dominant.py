"""SLM-dominant routing policy."""

from typing import Any, Dict


class SLMDominantRouter:
    """Prefer SLM routes and escalate on contract or confidence failures."""

    def route(self, prompt: str) -> Dict[str, Any]:
        """Return routing decision metadata."""
        return {"strategy": "slm_dominant", "prompt": prompt}
