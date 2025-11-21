"""SLM-dominant routing policy."""

from typing import Any, Dict, Optional

from RouterGym.routing.base import BaseRouter


class SLMDominantRouter(BaseRouter):
    """Prefer SLM routes and escalate on contract or confidence failures."""

    def route(self, ticket: Any, kb_retriever: Optional[Any] = None) -> Dict[str, Any]:
        """Return routing decision metadata; fallback flag left False for scaffold."""
        prompt = ticket if isinstance(ticket, str) else ticket.get("text", "")
        return {
            "strategy": "slm_dominant",
            "prompt": prompt,
            "kb_attached": kb_retriever is not None,
            "target_model": "slm",
            "fallback": False,
        }
