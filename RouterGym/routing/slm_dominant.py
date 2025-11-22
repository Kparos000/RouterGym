"""SLM-dominant routing policy."""

from typing import Any, Dict

from RouterGym.routing.base import BaseRouter


class SLMDominantRouter(BaseRouter):
    """Prefer SLM routes and escalate on contract or confidence failures."""

    def route(self, ticket: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return routing decision metadata; fallback flag left False for scaffold."""
        prompt = ticket if isinstance(ticket, str) else ticket.get("text", "")
        return {
            "strategy": "slm_dominant",
            "prompt": prompt,
            "kb_attached": kwargs.get("kb_retriever") is not None,
            "target_model": "slm",
            "fallback": False,
        }
