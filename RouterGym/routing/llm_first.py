"""LLM-first routing policy."""

from typing import Any, Dict

from RouterGym.routing.base import BaseRouter


class LLMFirstRouter(BaseRouter):
    """Always prefer LLM, with optional downshift hooks."""

    def route(self, ticket: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return routing decision metadata (always LLM, fallback only on failure)."""
        prompt = ticket if isinstance(ticket, str) else ticket.get("text", "")
        return {
            "strategy": "llm_first",
            "prompt": prompt,
            "kb_attached": kwargs.get("kb_retriever") is not None,
            "target_model": "llm",
            "fallback": False,
        }
