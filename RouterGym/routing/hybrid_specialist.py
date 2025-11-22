"""Hybrid specialist routing policy."""

from typing import Any, Dict

from RouterGym.routing.base import BaseRouter


class HybridSpecialistRouter(BaseRouter):
    """Route to specialized SLMs by domain; fallback to LLM as needed."""

    def __init__(self) -> None:
        self.category_to_model = {
            "access": "slm",
            "hardware": "slm",
            "hr_support": "slm",
        }

    def route(self, ticket: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return routing decision metadata; uses category to pick model."""
        prompt = ticket if isinstance(ticket, str) else ticket.get("text", "")
        category = None
        if not isinstance(ticket, str):
            category = ticket.get("category")
        cat_key = str(category) if category is not None else ""

        target_model = self.category_to_model.get(cat_key, "llm")
        return {
            "strategy": "hybrid_specialist",
            "prompt": prompt,
            "kb_attached": kwargs.get("kb_retriever") is not None,
            "target_model": target_model,
            "fallback": target_model == "llm",
        }
