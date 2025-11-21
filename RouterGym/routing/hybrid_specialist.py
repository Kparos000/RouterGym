"""Hybrid specialist routing policy."""

from typing import Any, Dict, Optional

from RouterGym.routing.base import BaseRouter


class HybridSpecialistRouter(BaseRouter):
    """Route to specialized SLMs by domain; fallback to LLM as needed."""

    def __init__(self) -> None:
        self.category_to_model = {
            "access": "slm",
            "hardware": "slm",
            "hr_support": "slm",
        }

    def route(self, ticket: Any, kb_retriever: Optional[Any] = None) -> Dict[str, Any]:
        """Return routing decision metadata; uses category to pick model."""
        prompt = ticket if isinstance(ticket, str) else ticket.get("text", "")
        category = None
        if not isinstance(ticket, str):
            category = ticket.get("category")

        target_model = self.category_to_model.get(category, "llm")
        return {
            "strategy": "hybrid_specialist",
            "prompt": prompt,
            "kb_attached": kb_retriever is not None,
            "target_model": target_model,
            "fallback": target_model == "llm",
        }
