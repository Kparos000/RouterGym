"""Model interface and ticket agent with optional KB retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from RouterGym.agents.generator import build_prompt, generate_response


class TicketAgent:
    """Minimal ticket agent that can enrich prompts with KB snippets."""

    def __init__(self, kb_retriever: Optional[Any] = None) -> None:
        self.kb_retriever = kb_retriever

    def generate(self, ticket: Dict[str, Any], top_k: int = 3) -> str:
        """Generate a response using ticket text and optional KB context."""
        text = ticket.get("text") or ticket.get("body") or ""
        snippets: List[str] = []
        if self.kb_retriever is not None and text:
            try:
                results = self.kb_retriever.retrieve(text, top_k=top_k)
                for item in results:
                    chunk = item.get("chunk") or item.get("snippet") or ""
                    if chunk:
                        snippets.append(chunk)
            except Exception:
                snippets = []
        prompt = build_prompt(text, snippets)
        return generate_response(prompt)

