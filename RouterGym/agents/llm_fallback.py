"""LLM fallback agent.

Escalates requests that fail contracts, confidence, or safety thresholds.
"""


class LLMFallbackAgent:
    """Placeholder LLM-backed agent for escalations."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id

    def generate(self, prompt: str) -> str:
        """Generate a response via LLM (stub)."""
        return ""
