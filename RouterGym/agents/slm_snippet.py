"""SLM snippet agent.

Produces short, structured snippets (summaries, key steps) under contract.
"""


class SLMSnippetAgent:
    """Placeholder snippet generator using an SLM backend."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id

    def generate_snippet(self, prompt: str) -> str:
        """Generate a concise snippet (stubbed)."""
        return ""
