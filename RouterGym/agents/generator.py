"""Prompt builder and response generator scaffolding."""

from __future__ import annotations

from typing import List


def build_prompt(ticket_text: str, kb_snippets: List[str]) -> str:
    """Construct a prompt with KB references."""
    prompt_parts = [ticket_text.strip()]
    for idx, snippet in enumerate(kb_snippets, start=1):
        prompt_parts.append(f"### KB Reference {idx}:\n> {snippet.strip()}")
    return "\n\n".join([p for p in prompt_parts if p])


def generate_response(prompt: str) -> str:
    """Stub generation function."""
    return f"[DRAFT RESPONSE]\n{prompt}"

