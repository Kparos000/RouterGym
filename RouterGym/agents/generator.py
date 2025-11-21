"""Prompt builder and response generator scaffolding."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def build_prompt(ticket_text: str, kb_snippets: List[str]) -> str:
    """Construct a prompt with KB references."""
    prompt_parts = [ticket_text.strip()]
    for idx, snippet in enumerate(kb_snippets, start=1):
        prompt_parts.append(f"### KB Reference {idx}:\n> {snippet.strip()}")
    return "\n\n".join([p for p in prompt_parts if p])


def generate_response(prompt: str) -> str:
    """Stub generation function."""
    return f"[DRAFT RESPONSE]\n{prompt}"


class JSONContract:
    """Validate JSON structure."""

    def __init__(self, required_fields: Optional[List[str]] = None) -> None:
        self.required_fields = required_fields or []

    def validate(self, text: str) -> bool:
        """Return True if JSON parses and required fields exist."""
        try:
            data = json.loads(text)
        except Exception:
            return False
        return all(field in data for field in self.required_fields)


class SchemaContract(JSONContract):
    """Validate schema for classification, answer, reasoning."""

    def __init__(self) -> None:
        super().__init__(required_fields=["classification", "answer", "reasoning"])


class SelfRepair:
    """Retry loop to repair invalid JSON outputs."""

    def __init__(self, model_interface: Any, contract: JSONContract, max_retries: int = 2) -> None:
        self.model_interface = model_interface
        self.contract = contract
        self.max_retries = max_retries

    def repair(self, prompt: str, initial_output: str) -> str:
        """Attempt to fix JSON output by re-prompting the model."""
        if self.contract.validate(initial_output):
            return initial_output
        attempt_output = initial_output
        for _ in range(self.max_retries):
            fix_prompt = f"{prompt}\n\nThe previous JSON was invalid. Fix JSON only."
            attempt_output = self.model_interface.generate(fix_prompt)
            if self.contract.validate(attempt_output):
                return attempt_output
        return attempt_output


class ResponseGenerator:
    """Combine tickets, memory, and KB into a prompt and generate with contracts."""

    def __init__(self, model_interface: Any, contracts: Optional[JSONContract] = None) -> None:
        self.model_interface = model_interface
        self.contract = contracts or SchemaContract()
        self.self_repair = SelfRepair(model_interface, self.contract)

    def build_prompt(self, ticket: Dict[str, Any], memory_context: str, kb_snippets: List[str]) -> str:
        """Build a structured prompt from ticket, memory, and KB."""
        base_text = ticket.get("text", "")
        kb_section = [f"### KB Reference {i+1}:\n> {s.strip()}" for i, s in enumerate(kb_snippets)]
        memory_section = f"### Memory Context:\n{memory_context}" if memory_context else ""
        parts = [
            base_text,
            memory_section,
            "\n\n".join(kb_section) if kb_section else "",
            "### Respond with JSON containing classification, answer, reasoning.",
        ]
        return "\n\n".join([p for p in parts if p])

    def generate(self, ticket: Dict[str, Any], memory_context: str, kb_snippets: List[str]) -> str:
        """Generate a response and self-repair if contract fails."""
        prompt = self.build_prompt(ticket, memory_context, kb_snippets)
        raw_output = self.model_interface.generate(prompt)
        return self.self_repair.repair(prompt, raw_output)

