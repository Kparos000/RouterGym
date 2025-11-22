"""Prompt builder and response generator with contract enforcement."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from RouterGym.contracts.json_contract import JSONContract
from RouterGym.contracts.schema_contract import SchemaContract
from RouterGym.utils.logger import get_logger

log = get_logger(__name__)


def build_prompt(ticket_text: str, kb_snippets: List[str]) -> str:
    """Construct a prompt with KB references."""
    prompt_parts = [ticket_text.strip()]
    for idx, snippet in enumerate(kb_snippets, start=1):
        prompt_parts.append(f"### KB Reference {idx}:\n> {snippet.strip()}")
    return "\n\n".join([p for p in prompt_parts if p])


def generate_response(prompt: str) -> str:
    """Stub generation function."""
    return f"[DRAFT RESPONSE]\n{prompt}"


class SelfRepair:
    """Repair invalid JSON outputs using contract validation."""

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def repair(
        self,
        model: Any,
        prompt: str,
        bad_output: str,
        schema: SchemaContract,
    ) -> str:
        """Attempt to fix bad output by re-prompting the model."""
        json_contract = JSONContract()

        valid_json, parsed = json_contract.validate(bad_output)
        if not valid_json:
            log.error("Contract failure: invalid JSON")
        else:
            is_valid, errors = schema.validate(parsed)
            if is_valid:
                return bad_output
            log.error(f"Schema errors: {errors}")

        attempt_output = bad_output
        for attempt in range(self.max_retries):
            repair_prompt = (
                f"{prompt}\n\nYour previous output violated the schema. "
                "Fix only the missing/incorrect fields and return valid JSON."
            )
            attempt_output = model.generate(repair_prompt)
            ok_json, parsed = json_contract.validate(attempt_output)
            if not ok_json:
                continue
            ok_schema, _ = schema.validate(parsed)
            if ok_schema:
                log.info(f"Repair succeeded on attempt {attempt+1}")
                return attempt_output

        # Best-effort fallback
        try:
            data = json.loads(attempt_output)
            if not isinstance(data, dict):
                raise ValueError
        except Exception:
            data = {}
        for field in schema.required_fields:
            if field not in data:
                data[field] = "Unknown" if field != "action_steps" else []
        log.error("Repair failed after retries; returning best-effort output")
        return json.dumps(data)


class ResponseGenerator:
    """Combine tickets, memory, and KB into a prompt and generate with contracts."""

    def __init__(self, model_interface: Any, contracts: Optional[SchemaContract] = None) -> None:
        self.model_interface = model_interface
        self.contract = contracts or SchemaContract()
        self.self_repair = SelfRepair()

    def build_prompt(self, ticket: Dict[str, Any], memory_context: str, kb_snippets: List[str]) -> str:
        """Build a structured prompt from ticket, memory, and KB."""
        base_text = ticket.get("text", "")
        kb_section = [f"### KB Reference {i+1}:\n> {s.strip()}" for i, s in enumerate(kb_snippets)]
        memory_section = f"### Memory Context:\n{memory_context}" if memory_context else ""
        parts = [
            base_text,
            memory_section,
            "\n\n".join(kb_section) if kb_section else "",
            "### Respond with JSON containing classification, action_steps, final_answer, reasoning.",
        ]
        return "\n\n".join([p for p in parts if p])

    def generate(self, ticket: Dict[str, Any], memory_context: str, kb_snippets: List[str]) -> str:
        """Generate a response and repair if contracts fail."""
        prompt = self.build_prompt(ticket, memory_context, kb_snippets)
        raw_output = self.model_interface.generate(prompt)
        return self.self_repair.repair(self.model_interface, prompt, raw_output, self.contract)
