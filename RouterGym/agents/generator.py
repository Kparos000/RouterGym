"""Prompt builder and response generator with contract enforcement."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from RouterGym.contracts.json_contract import JSONContract
from RouterGym.contracts.schema_contract import SchemaContract
from RouterGym.engines.model_registry import get_repair_model
from RouterGym.label_space import CANONICAL_LABELS, CANONICAL_LABEL_SET, canonical_label
from RouterGym.utils.logger import get_logger

log = get_logger(__name__)

CLASS_LABELS = CANONICAL_LABELS

LABELS_LIST_TEXT = ", ".join(CLASS_LABELS)

def classification_instruction() -> str:
    """Instruction block enforcing JSON contract and allowed labels for classification."""
    return "\n".join(
        [
            "You are a support ticket classifier. Choose EXACTLY ONE label from this fixed set:",
            '- access: account/login/password/MFA/S S O issues (e.g., "locked out of account", "reset MFA token")',
            '- administrative rights: permission/role/entitlement changes (e.g., "add to security group", "need admin rights to install")',
            '- hardware: laptops/printers/monitors/docks/devices (e.g., "screen cracked", "printer jam")',
            '- hr support: leave/vacation/benefits/payroll/HR questions (e.g., "extend parental leave", "benefits enrollment")',
            '- purchase: buying/ordering/billing/licenses/subscriptions/invoices (e.g., "raise PO for software", "renew subscription")',
            "- miscellaneous: only if none of the above clearly fits; if about accounts, permissions, entitlements, or billing, prefer a specific label instead of miscellaneous.",
            "",
            "Rules:",
            "* Return strictly valid JSON with exactly one label and a short rationale.",
            "* Allowed labels: access, administrative rights, hardware, hr support, purchase, miscellaneous.",
            "* Do NOT invent labels. Use 'miscellaneous' only when no other label clearly applies.",
            "",
            "Respond ONLY with JSON:",
            '{"label": "<one of access|administrative rights|hardware|hr support|purchase|miscellaneous>", "rationale": "<short why>"}',
        ]
    )


def _call_model(model: Any, prompt: str) -> str:
    """Invoke a model or pipeline and normalize the output to string."""
    output = None
    if hasattr(model, "generate"):
        try:
            output = model.generate(prompt, max_new_tokens=256, temperature=0.2)
        except TypeError:
            output = model.generate(prompt)  # type: ignore[call-arg]
    elif callable(model):
        try:
            output = model(prompt, max_new_tokens=256, temperature=0.2)
        except TypeError:
            output = model(prompt)
    else:
        return str(prompt)

    if isinstance(output, str):
        return output
    if isinstance(output, list) and output and isinstance(output[0], dict) and "generated_text" in output[0]:
        return output[0]["generated_text"]
    return str(output)


def _extract_json_fragment(text: str) -> Any:
    """Try to extract a JSON object substring from arbitrary text."""
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def normalize_output(output: Any) -> Dict[str, str]:
    """Normalize any model output into a dict with final_answer and reasoning."""
    parsed: Dict[str, Any] = {}
    if isinstance(output, dict):
        parsed = output
    elif isinstance(output, str):
        try:
            maybe = json.loads(output)
            if isinstance(maybe, dict):
                parsed = maybe
        except Exception:
            fragment = _extract_json_fragment(output)
            if isinstance(fragment, dict):
                parsed = fragment
    if not parsed and not isinstance(output, dict):
        parsed = {}
    raw_pred = parsed.get("predicted_category") or parsed.get("category") or ""
    final_answer = str(parsed.get("final_answer", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()
    predicted = _normalize_category(str(raw_pred), context=f"{final_answer} {reasoning}")
    if not predicted:
        predicted = "unknown"
    return {
        "final_answer": final_answer,
        "reasoning": reasoning,
        "predicted_category": predicted,
    }


def _ensure_minimum_fields(data: Dict[str, str]) -> Dict[str, str]:
    """Guarantee required fields are present and non-empty."""
    return {
        "final_answer": data.get("final_answer") or "No valid answer produced",
        "reasoning": data.get("reasoning") or "",
        "predicted_category": data.get("predicted_category") or "unknown",
    }


def build_prompt(ticket_text: str, kb_snippets: List[str]) -> str:
    """Construct a prompt with KB references."""
    prompt_parts = [ticket_text.strip()]
    for idx, snippet in enumerate(kb_snippets, start=1):
        prompt_parts.append(f"### KB Reference {idx}:\n> {snippet.strip()}")
    prompt_parts.append("Respond with JSON containing: final_answer, reasoning.")
    return "\n\n".join([p for p in prompt_parts if p])


def generate_response(prompt: str) -> str:
    """Stub generation function."""
    return f"[DRAFT RESPONSE]\n{prompt}"


def _normalize_category(raw: str, context: str = "") -> str:
    """Map raw category text + context into canonical CLASS_LABELS or unknown."""
    text = (raw or "").lower()
    text = re.sub("[^a-z0-9\\s_-]", " ", text).strip()
    combined = f"{text} {context.lower()}".strip()

    strong_access = {
        "login",
        "log in",
        "signin",
        "sign in",
        "password",
        "credential",
        "sso",
        "mfa",
        "otp",
        "lockout",
        "locked out",
        "cannot access",
        "access denied",
        "access",
        "account",
        "auth",
    }
    strong_hardware = {
        "laptop",
        "printer",
        "device",
        "hardware",
        "dock",
        "monitor",
        "screen",
        "keyboard",
        "mouse",
        "pc",
        "desktop",
        "computer",
        "headset",
    }

    def contains_any(keys: set[str]) -> bool:
        return any(k in combined for k in keys)

    if contains_any(strong_access):
        return "access"
    if contains_any(strong_hardware):
        return "hardware"

    if text in CANONICAL_LABEL_SET:
        return text

    keyword_map = [
        (
            {"admin", "administrator", "permission", "privilege", "rights", "entitlement", "group"},
            "administrative rights",
        ),
        ({"hr", "benefit", "leave", "vacation", "payroll"}, "hr support"),
        (
            {
                "buy",
                "purchase",
                "order",
                "procure",
                "invoice",
                "billing",
                "subscription",
                "license",
                "quote",
                "po",
            },
            "purchase",
        ),
    ]
    for keywords, label in keyword_map:
        if any(k in combined for k in keywords):
            return label

    if "misc" in combined or "general" in combined or "other" in combined:
        return "miscellaneous"
    # If no strong match, prefer miscellaneous over unknown to avoid empty labels
    return canonical_label(text)


def infer_category_from_text(text: str) -> str:
    """Heuristic mapping from ticket text to canonical labels."""
    lower = (text or "").lower()
    keyword_map = [
        ({"login", "password", "account", "access", "credential", "mfa", "sso"}, "access"),
        (
            {
                "admin",
                "administrator",
                "permission",
                "privilege",
                "rights",
                "entitlement",
                "role",
                "group",
                "security group",
            },
            "administrative rights",
        ),
        ({"laptop", "printer", "device", "hardware", "dock", "keyboard", "mouse", "monitor", "screen"}, "hardware"),
        ({"hr", "benefit", "leave", "vacation", "payroll"}, "hr support"),
        (
            {"buy", "purchase", "order", "procure", "invoice", "billing", "subscription", "license", "quote", "po"},
            "purchase",
        ),
    ]
    for keywords, label in keyword_map:
        if any(k in lower for k in keywords):
            return label
    if "misc" in lower or "general" in lower or "other" in lower:
        return "miscellaneous"
    return "miscellaneous"


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
    ) -> Dict[str, str]:
        """Attempt to fix bad output by re-prompting the strongest LLM."""
        json_contract = JSONContract()
        try:
            repair_model = get_repair_model()
        except Exception:
            repair_model = model

        valid_json, parsed = json_contract.validate(bad_output)
        if valid_json and parsed:
            is_valid, errors = schema.validate(normalize_output(parsed))
            if is_valid:
                return _ensure_minimum_fields(normalize_output(parsed))
            log.error(f"Schema errors: {errors}")
        else:
            log.error("Contract failure: invalid JSON")

        attempt_output = bad_output
        max_attempts = 1 if callable(repair_model) else self.max_retries
        for attempt in range(max_attempts):
            repair_prompt = (
                f"{prompt}\n\nYour previous output violated the schema. "
                "Fix only the missing/incorrect fields and return valid JSON."
            )
            try:
                attempt_output = _call_model(repair_model, repair_prompt)
            except Exception:
                attempt_output = _call_model(model, repair_prompt)
            ok_json, parsed = json_contract.validate(attempt_output)
            candidate = normalize_output(parsed if ok_json and parsed else attempt_output)
            ok_schema, _ = schema.validate(candidate)
            if ok_schema:
                log.info(f"Repair succeeded on attempt {attempt+1}")
                return _ensure_minimum_fields(candidate)

        # Best-effort fallback
        try:
            data = json.loads(attempt_output)
            if not isinstance(data, dict):
                raise ValueError
        except Exception:
            data = {}

        defaults = {
            "reasoning": "Unable to repair output",
            "final_answer": "No valid answer produced",
            "predicted_category": "unknown",
        }
        for field in schema.required_fields:
            if field not in data or not data[field]:
                data[field] = defaults[field]
        log.error("Repair failed after retries; returning best-effort output")
        return _ensure_minimum_fields(normalize_output(data))


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
            "### Respond with JSON containing final_answer and reasoning.",
        ]
        return "\n\n".join([p for p in parts if p])

    def generate(self, ticket: Dict[str, Any], memory_context: str, kb_snippets: List[str]) -> Dict[str, str]:
        """Generate a response and repair if contracts fail."""
        prompt = self.build_prompt(ticket, memory_context, kb_snippets)
        raw_output = _call_model(self.model_interface, prompt)
        repaired = self.self_repair.repair(self.model_interface, prompt, raw_output, self.contract)
        return repaired
