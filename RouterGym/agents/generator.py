"""Prompt builder and response generator with contract enforcement."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from RouterGym.contracts.json_contract import JSONContract, validate_agent_output
from RouterGym.contracts.schema_contract import (
    ALLOWED_CONTEXT_MODES,
    DraftOutputSchema,
    SchemaContract,
)
from RouterGym.engines.telemetry import (
    ModelCallTelemetry,
    aggregate_model_call_telemetry,
    invoke_model_with_telemetry,
    telemetry_records_as_dicts,
)
from RouterGym.label_space import CANONICAL_LABELS, CANONICAL_LABEL_SET, canonicalize_label
from RouterGym.routing.policy import ROUTING_POLICY_VERSION, build_routing_decision
from RouterGym.utils.logger import get_logger

log = get_logger(__name__)

# These names remain patchable for tests, but default to lazy import on first
# runtime use so importing this module does not pull the full classifier/memory/
# model stack into collection-time code paths.
EncoderClassifier: Any = None
resolve_encoder_head_mode: Any = None
get_memory_class: Any = None
load_models: Any = None
get_repair_model: Any = None

CLASS_LABELS = CANONICAL_LABELS

LABELS_LIST_TEXT = ", ".join(CLASS_LABELS)
PLACEHOLDER_FINAL_ANSWER = "No valid answer produced"
GENERATION_INVALID_REASON = "Generation invalid"
MAX_TICKET_REQUEST_WORDS = 40
MAX_FINAL_ANSWER_WORDS = 150
MAX_REASONING_WORDS = 80
MAX_RESOLUTION_STEP_WORDS = 25


@dataclass(frozen=True)
class DraftParseResult:
    """Intermediate model draft plus parse/validation diagnostics."""

    raw_model_response_text: str
    parsed_output_before_validation: Dict[str, Any]
    normalized_output: Dict[str, Any]
    parse_error: Optional[str]
    validation_error: Optional[str]
    generation_valid: bool
    parser_mode: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "raw_model_response_text": self.raw_model_response_text,
            "parsed_output_before_validation": dict(self.parsed_output_before_validation),
            "normalized_output": dict(self.normalized_output),
            "parse_error": self.parse_error,
            "validation_error": self.validation_error,
            "generation_valid": self.generation_valid,
            "parser_mode": self.parser_mode,
        }


def resolve_max_output_tokens(override: Optional[int] = None) -> int:
    """Return the active output cap from an explicit override or environment."""

    if override is not None:
        if int(override) <= 0:
            raise ValueError("max_output_tokens must be > 0")
        return int(override)
    env_value = str(os.getenv("ROUTERGYM_MAX_OUTPUT_TOKENS", "")).strip()
    if env_value:
        try:
            resolved = int(env_value)
        except ValueError as exc:
            raise ValueError("ROUTERGYM_MAX_OUTPUT_TOKENS must be an integer") from exc
        if resolved <= 0:
            raise ValueError("ROUTERGYM_MAX_OUTPUT_TOKENS must be > 0")
        return resolved
    return 1024


def get_confidence_bucket(conf: float) -> str:
    """Map a numeric confidence into low/medium/high buckets."""
    from RouterGym.routing.policy import get_confidence_bucket as routing_get_confidence_bucket

    return routing_get_confidence_bucket(conf)


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def classification_instruction() -> str:
    """High-quality instruction prompt for ticket classification with hard boundaries and examples."""
    return "\n".join(
        [
            "You are an expert IT support triage assistant.",
            "",
            "Classify a single IT support ticket into EXACTLY ONE category:",
            "- Access: login failures, password resets, MFA/SSO/VPN access issues, permission denied for portals.",
            "- Administrative rights: requests for elevated/admin privileges to install or configure software, group or role changes granting admin powers.",
            "- Hardware: physical device or peripheral issues (laptop, monitor, keyboard, mouse, docking station, printer).",
            "- HR Support: payroll/benefits/leave/onboarding/offboarding/employment status/HR portal content questions.",
            "- Purchase: requests to buy/order/procure/renew/pay for hardware, software, licenses, subscriptions, invoices, vendor spend.",
            "- Internal Project: internal initiative/project work (setup, coordination, project-specific tooling).",
            "- Storage: storage capacity, quotas, disk/drive space, backups/archival.",
            "- Miscellaneous: genuinely unclear, mixed, or off-topic IT questions. Only use 'Miscellaneous' if none of the above clearly apply.",
            "",
            "Hard boundary examples (resolve ambiguity):",
            '- "Need access to HR portal" -> Access (NOT HR Support; portal access issue).',
            '- "Need admin rights to install HR payroll tool" -> Administrative rights (NOT HR Support).',
            '- "Question about benefits enrollment" -> HR Support (NOT Access).',
            '- "Need to order new monitors for the team" -> Purchase (NOT Hardware; it is a buying request).',
            '- "Create repo for internal project Apollo" -> Internal Project (NOT Miscellaneous).',
            '- "Increase my OneDrive quota" -> Storage (NOT Miscellaneous).',
            "",
            "Think step-by-step before answering:",
            "1) Restate the main request briefly.",
            "2) Identify strong cues (order/buy/purchase, access/login/password, admin/rights, benefits/payroll/leave, device names).",
            "3) Pick the SINGLE best category that matches the primary intent.",
            "4) Only use 'Miscellaneous' if no other label reasonably fits after re-reading.",
            "",
            "Respond with STRICT JSON only:",
            '{"reasoning": "<short explanation>", "category": "<one of: Access, Administrative rights, Hardware, HR Support, Purchase, Internal Project, Storage, Miscellaneous>"}',
        ]
    )


def _call_model(model: Any, prompt: str) -> str:
    """Invoke a model or pipeline and normalize the output to string."""
    model_key = str(getattr(model, "model_key", getattr(model, "name", "llm1")) or "llm1")
    output_text, _ = invoke_model_with_telemetry(model, prompt, model_key=model_key)
    return output_text


def _parse_model_output(text: str) -> Dict[str, Any]:
    """Parse model output into a dict if possible, otherwise empty dict."""
    contract = JSONContract()
    ok, parsed = contract.validate(text)
    if ok and isinstance(parsed, dict):
        return parsed
    fragment = _extract_json_fragment(text)
    if isinstance(fragment, dict):
        return fragment
    return {}


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


def _compact_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate_words(text: Any, max_words: int) -> str:
    compact = _compact_text(text)
    if not compact or max_words <= 0:
        return compact
    words = compact.split()
    if len(words) <= max_words:
        return compact
    return " ".join(words[:max_words]).rstrip(" ,;:")


def _looks_json_like(text: str) -> bool:
    stripped = str(text or "").lstrip()
    return stripped.startswith("{") or ('"' in stripped and ":" in stripped)


def _derive_ticket_request(value: Any, fallback_text: str = "") -> str:
    source = _compact_text(value) or _compact_text(fallback_text)
    if not source:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", source, maxsplit=1)[0]
    candidate = first_sentence or source
    return _truncate_words(candidate, MAX_TICKET_REQUEST_WORDS)


def _normalize_resolution_steps(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_compact_text(step) for step in value if _compact_text(step)]
    if isinstance(value, str):
        return [step for step in (_compact_text(line) for line in value.splitlines()) if step]
    return []


def _extract_json_string_field(text: str, *field_names: str) -> str:
    for field_name in field_names:
        pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"'
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        try:
            return _compact_text(json.loads(f'"{match.group(1)}"'))
        except Exception:
            return _compact_text(match.group(1))
    return ""


def _extract_json_array_field(text: str, field_name: str) -> List[str]:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    inner = match.group(1)
    try:
        parsed = json.loads(f"[{inner}]")
    except Exception:
        parsed = re.findall(r'"((?:\\.|[^"\\])*)"', inner)
    return _normalize_resolution_steps(parsed)


def _extract_json_object_field(text: str, field_name: str) -> Dict[str, Any]:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*(\{{.*?\}})'
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_resolution_steps_from_text(text: str) -> List[str]:
    steps: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?:[-*•]|\d+[\).\:-]?)\s+(.*)$", line)
        if match:
            step = _compact_text(match.group(1))
            if step:
                steps.append(step)
    return steps


def _extract_answer_paragraph(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped or _looks_json_like(stripped):
        return ""
    prose_lines: List[str] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            if prose_lines:
                break
            continue
        if re.match(r"^(?:[-*•]|\d+[\).\:-]?)\s+", line):
            break
        if ":" in line:
            key, value = line.split(":", 1)
            key_normalized = _compact_text(key).lower()
            if key_normalized in {
                "ticket_request",
                "rewritten_query",
                "original_query",
                "final_answer",
                "answer",
                "reasoning",
                "predicted_category",
                "category",
                "resolution_steps",
            }:
                if key_normalized in {"final_answer", "answer"} and _compact_text(value):
                    return _compact_text(value)
                continue
        prose_lines.append(line)
    return _compact_text(" ".join(prose_lines))


def _normalize_escalation_flags(
    value: Any, *, default_needs_llm_escalation: bool = False
) -> Dict[str, Any]:
    flags = value if isinstance(value, dict) else {}
    reasons = flags.get("reasons", [])
    if reasons is None:
        reasons = []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "needs_human": bool(flags.get("needs_human", False)),
        "needs_llm_escalation": bool(
            flags.get("needs_llm_escalation", default_needs_llm_escalation)
        ),
        "policy_gap": bool(flags.get("policy_gap", False)),
        "reasons": [_compact_text(reason) for reason in reasons if _compact_text(reason)],
    }


def _is_placeholder_answer(text: Any) -> bool:
    return _compact_text(text) == PLACEHOLDER_FINAL_ANSWER


def _normalize_draft_output(
    raw_output: str,
    parsed_output: Dict[str, Any],
    classifier_label: str,
    ticket_text: str,
) -> Tuple[Dict[str, Any], str]:
    parser_mode = "json" if parsed_output else "natural_language"
    stripped = str(raw_output or "").strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    labeled_values: Dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_normalized = _compact_text(key).lower()
        labeled_values[key_normalized] = _compact_text(value)

    source = dict(parsed_output or {})
    ticket_request = _derive_ticket_request(
        source.get("ticket_request")
        or source.get("rewritten_query")
        or source.get("original_query")
        or labeled_values.get("ticket_request")
        or labeled_values.get("rewritten_query")
        or labeled_values.get("original_query")
        or _extract_json_string_field(
            stripped, "ticket_request", "rewritten_query", "original_query"
        ),
        fallback_text=ticket_text,
    )
    resolution_steps = (
        _normalize_resolution_steps(source.get("resolution_steps", []))
        or _normalize_resolution_steps(labeled_values.get("resolution_steps", ""))
        or _extract_json_array_field(stripped, "resolution_steps")
        or _extract_resolution_steps_from_text(stripped)
    )
    resolution_steps = [
        _truncate_words(step, MAX_RESOLUTION_STEP_WORDS)
        for step in resolution_steps
        if _compact_text(step)
    ]
    final_answer = _compact_text(
        source.get("final_answer")
        or labeled_values.get("final_answer")
        or labeled_values.get("answer")
        or _extract_json_string_field(stripped, "final_answer", "answer")
        or _extract_answer_paragraph(stripped)
    )
    if final_answer and not re.search(r"[A-Za-z0-9]", final_answer):
        final_answer = ""
    final_answer = _truncate_words(final_answer, MAX_FINAL_ANSWER_WORDS)
    reasoning = _truncate_words(
        source.get("reasoning")
        or labeled_values.get("reasoning")
        or _extract_json_string_field(stripped, "reasoning"),
        MAX_REASONING_WORDS,
    )
    if not reasoning and final_answer and not _looks_json_like(stripped):
        reasoning = "Normalized from non-JSON model output."

    predicted = _normalize_category(
        source.get("predicted_category")
        or source.get("category")
        or labeled_values.get("predicted_category")
        or labeled_values.get("category")
        or _extract_json_string_field(stripped, "predicted_category", "category")
        or classifier_label,
        context=f"{ticket_request} {final_answer} {reasoning}",
    )
    escalation_flags = _normalize_escalation_flags(
        source.get("escalation_flags") or _extract_json_object_field(stripped, "escalation_flags")
    )
    return (
        {
            "ticket_request": ticket_request,
            "final_answer": final_answer,
            "reasoning": reasoning,
            "predicted_category": predicted or classifier_label,
            "resolution_steps": resolution_steps,
            "rewritten_query": ticket_request,
            "escalation_flags": escalation_flags,
        },
        parser_mode,
    )


def _parse_draft_output(
    raw_output: str, classifier_label: str, ticket_text: str
) -> DraftParseResult:
    raw_text = str(raw_output or "")
    contract = JSONContract()
    ok_json, parsed_json = contract.validate(raw_text)
    parsed_output: Dict[str, Any] = {}
    if ok_json and isinstance(parsed_json, dict):
        parsed_output = parsed_json
    else:
        fragment = _extract_json_fragment(raw_text)
        if isinstance(fragment, dict):
            parsed_output = fragment
    parse_error = None if ok_json or parsed_output else "Model output is not valid JSON."
    normalized_output, parser_mode = _normalize_draft_output(
        raw_text, parsed_output, classifier_label, ticket_text
    )
    draft_ok, draft_errors = DraftOutputSchema().validate(normalized_output)
    if not raw_text.strip():
        draft_errors.append("raw_model_response_text is empty")
        draft_ok = False
    if _is_placeholder_answer(normalized_output.get("final_answer", "")):
        draft_errors.append("final_answer is a placeholder")
        draft_ok = False
    validation_error = "; ".join(draft_errors) if draft_errors else None
    return DraftParseResult(
        raw_model_response_text=raw_text,
        parsed_output_before_validation=dict(parsed_output),
        normalized_output=normalized_output,
        parse_error=parse_error,
        validation_error=validation_error,
        generation_valid=draft_ok,
        parser_mode=parser_mode,
    )


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
    ticket_request = _derive_ticket_request(
        parsed.get("ticket_request")
        or parsed.get("rewritten_query")
        or parsed.get("original_query")
    )
    final_answer = str(parsed.get("final_answer", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()
    predicted = _normalize_category(str(raw_pred), context=f"{final_answer} {reasoning}")
    if not predicted:
        predicted = "unknown"
    return {
        "ticket_request": ticket_request,
        "final_answer": final_answer,
        "reasoning": reasoning,
        "predicted_category": predicted,
    }


def _ensure_minimum_fields(data: Dict[str, str]) -> Dict[str, str]:
    """Guarantee required fields are present and non-empty."""
    return {
        "ticket_request": data.get("ticket_request") or "",
        "final_answer": data.get("final_answer") or "No valid answer produced",
        "reasoning": data.get("reasoning") or "",
        "predicted_category": data.get("predicted_category") or "unknown",
    }


def build_prompt(
    ticket_text: str,
    kb_snippets: List[str],
    classifier_label: str = "",
    confidence_bucket: str = "",
    memory_mode: str = "none",
) -> str:
    """Construct a prompt for the minimal model-generated agent payload."""
    prompt_parts = [f"Ticket:\n{ticket_text.strip()}" if ticket_text else "Ticket: (missing)"]
    if classifier_label:
        prompt_parts.append(
            f"Classifier prediction: {classifier_label} (confidence bucket: {confidence_bucket or 'unknown'})"
        )
    if kb_snippets:
        for idx, snippet in enumerate(kb_snippets, start=1):
            prompt_parts.append(f"### KB Reference {idx}:\n> {snippet.strip()}")
        prompt_parts.append("KB snippets are internal policies; treat them as primary references.")
    else:
        prompt_parts.append("No KB context provided; rely on ticket details and best practices.")
    prompt_parts.append(f"Memory mode: {memory_mode}")
    prompt_parts.append(
        "Return STRICT JSON only with EXACTLY these keys and no others: "
        "ticket_request, final_answer, reasoning, predicted_category, resolution_steps, escalation_flags."
    )
    prompt_parts.append(
        "The entire response must be one JSON object. Start with { and end with }. "
        "Do not include explanations, markdown, comments, or text outside the JSON object."
    )
    prompt_parts.append(
        "ticket_request: one clean one-sentence restatement of the support request, under 40 words, with no diagnosis or benchmark metadata."
    )
    prompt_parts.append("final_answer: under 150 words.")
    prompt_parts.append("reasoning: under 80 words.")
    prompt_parts.append("resolution_steps: 2 to 5 short actionable steps, each under 25 words.")
    prompt_parts.append("escalation_flags.reasons must be an array, even if empty.")
    prompt_parts.append(
        "Do NOT return benchmark metadata or extra keys such as ticket_id, original_query, rewritten_query, "
        "topic_group, model_name, router_mode, classifier_label, classifier_confidence, "
        "classifier_confidence_bucket, memory_mode, kb_policy_ids, kb_categories, metrics, token counts, "
        "costs, latency, raw_model_response_text, generation_valid, success, or error."
    )
    prompt_parts.append("Return JSON only. No markdown fences. No prose before or after the JSON.")
    prompt_parts.append(
        '{"ticket_request":"...","final_answer":"...","reasoning":"...","predicted_category":"...","resolution_steps":["...","..."],"escalation_flags":{"needs_human":false,"needs_llm_escalation":false,"policy_gap":false,"reasons":[]}}'
    )
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
        return canonicalize_label("Access")
    if contains_any(strong_hardware):
        return canonicalize_label("Hardware")

    if text in CANONICAL_LABEL_SET:
        return canonicalize_label(text)

    keyword_map = [
        (
            {"admin", "administrator", "permission", "privilege", "rights", "entitlement", "group"},
            "Administrative rights",
        ),
        ({"hr", "benefit", "leave", "vacation", "payroll"}, "HR Support"),
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
            "Purchase",
        ),
    ]
    for keywords, label in keyword_map:
        if any(k in combined for k in keywords):
            return canonicalize_label(label)

    if "internal project" in combined:
        return canonicalize_label("Internal Project")
    if "storage" in combined or "quota" in combined or "disk" in combined:
        return canonicalize_label("Storage")
    if "misc" in combined or "general" in combined or "other" in combined:
        return canonicalize_label("Miscellaneous")
    # If no strong match, prefer Miscellaneous over unknown to avoid empty labels while still canonicalizing.
    try:
        return canonicalize_label(text)
    except RuntimeError:
        return "Miscellaneous"


def infer_category_from_text(text: str) -> str:
    """Heuristic mapping from ticket text to canonical labels."""
    lower = (text or "").lower()
    keyword_map = [
        ({"login", "password", "account", "access", "credential", "mfa", "sso"}, "Access"),
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
            "Administrative rights",
        ),
        (
            {
                "laptop",
                "printer",
                "device",
                "hardware",
                "dock",
                "keyboard",
                "mouse",
                "monitor",
                "screen",
            },
            "Hardware",
        ),
        ({"hr", "benefit", "leave", "vacation", "payroll"}, "HR Support"),
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
            "Purchase",
        ),
        ({"internal project", "project work"}, "Internal Project"),
        ({"storage", "quota", "disk", "drive"}, "Storage"),
    ]
    for keywords, label in keyword_map:
        if any(k in lower for k in keywords):
            return canonicalize_label(label)
    if "misc" in lower or "general" in lower or "other" in lower:
        return canonicalize_label("Miscellaneous")
    return "Miscellaneous"


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
            repair_factory: Any = get_repair_model
            if repair_factory is None:
                from RouterGym.engines.model_registry import get_repair_model as repair_factory

            repair_model = repair_factory()
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
                log.info(f"Repair succeeded on attempt {attempt + 1}")
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

    def build_prompt(
        self, ticket: Dict[str, Any], memory_context: str, kb_snippets: List[str]
    ) -> str:
        """Build a structured prompt from ticket, memory, and KB."""
        base_text = ticket.get("text", "")
        context_mode = ticket.get("context_mode", "none")
        kb_section = [
            f"### KB Reference {i + 1}:\n> {s.strip()}" for i, s in enumerate(kb_snippets)
        ]
        memory_section = (
            f"### Memory Context (mode={context_mode}):\n{memory_context}" if memory_context else ""
        )
        kb_intro = (
            "The following KB snippets are internal policies; treat them as primary sources when present."
            if kb_section
            else "No KB context provided; rely only on the ticket and best practices."
        )
        schema_hint = (
            "Return STRICT JSON only with exactly these keys: ticket_request, final_answer, reasoning, "
            "predicted_category, resolution_steps, escalation_flags. Do not return benchmark metadata or extra keys."
        )
        parts = [
            base_text,
            memory_section,
            "\n\n".join(kb_section) if kb_section else "",
            kb_intro,
            schema_hint,
            "ticket_request under 40 words; final_answer under 150 words; reasoning under 80 words; "
            "resolution_steps must contain 2 to 5 short actionable steps; no markdown fences.",
        ]
        return "\n\n".join([p for p in parts if p])

    def generate(
        self, ticket: Dict[str, Any], memory_context: str, kb_snippets: List[str]
    ) -> Dict[str, str]:
        """Generate a response and repair if contracts fail."""
        prompt = self.build_prompt(ticket, memory_context, kb_snippets)
        raw_output = _call_model(self.model_interface, prompt)
        repaired = self.self_repair.repair(self.model_interface, prompt, raw_output, self.contract)
        return repaired


def run_ticket_pipeline(
    ticket: Dict[str, Any],
    router_mode: str,
    memory_mode: str,
    base_model_name: str,
    escalation_model_name: Optional[str] = None,
    max_retries: int = 2,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Classify -> Retrieve -> Respond for a single ticket with optional escalation."""

    if memory_mode not in ALLOWED_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported memory_mode {memory_mode}; allowed: {sorted(ALLOWED_CONTEXT_MODES)}"
        )

    text = str(ticket.get("text") or ticket.get("Document") or ticket.get("document") or "").strip()
    if not text:
        raise ValueError("Ticket text is empty")
    ticket_id = ticket.get("ticket_id") or ticket.get("id") or ticket.get("ticketid") or ""
    ticket_id = str(ticket_id) if ticket_id is not None else ""

    t_start = time.perf_counter()
    resolved_max_output_tokens = resolve_max_output_tokens(max_output_tokens)

    # 1) Classification via calibrated encoder
    encoder_classifier_cls: Any = EncoderClassifier
    if encoder_classifier_cls is None:
        from RouterGym.classifiers.encoder_classifier import (
            EncoderClassifier as encoder_classifier_cls,
        )

    load_models_fn: Any = load_models
    if load_models_fn is None:
        from RouterGym.engines.model_registry import load_models as load_models_fn

    resolve_encoder_head_mode_fn: Any = resolve_encoder_head_mode
    if resolve_encoder_head_mode_fn is None:
        from RouterGym.classifiers.encoder_classifier import (
            resolve_encoder_head_mode as resolve_encoder_head_mode_fn,
        )

    get_memory_class_fn: Any = get_memory_class
    if get_memory_class_fn is None:
        from RouterGym.memory import get_memory_class as get_memory_class_fn

    classifier = encoder_classifier_cls(
        head_mode=resolve_encoder_head_mode_fn(),
        use_lexical_prior=True,
    )
    classify_start = time.perf_counter()
    probabilities = classifier.predict_proba(text)
    classify_latency = (time.perf_counter() - classify_start) * 1000.0
    classifier_label = max(probabilities, key=probabilities.__getitem__)
    classifier_confidence = float(probabilities.get(classifier_label, 0.0))
    classifier_confidence_bucket = get_confidence_bucket(classifier_confidence)

    # 2) Retrieval based on memory_mode
    mem_cls = get_memory_class_fn(memory_mode)
    if mem_cls is None:
        raise ValueError(f"Unknown memory backend for mode {memory_mode}")
    memory = mem_cls()
    try:
        memory.load(ticket)
    except Exception:
        pass
    retrieval = memory.retrieve(text)
    snippets = []
    if isinstance(retrieval.retrieval_metadata, dict):
        meta_snippets = retrieval.retrieval_metadata.get("snippets", [])
        if isinstance(meta_snippets, list):
            snippets = [s for s in meta_snippets if isinstance(s, dict)]
    kb_policy_ids = _dedupe_preserve(
        str(s.get("policy_id", "")) for s in snippets if s.get("policy_id")
    )
    kb_categories = _dedupe_preserve(
        str(s.get("category", "")) for s in snippets if s.get("category")
    )
    kb_texts = [str(s.get("text", "")) for s in snippets if s.get("text")]

    # Models: load both base and escalation if needed.
    subset = [base_model_name]
    if escalation_model_name:
        subset.append(escalation_model_name)
    models = load_models_fn(sanity=True, slm_subset=subset)
    base_model = models.get(base_model_name)
    if base_model is None:
        raise RuntimeError(
            f"Model '{base_model_name}' is not available; check model registry or subset filter."
        )
    escalation_model = models.get(escalation_model_name) if escalation_model_name else None

    def _call_and_parse(
        model: Any, model_key: str
    ) -> Tuple[DraftParseResult, List[ModelCallTelemetry]]:
        prompt = build_prompt(
            ticket_text=text,
            kb_snippets=kb_texts,
            classifier_label=classifier_label,
            confidence_bucket=classifier_confidence_bucket,
            memory_mode=memory_mode,
        )
        telemetry_records: List[ModelCallTelemetry] = []
        last_result = DraftParseResult(
            raw_model_response_text="",
            parsed_output_before_validation={},
            normalized_output={
                "ticket_request": _derive_ticket_request(text),
                "final_answer": "",
                "reasoning": "",
                "predicted_category": classifier_label,
                "resolution_steps": [],
                "rewritten_query": _derive_ticket_request(text),
                "escalation_flags": {},
            },
            parse_error="Model returned an empty response.",
            validation_error="Draft output must include ticket_request, a non-empty final_answer, and at least one resolution step",
            generation_valid=False,
            parser_mode="empty",
        )
        for attempt in range(max_retries):
            raw_output, telemetry = invoke_model_with_telemetry(
                model,
                prompt,
                model_key=model_key,
                max_new_tokens=resolved_max_output_tokens,
            )
            telemetry_records.append(telemetry)
            last_result = _parse_draft_output(raw_output, classifier_label, text)
            if last_result.generation_valid:
                break
        return last_result, telemetry_records

    # Routing decision
    chosen_model_name = base_model_name
    base_draft, model_call_telemetry = _call_and_parse(base_model, base_model_name)
    parsed_output = dict(base_draft.normalized_output)
    initial_steps = parsed_output.get("resolution_steps", [])
    if not isinstance(initial_steps, list):
        initial_steps = []
    retrieval_score = getattr(retrieval, "relevance_score", 0.0) or retrieval.relevance_score
    initial_answer = str(parsed_output.get("final_answer", "") or "")
    initial_schema_valid, _ = SchemaContract().validate(
        {
            "final_answer": initial_answer,
            "reasoning": str(parsed_output.get("reasoning", "") or ""),
            "predicted_category": str(
                parsed_output.get("predicted_category", classifier_label) or ""
            ),
        }
    )
    routing_decision = build_routing_decision(
        router_mode=router_mode,
        text=text,
        base_model_name=base_model_name,
        escalation_model_name=escalation_model_name,
        category=classifier_label,
        classifier_confidence=classifier_confidence,
        retrieval_score=retrieval_score,
        final_answer=initial_answer,
        resolution_steps_count=len(initial_steps),
        schema_valid=initial_schema_valid,
    )
    escalated = False
    escalation_reasons = list(routing_decision.escalation_reasons)
    chosen_draft = base_draft
    escalation_draft: Optional[DraftParseResult] = None
    should_execute_escalation = (
        routing_decision.escalated
        and escalation_model is not None
        and routing_decision.final_model != base_model_name
    )
    if router_mode == "slm_dominant" and routing_decision.escalated and escalation_model is None:
        raise ValueError("slm_dominant requires an escalation_model_name (llm1 or llm2).")
    if should_execute_escalation:
        escalation_draft, escalation_telemetry = _call_and_parse(
            escalation_model,  # type: ignore[arg-type]
            escalation_model_name or base_model_name,
        )
        chosen_draft = escalation_draft
        parsed_output = dict(escalation_draft.normalized_output)
        model_call_telemetry.extend(escalation_telemetry)
        chosen_model_name = escalation_model_name or base_model_name
        escalated = True

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    ticket_request = _derive_ticket_request(
        parsed_output.get("ticket_request") or parsed_output.get("rewritten_query") or text,
        fallback_text=text,
    )
    resolution_steps = parsed_output.get("resolution_steps", [])
    if not isinstance(resolution_steps, list):
        resolution_steps = []
    invalid_reason = (
        chosen_draft.validation_error or chosen_draft.parse_error or GENERATION_INVALID_REASON
    )
    default_reasoning = parsed_output.get("reasoning", "")
    if not default_reasoning:
        if chosen_draft.generation_valid:
            default_reasoning = (
                f"Classified as {classifier_label} with confidence {classifier_confidence:.3f}."
            )
        else:
            default_reasoning = f"{GENERATION_INVALID_REASON}: {invalid_reason}"
    default_answer = parsed_output.get("final_answer", "") or PLACEHOLDER_FINAL_ANSWER
    placeholder_answer = default_answer == PLACEHOLDER_FINAL_ANSWER
    has_real_final_answer = bool(default_answer.strip()) and not placeholder_answer
    has_resolution_steps = bool(resolution_steps)
    raw_response_saved = bool(chosen_draft.raw_model_response_text.strip())
    telemetry_summary = aggregate_model_call_telemetry(model_call_telemetry)
    parsed_escalation = parsed_output.get("escalation_flags") or {}
    escalation_flags = {
        "needs_human": bool(parsed_escalation.get("needs_human", False)),
        "needs_llm_escalation": bool(escalated),
        "policy_gap": bool(parsed_escalation.get("policy_gap", False)),
        "reasons": escalation_reasons if routing_decision.escalated else [],
    }
    generation_debug: Dict[str, Any] = {
        "selected_stage": "escalation" if escalated else "base",
        "base_stage": base_draft.as_dict(),
    }
    if escalation_draft is not None:
        generation_debug["escalation_stage"] = escalation_draft.as_dict()

    raw_generated_predicted_category = chosen_draft.parsed_output_before_validation.get(
        "predicted_category"
    ) or chosen_draft.parsed_output_before_validation.get("category")
    generated_predicted_category = (
        _normalize_category(str(raw_generated_predicted_category))
        if raw_generated_predicted_category
        else None
    )
    payload: Dict[str, Any] = {
        "ticket_id": ticket_id,
        "original_query": text,
        "ticket_request": ticket_request,
        "rewritten_query": ticket_request,
        "topic_group": classifier_label,
        "predicted_category": classifier_label,
        "classifier_predicted_category": classifier_label,
        "prediction_source": classifier.backend_name,
        "generated_predicted_category": generated_predicted_category,
        "model_name": chosen_model_name,
        "router_mode": router_mode,
        "base_model_name": base_model_name,
        "escalation_model_name": escalation_model_name,
        "classifier_label": classifier_label,
        "classifier_confidence": classifier_confidence,
        "classifier_confidence_bucket": classifier_confidence_bucket,
        "classifier_backend": classifier.backend_name,
        "classification": {
            "label": classifier_label,
            "confidence": classifier_confidence,
            "confidence_bucket": classifier_confidence_bucket,
        },
        "memory_mode": memory_mode,
        "context_mode": memory_mode,
        "kb_policy_ids": kb_policy_ids,
        "kb_categories": kb_categories,
        "resolution_steps": resolution_steps,
        "final_answer": default_answer,
        "reasoning": default_reasoning,
        "escalation_flags": escalation_flags,
        "metrics": {
            "latency_ms": total_latency_ms,
            "max_output_tokens": resolved_max_output_tokens,
            "total_input_tokens": telemetry_summary["total_input_tokens"],
            "total_output_tokens": telemetry_summary["total_output_tokens"],
            "total_tokens": telemetry_summary["total_tokens"],
            "total_input_cost_usd": telemetry_summary["total_input_cost_usd"],
            "total_output_cost_usd": telemetry_summary["total_output_cost_usd"],
            "total_cost_usd": telemetry_summary["total_cost_usd"],
            "slm_input_tokens": telemetry_summary["slm_input_tokens"],
            "slm_output_tokens": telemetry_summary["slm_output_tokens"],
            "slm_total_tokens": telemetry_summary["slm_total_tokens"],
            "slm_cost_usd": telemetry_summary["slm_cost_usd"],
            "llm_input_tokens": telemetry_summary["llm_input_tokens"],
            "llm_output_tokens": telemetry_summary["llm_output_tokens"],
            "llm_total_tokens": telemetry_summary["llm_total_tokens"],
            "llm_cost_usd": telemetry_summary["llm_cost_usd"],
            "token_count_method_summary": telemetry_summary["token_count_method_summary"],
            "pricing_version": telemetry_summary["pricing_version"],
            "pricing_source": telemetry_summary["pricing_source"],
            # Optional: add classification latency for debugging
            "classification_latency_ms": classify_latency,
        },
        "model_call_telemetry": telemetry_records_as_dicts(model_call_telemetry),
        "max_output_tokens": resolved_max_output_tokens,
        "total_input_tokens": telemetry_summary["total_input_tokens"],
        "total_output_tokens": telemetry_summary["total_output_tokens"],
        "total_tokens": telemetry_summary["total_tokens"],
        "total_cost_usd": telemetry_summary["total_cost_usd"],
        "slm_cost_usd": telemetry_summary["slm_cost_usd"],
        "llm_cost_usd": telemetry_summary["llm_cost_usd"],
        "token_count_method_summary": telemetry_summary["token_count_method_summary"],
        "pricing_version": telemetry_summary["pricing_version"],
        "pricing_source": telemetry_summary["pricing_source"],
        "initial_model": routing_decision.initial_model,
        "final_model": chosen_model_name,
        "escalated": bool(escalated),
        "escalation_reasons": escalation_reasons if routing_decision.escalated else [],
        "confidence_bucket": routing_decision.confidence_bucket,
        "retrieval_score": routing_decision.retrieval_score,
        "routing_policy_version": ROUTING_POLICY_VERSION,
        "router_confidence_score": routing_decision.router_confidence_score,
        "router_decision_reason": routing_decision.router_decision_reason,
        "raw_model_response_text": chosen_draft.raw_model_response_text,
        "parse_error": chosen_draft.parse_error,
        "validation_error": chosen_draft.validation_error,
        "parsed_output_before_validation": dict(chosen_draft.parsed_output_before_validation),
        "raw_model_response_text_base": base_draft.raw_model_response_text,
        "raw_model_response_text_escalation": (
            escalation_draft.raw_model_response_text if escalation_draft is not None else ""
        ),
        "generation_parser_mode": chosen_draft.parser_mode,
        "generation_valid": bool(chosen_draft.generation_valid),
        "generation_invalid_reason": ("" if chosen_draft.generation_valid else invalid_reason),
        "has_real_final_answer": has_real_final_answer,
        "has_resolution_steps": has_resolution_steps,
        "placeholder_answer": placeholder_answer,
        "raw_response_saved": raw_response_saved,
        "generation_debug": generation_debug,
    }
    try:
        return validate_agent_output(payload)
    except Exception as exc:
        payload["generation_valid"] = False
        merged_validation_error = "; ".join(
            part
            for part in [
                (payload.get("validation_error") or "").strip(),
                str(exc).strip(),
            ]
            if part
        )
        payload["validation_error"] = merged_validation_error
        payload["generation_invalid_reason"] = merged_validation_error or GENERATION_INVALID_REASON
        payload["final_answer"] = payload.get("final_answer") or PLACEHOLDER_FINAL_ANSWER
        payload["placeholder_answer"] = _is_placeholder_answer(payload.get("final_answer"))
        payload["has_real_final_answer"] = not bool(payload["placeholder_answer"])
        payload["has_resolution_steps"] = False
        payload["reasoning"] = (
            payload.get("reasoning")
            or f"{GENERATION_INVALID_REASON}: {payload['generation_invalid_reason']}"
        )
        return payload
