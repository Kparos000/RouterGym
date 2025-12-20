"""Prompt builder and response generator with contract enforcement."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional

from RouterGym.classifiers.encoder_classifier import EncoderClassifier
from RouterGym.contracts.json_contract import JSONContract, validate_agent_output
from RouterGym.contracts.schema_contract import ALLOWED_CONTEXT_MODES, SchemaContract
from RouterGym.engines.model_registry import get_repair_model, load_models
from RouterGym.label_space import CANONICAL_LABELS, CANONICAL_LABEL_SET, canonicalize_label
from RouterGym.memory import get_memory_class
from RouterGym.memory.base import MemoryRetrieval
from RouterGym.utils.logger import get_logger

log = get_logger(__name__)

CLASS_LABELS = CANONICAL_LABELS

LABELS_LIST_TEXT = ", ".join(CLASS_LABELS)

CONFIDENCE_HIGH_THRESHOLD = 0.80
CONFIDENCE_MEDIUM_THRESHOLD = 0.50


def get_confidence_bucket(conf: float) -> str:
    """Map a numeric confidence into low/medium/high buckets."""
    if conf >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if conf >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


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
    output = None
    if hasattr(model, "generate"):
        try:
            output = model.generate(prompt, max_new_tokens=256, temperature=0.0, top_p=1.0)
        except TypeError:
            output = model.generate(prompt)  # type: ignore[call-arg]
    elif callable(model):
        try:
            output = model(prompt, max_new_tokens=256, temperature=0.0, top_p=1.0)
        except TypeError:
            output = model(prompt)
    else:
        return str(prompt)

    if isinstance(output, str):
        return output
    if isinstance(output, list) and output and isinstance(output[0], dict) and "generated_text" in output[0]:
        return output[0]["generated_text"]
    return str(output)


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


def build_prompt(
    ticket_text: str,
    kb_snippets: List[str],
    classifier_label: str = "",
    confidence_bucket: str = "",
    memory_mode: str = "none",
) -> str:
    """Construct a prompt with KB references and schema guidance for AgentOutput."""
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
        "Respond with STRICT JSON for AgentOutput fields: ticket_id, original_query, rewritten_query, "
        "topic_group, model_name, router_mode, classifier_label, classifier_confidence, "
        "classifier_confidence_bucket, memory_mode, kb_policy_ids (list), kb_categories (list), "
        "final_answer, resolution_steps (list of strings), reasoning, escalation_flags "
        "{needs_human, needs_llm_escalation, policy_gap}, metrics {latency_ms, total_input_tokens, "
        "total_output_tokens, total_cost_usd}."
    )
    prompt_parts.append("Return JSON only without extra text.")
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
        ({"laptop", "printer", "device", "hardware", "dock", "keyboard", "mouse", "monitor", "screen"}, "Hardware"),
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
        context_mode = ticket.get("context_mode", "none")
        kb_section = [f"### KB Reference {i+1}:\n> {s.strip()}" for i, s in enumerate(kb_snippets)]
        memory_section = f"### Memory Context (mode={context_mode}):\n{memory_context}" if memory_context else ""
        kb_intro = (
            "The following KB snippets are internal policies; treat them as primary sources when present."
            if kb_section
            else "No KB context provided; rely only on the ticket and best practices."
        )
        schema_hint = (
            "Output STRICT JSON with fields: ticket_id, original_query, rewritten_query, topic_group, model_name, "
            "router_mode, classifier_label, classifier_confidence, classifier_confidence_bucket, memory_mode, "
            "kb_policy_ids (list), kb_categories (list), final_answer, resolution_steps (list of strings), "
            "reasoning, escalation_flags {needs_human, needs_llm_escalation, policy_gap}, metrics "
            "{latency_ms, total_input_tokens, total_output_tokens, total_cost_usd}."
        )
        parts = [
            base_text,
            memory_section,
            "\n\n".join(kb_section) if kb_section else "",
            kb_intro,
            schema_hint,
            "### Respond with JSON only.",
        ]
        return "\n\n".join([p for p in parts if p])

    def generate(self, ticket: Dict[str, Any], memory_context: str, kb_snippets: List[str]) -> Dict[str, str]:
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
) -> Dict[str, Any]:
    """Run Classify -> Retrieve -> Respond for a single ticket with optional escalation."""

    if memory_mode not in ALLOWED_CONTEXT_MODES:
        raise ValueError(f"Unsupported memory_mode {memory_mode}; allowed: {sorted(ALLOWED_CONTEXT_MODES)}")

    text = str(ticket.get("text") or ticket.get("Document") or ticket.get("document") or "").strip()
    if not text:
        raise ValueError("Ticket text is empty")
    ticket_id = ticket.get("ticket_id") or ticket.get("id") or ticket.get("ticketid") or ""
    ticket_id = str(ticket_id) if ticket_id is not None else ""

    t_start = time.perf_counter()

    # 1) Classification via calibrated encoder
    classifier = EncoderClassifier(head_mode="calibrated", use_lexical_prior=True)
    classify_start = time.perf_counter()
    probabilities = classifier.predict_proba(text)
    classify_latency = (time.perf_counter() - classify_start) * 1000.0
    classifier_label = max(probabilities, key=probabilities.__getitem__)
    classifier_confidence = float(probabilities.get(classifier_label, 0.0))
    classifier_confidence_bucket = get_confidence_bucket(classifier_confidence)

    # 2) Retrieval based on memory_mode
    mem_cls = get_memory_class(memory_mode)
    if mem_cls is None:
        raise ValueError(f"Unknown memory backend for mode {memory_mode}")
    memory = mem_cls()
    try:
        memory.load(ticket)
    except Exception:
        pass
    retrieval: MemoryRetrieval = memory.retrieve(text)
    snippets = []
    if isinstance(retrieval.retrieval_metadata, dict):
        meta_snippets = retrieval.retrieval_metadata.get("snippets", [])
        if isinstance(meta_snippets, list):
            snippets = [s for s in meta_snippets if isinstance(s, dict)]
    kb_policy_ids = _dedupe_preserve(str(s.get("policy_id", "")) for s in snippets if s.get("policy_id"))
    kb_categories = _dedupe_preserve(str(s.get("category", "")) for s in snippets if s.get("category"))
    kb_texts = [str(s.get("text", "")) for s in snippets if s.get("text")]

    # Models: load both base and escalation if needed.
    subset = [base_model_name]
    if escalation_model_name:
        subset.append(escalation_model_name)
    models = load_models(sanity=True, slm_subset=subset)
    base_model = models.get(base_model_name)
    if base_model is None:
        raise RuntimeError(f"Model '{base_model_name}' is not available; check model registry or subset filter.")
    escalation_model = models.get(escalation_model_name) if escalation_model_name else None

    def _call_and_parse(model: Any) -> Dict[str, Any]:
        prompt = build_prompt(
            ticket_text=text,
            kb_snippets=kb_texts,
            classifier_label=classifier_label,
            confidence_bucket=classifier_confidence_bucket,
            memory_mode=memory_mode,
        )
        parsed_output: Dict[str, Any] = {}
        for attempt in range(max_retries):
            raw_output = _call_model(model, prompt)
            parsed_output = _parse_model_output(raw_output)
            try:
                validate_agent_output({"original_query": text, **parsed_output})
                break
            except Exception:
                if attempt == max_retries - 1:
                    parsed_output = {}
                continue
        return parsed_output

    # Routing decision
    chosen_model_name = base_model_name
    parsed_output = _call_and_parse(base_model)
    escalated = False
    escalation_reasons: List[str] = []
    if router_mode == "slm_dominant":
        if escalation_model is None:
            raise ValueError("slm_dominant requires an escalation_model_name (llm1 or llm2).")
        relevance = getattr(retrieval, "relevance_score", 0.0) or retrieval.relevance_score
        final_answer = str(parsed_output.get("final_answer", "") or "")
        resolution_steps = parsed_output.get("resolution_steps", [])
        if not isinstance(resolution_steps, list):
            resolution_steps = []
        low_confidence = classifier_confidence_bucket == "low"
        weak_kb = relevance < 0.10
        no_answer = not final_answer.strip()
        no_steps = len(resolution_steps) == 0
        answer_lower = final_answer.lower()
        ai_disclaimer = any(
            phrase in answer_lower for phrase in ("as an ai", "as a language model", "i cannot", "i'm unable", "i am unable")
        )
        short_answer = len(final_answer.split()) < 10
        if low_confidence:
            escalation_reasons.append("low_confidence")
        if weak_kb:
            escalation_reasons.append("weak_kb")
        if no_answer:
            escalation_reasons.append("no_answer")
        if no_steps:
            escalation_reasons.append("no_steps")
        if ai_disclaimer:
            escalation_reasons.append("ai_disclaimer")
        if short_answer:
            escalation_reasons.append("short_answer")
        needs_escalation = bool(escalation_reasons)
        if needs_escalation:
            parsed_output = _call_and_parse(escalation_model)  # type: ignore[arg-type]
            chosen_model_name = escalation_model_name or base_model_name
            escalated = True
    elif router_mode == "llm_only":
        chosen_model_name = base_model_name  # expected llm
    elif router_mode == "slm_only":
        chosen_model_name = base_model_name
    elif router_mode == "hybrid_specialist":
        # TODO: implement specialist gating; for now, use base_model_name as-is.
        chosen_model_name = base_model_name
    else:
        chosen_model_name = base_model_name

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    resolution_steps = parsed_output.get("resolution_steps", [])
    if not isinstance(resolution_steps, list):
        resolution_steps = []
    default_reasoning = (
        parsed_output.get("reasoning", "")
        or f"Classified as {classifier_label} with confidence {classifier_confidence:.3f}."
    )
    default_answer = parsed_output.get("final_answer", "") or "No valid answer produced"
    parsed_escalation = parsed_output.get("escalation_flags") or {}
    escalation_flags = {
        "needs_human": bool(parsed_escalation.get("needs_human", False)),
        "needs_llm_escalation": bool(escalated),
        "policy_gap": bool(parsed_escalation.get("policy_gap", False)),
        "reasons": escalation_reasons if escalated else [],
    }

    payload: Dict[str, Any] = {
        "ticket_id": ticket_id,
        "original_query": text,
        "rewritten_query": parsed_output.get("rewritten_query", text),
        "topic_group": classifier_label,
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
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            # Optional: add classification latency for debugging
            "classification_latency_ms": classify_latency,
        },
    }
    return validate_agent_output(payload)
