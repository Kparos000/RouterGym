"""Heuristic SLM-first routing policy.

This router prefers an SLM by default and only escalates to the LLM when
simple hand-crafted rules mark the ticket as high-risk or too complex.
The heuristics are transparent and rely on ticket length, coarse
category hardness, and (optional) classifier confidence metadata.
"""

from typing import Any, Dict, Optional

from RouterGym.routing.base import BaseRouter
from RouterGym.agents.generator import (
    CLASS_LABELS,
    SchemaContract,
    SelfRepair,
    classification_instruction,
    infer_category_from_text,
    normalize_output,
    _call_model,
)
from RouterGym.contracts.json_contract import JSONContract
from RouterGym.utils.kb_utils import coerce_kb_hits


def _infer_category(text: str, default: str = "") -> str:
    lower = text.lower()
    if "vpn" in lower or "network" in lower:
        return "network"
    if "password" in lower or "login" in lower or "access" in lower:
        return "access"
    if "hr" in lower or "leave" in lower:
        return "hr_support"
    if "printer" in lower or "laptop" in lower or "hardware" in lower:
        return "hardware"
    return default or "unknown"


class SLMDominantRouter(BaseRouter):
    """Prefer SLM routes and escalate on contract or confidence failures."""

    def route(
        self,
        ticket: Dict[str, Any],
        kb: Optional[Any] = None,
        models: Optional[Dict[str, Any]] = None,
        memory: Optional[Any] = None,
        force_llm: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        text = ticket.get("text", "") if isinstance(ticket, dict) else str(ticket)
        models = models or {}
        slm = models.get("slm1") or models.get("slm2")
        llm = models.get("llm1") or models.get("llm2")
        steps = []

        if memory:
            memory.add(text)
        memory_context = memory.get_context() if memory else ""
        kb_snippets = []
        if kb is not None:
            try:
                hits = coerce_kb_hits(kb.retrieve(text, top_k=3) if hasattr(kb, "retrieve") else [])
                kb_snippets = [h["text"] for h in hits if h["text"]]
            except Exception:
                kb_snippets = []

        prompt_parts = [text]
        if memory_context:
            prompt_parts.append(f"[Memory]\n{memory_context}")
        if kb_snippets:
            prompt_parts.append("\n\n".join([f"[KB]\n{snip}" for snip in kb_snippets]))
        prompt_parts.append(classification_instruction())
        prompt_parts.append(
            f"Use predicted_category from: {', '.join(CLASS_LABELS)}. Return JSON only."
        )
        prompt = "\n\n".join(prompt_parts)

        confidence = float(ticket.get("classifier_confidence", 0.8 if ticket.get("category") else 0.4))
        contract = SchemaContract()

        classifier_label = str(ticket.get("category") or "")
        escalate, decision_reason, risk_score = should_escalate_heuristic(
            text,
            category=classifier_label or _infer_category(text),
            classifier_confidence=confidence,
        )
        if force_llm:
            escalate = True
            decision_reason = "escalate: force_llm"
            risk_score = 1.0

        initial_model = llm if escalate else slm
        raw_output = _call_model(initial_model, prompt) if initial_model else ""
        jc = JSONContract()
        sr = SelfRepair()
        ok_json, parsed = jc.validate(raw_output) if isinstance(raw_output, str) else (True, raw_output if isinstance(raw_output, dict) else None)
        schema_ok = False
        final_output = normalize_output(raw_output)
        if ok_json and parsed:
            schema_ok, _ = contract.validate(normalize_output(parsed))
        fallback = False
        if not schema_ok:
            fallback = True
            raw_output = _call_model(llm, prompt) if llm else raw_output
            ok_json, parsed = jc.validate(raw_output) if isinstance(raw_output, str) else (True, raw_output if isinstance(raw_output, dict) else None)
            schema_ok = False
            if ok_json and parsed:
                schema_ok, _ = contract.validate(normalize_output(parsed))
            if not schema_ok and llm is not None:
                repaired = sr.repair(llm, prompt, raw_output, contract)
                final_output = normalize_output(repaired)
            else:
                final_output = normalize_output(parsed if parsed else raw_output)
        else:
            final_output = normalize_output(parsed if parsed else raw_output)

        if not final_output.get("predicted_category"):
            final_output["predicted_category"] = infer_category_from_text(text)

        steps.append({"stage": "generate", "output": final_output, "confidence": confidence})
        if fallback:
            steps.append({"stage": "fallback_llm", "output": final_output})
            decision_reason = decision_reason + " + schema_fallback"
            risk_score = max(risk_score, 0.5)

        return {
            "strategy": "slm_dominant",
            "target_model": "llm" if fallback else "slm",
            "model_used": "llm" if fallback else "slm",
            "steps": steps,
            "final_output": final_output,
            "json_valid": bool(ok_json),
            "schema_valid": bool(schema_ok),
            "predicted_category": final_output.get("predicted_category", ""),
            "kb_attached": bool(kb_snippets),
            "kb_snippets": kb_snippets,
            "prompt": prompt,
            "router_confidence_score": risk_score,
            "router_decision_reason": decision_reason,
        }
HARD_CATEGORIES = {"security", "benefits", "legal", "compliance"}
LENGTH_THRESHOLD = 512  # characters
LOW_CONFIDENCE = 0.3


def should_escalate_heuristic(
    text: str,
    category: str = "",
    classifier_confidence: Optional[float] = None,
) -> tuple[bool, str, float]:
    """Return (escalate?, reason, score) based on simple rules."""
    normalized_cat = (category or "").strip().lower()
    is_hard = normalized_cat in HARD_CATEGORIES
    is_long = len(text) >= LENGTH_THRESHOLD
    conf = classifier_confidence if classifier_confidence is not None else 0.5
    low_conf = conf < LOW_CONFIDENCE

    triggers = []
    if is_long:
        triggers.append("long_ticket")
    if is_hard:
        triggers.append("hard_category")
    if low_conf:
        triggers.append("low_confidence")

    escalate = bool(triggers)
    if escalate:
        reason = f"escalate: {' + '.join(triggers)}"
        score = max(0.0, min(1.0, 1.0 - conf))
    else:
        reason = "stay_on_slm: heuristic_safe"
        score = conf
    return escalate, reason, score
