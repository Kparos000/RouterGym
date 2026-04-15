"""Canonical SLM-dominant routing wrapper.

This router now delegates escalation thresholds to ``RouterGym.routing.policy``
so the benchmark has one explainable source of truth for SLM-first routing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from RouterGym.agents.generator import (
    CLASS_LABELS,
    SchemaContract,
    SelfRepair,
    _call_model,
    classification_instruction,
    infer_category_from_text,
    normalize_output,
)
from RouterGym.contracts.json_contract import JSONContract
from RouterGym.routing.base import BaseRouter
from RouterGym.routing.policy import build_routing_decision, should_escalate_heuristic
from RouterGym.utils.kb_utils import coerce_kb_hits


def _fallback_category(text: str, default: str = "") -> str:
    category = infer_category_from_text(text)
    if category and category != "unknown":
        return category
    return default or "miscellaneous"


def _extract_resolution_step_count(parsed: Any) -> int:
    if not isinstance(parsed, dict):
        return 0
    steps = parsed.get("resolution_steps", [])
    if isinstance(steps, list):
        return sum(1 for step in steps if isinstance(step, str) and step.strip())
    return 0


class SLMDominantRouter(BaseRouter):
    """Prefer the SLM first, then escalate using the canonical policy."""

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

        if memory:
            memory.add(text)
        memory_context = memory.get_context() if memory else ""

        kb_snippets = []
        if kb is not None:
            try:
                hits = coerce_kb_hits(kb.retrieve(text, top_k=3) if hasattr(kb, "retrieve") else [])
                kb_snippets = [hit["text"] for hit in hits if hit["text"]]
            except Exception:
                kb_snippets = []

        prompt_parts = [text]
        if memory_context:
            prompt_parts.append(f"[Memory]\n{memory_context}")
        if kb_snippets:
            prompt_parts.append("\n\n".join(f"[KB]\n{snippet}" for snippet in kb_snippets))
        prompt_parts.append(classification_instruction())
        prompt_parts.append(
            f"Use predicted_category from: {', '.join(CLASS_LABELS)}. Return JSON only."
        )
        prompt = "\n\n".join(prompt_parts)

        contract = SchemaContract()
        json_contract = JSONContract()
        self_repair = SelfRepair()

        classifier_confidence = float(
            ticket.get("classifier_confidence", 0.8 if ticket.get("category") else 0.4)
        )
        classifier_label = str(ticket.get("category") or "").strip()
        category = classifier_label or _fallback_category(text)

        initial_model = llm if force_llm and llm is not None else slm
        raw_output = _call_model(initial_model, prompt) if initial_model else ""
        ok_json, parsed = (
            json_contract.validate(raw_output)
            if isinstance(raw_output, str)
            else (True, raw_output if isinstance(raw_output, dict) else None)
        )
        parsed_dict = parsed if isinstance(parsed, dict) else {}
        final_output = normalize_output(parsed_dict if parsed_dict else raw_output)
        schema_valid = False
        if ok_json and parsed_dict:
            schema_valid, _ = contract.validate(final_output)

        if not schema_valid and initial_model is not None:
            repaired = self_repair.repair(initial_model, prompt, raw_output, contract)
            repaired_norm = normalize_output(repaired)
            repaired_valid, _ = contract.validate(repaired_norm)
            if repaired_valid:
                final_output = repaired_norm
                schema_valid = True
                if isinstance(repaired, dict):
                    parsed_dict = repaired

        routing_decision = build_routing_decision(
            router_mode="slm_dominant",
            text=text,
            base_model_name="slm",
            escalation_model_name="llm",
            category=category,
            classifier_confidence=classifier_confidence,
            retrieval_score=None,
            final_answer=str(final_output.get("final_answer", "") or ""),
            resolution_steps_count=_extract_resolution_step_count(parsed_dict),
            schema_valid=schema_valid,
            force_llm=force_llm,
        )

        model_used = "llm" if force_llm and llm is not None else "slm"
        if routing_decision.escalated and llm is not None and not (force_llm and llm is not None):
            raw_output = _call_model(llm, prompt)
            ok_json, parsed = (
                json_contract.validate(raw_output)
                if isinstance(raw_output, str)
                else (True, raw_output if isinstance(raw_output, dict) else None)
            )
            parsed_dict = parsed if isinstance(parsed, dict) else {}
            final_output = normalize_output(parsed_dict if parsed_dict else raw_output)
            schema_valid, _ = contract.validate(final_output)
            if not schema_valid:
                repaired = self_repair.repair(llm, prompt, raw_output, contract)
                final_output = normalize_output(repaired)
                schema_valid, _ = contract.validate(final_output)
            model_used = "llm"

        if not final_output.get("predicted_category"):
            final_output["predicted_category"] = infer_category_from_text(text)

        steps = [
            {"stage": "generate_initial", "model": "llm" if force_llm and llm is not None else "slm", "output": final_output},
        ]
        if routing_decision.escalated and model_used == "llm" and not (force_llm and llm is not None):
            steps.append({"stage": "escalate_rewrite", "model": "llm", "output": final_output})

        result = {
            "strategy": "slm_dominant",
            "target_model": routing_decision.final_model,
            "model_used": model_used,
            "steps": steps,
            "final_output": final_output,
            "json_valid": bool(ok_json),
            "schema_valid": bool(schema_valid),
            "predicted_category": final_output.get("predicted_category", ""),
            "kb_attached": bool(kb_snippets),
            "kb_snippets": kb_snippets,
            "prompt": prompt,
            "router_confidence_score": routing_decision.router_confidence_score,
            "router_decision_reason": routing_decision.router_decision_reason,
        }
        result.update(routing_decision.as_dict())
        return result


__all__ = ["SLMDominantRouter", "should_escalate_heuristic"]
