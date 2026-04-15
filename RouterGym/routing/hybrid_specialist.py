"""Canonical hybrid-specialist routing wrapper.

The risk calculation is delegated to ``RouterGym.routing.policy`` so the
benchmark exposes one centralized specialist baseline.
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
from RouterGym.routing.policy import RISK_HIGH, RISK_LOW, build_routing_decision, risk_score
from RouterGym.utils.kb_utils import coerce_kb_hits


def _extract_resolution_step_count(parsed: Any) -> int:
    if not isinstance(parsed, dict):
        return 0
    steps = parsed.get("resolution_steps", [])
    if isinstance(steps, list):
        return sum(1 for step in steps if isinstance(step, str) and step.strip())
    return 0


class HybridSpecialistRouter(BaseRouter):
    """Low-risk tickets stay on SLM; higher-risk tickets use the LLM rewrite path."""

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
        if memory:
            memory.add(text)

        models = models or {}
        slm = models.get("slm1") or models.get("slm2")
        llm = models.get("llm1") or models.get("llm2")

        classify_prompt = "\n".join(
            [
                f"[Classify] {text}",
                classification_instruction(),
                f"Use predicted_category from: {', '.join(CLASS_LABELS)}.",
            ]
        )
        classify_output = _call_model(llm if force_llm else slm, classify_prompt) if (llm if force_llm else slm) else ""

        snippet_text = ""
        if kb is not None:
            try:
                hits = coerce_kb_hits(kb.retrieve(text, top_k=3) if hasattr(kb, "retrieve") else [])
                if hits:
                    snippet_text = "\n".join(hit["text"] for hit in hits if hit["text"])
            except Exception:
                snippet_text = ""

        draft_prompt = "\n\n".join(
            [
                text,
                f"[Snippet]\n{snippet_text}" if snippet_text else "",
                classification_instruction(),
                f"Use predicted_category from: {', '.join(CLASS_LABELS)}.",
                "Draft JSON with final_answer, reasoning, predicted_category (classify the ticket).",
            ]
        )
        draft_raw = _call_model(llm if force_llm else slm, draft_prompt) if (llm if force_llm else slm) else ""

        contract = SchemaContract()
        json_contract = JSONContract()
        self_repair = SelfRepair()
        ok_json, parsed = (
            json_contract.validate(draft_raw)
            if isinstance(draft_raw, str)
            else (True, draft_raw if isinstance(draft_raw, dict) else None)
        )
        parsed_dict = parsed if isinstance(parsed, dict) else {}
        draft_norm = normalize_output(parsed_dict if parsed_dict else draft_raw)
        draft_valid = ok_json and contract.validate(draft_norm)[0]
        if not draft_valid:
            repaired = self_repair.repair(llm if force_llm else slm, draft_prompt, draft_raw, contract) if (llm or slm) else draft_norm
            draft_norm = normalize_output(repaired)
            draft_valid, _ = contract.validate(draft_norm)
            if isinstance(repaired, dict):
                parsed_dict = repaired
        if not draft_norm.get("predicted_category"):
            draft_norm["predicted_category"] = infer_category_from_text(text)

        classifier_confidence = float(ticket.get("classifier_confidence", 0.5))
        category = (
            str(ticket.get("category") or "").strip()
            or draft_norm.get("predicted_category", "")
            or infer_category_from_text(text)
        )
        routing_decision = build_routing_decision(
            router_mode="hybrid_specialist",
            text=text,
            base_model_name="slm",
            escalation_model_name="llm",
            category=category,
            classifier_confidence=classifier_confidence,
            final_answer=str(draft_norm.get("final_answer", "") or ""),
            resolution_steps_count=_extract_resolution_step_count(parsed_dict),
            schema_valid=draft_valid,
            force_llm=force_llm,
        )

        model_used = "llm" if force_llm and llm is not None else "slm"
        final_output = draft_norm
        if routing_decision.final_model == "llm" and llm is not None and not (force_llm and llm is not None):
            rewrite_prompt = f"Rewrite for clarity keeping JSON structure:\n{draft_norm}"
            final_output_raw = _call_model(llm, rewrite_prompt)
            ok_json, parsed = (
                json_contract.validate(final_output_raw)
                if isinstance(final_output_raw, str)
                else (True, final_output_raw if isinstance(final_output_raw, dict) else None)
            )
            final_output = normalize_output(parsed if parsed else final_output_raw)
            if not (ok_json and contract.validate(final_output)[0]):
                final_output = draft_norm
            model_used = "llm"
        elif routing_decision.final_model == "llm":
            final_output = draft_norm

        if not final_output.get("predicted_category"):
            final_output["predicted_category"] = draft_norm.get(
                "predicted_category", infer_category_from_text(text)
            )

        steps = [
            {"stage": "classify", "output": normalize_output(classify_output)},
            {"stage": "retrieve_snippet", "snippet": snippet_text},
            {"stage": "draft", "model": "llm" if force_llm and llm is not None else "slm", "output": draft_norm},
        ]
        if model_used == "llm":
            steps.append({"stage": "rewrite_final", "model": "llm", "output": final_output})

        result = {
            "strategy": "hybrid_specialist",
            "target_model": routing_decision.final_model,
            "model_used": model_used,
            "steps": steps,
            "final_output": final_output,
            "json_valid": bool(ok_json),
            "schema_valid": contract.validate(final_output)[0],
            "predicted_category": final_output.get("predicted_category", ""),
            "kb_attached": bool(snippet_text),
            "kb_snippets": [snippet_text] if snippet_text else [],
            "prompt": draft_prompt,
            "router_confidence_score": routing_decision.router_confidence_score,
            "router_decision_reason": routing_decision.router_decision_reason,
        }
        result.update(routing_decision.as_dict())
        return result


__all__ = ["HybridSpecialistRouter", "RISK_HIGH", "RISK_LOW", "risk_score"]
