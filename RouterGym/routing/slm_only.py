"""SLM-only routing wrapper.

This mode exists as a deterministic benchmark floor: always use the SLM path
and still emit the shared routing metadata for downstream analysis.
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
from RouterGym.routing.policy import build_routing_decision
from RouterGym.utils.kb_utils import coerce_kb_hits


class SLMOnlyRouter(BaseRouter):
    """Always stay on the SLM path."""

    def route(
        self,
        ticket: Dict[str, Any],
        kb: Optional[Any] = None,
        models: Optional[Dict[str, Any]] = None,
        memory: Optional[Any] = None,
        force_llm: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del force_llm, kwargs
        text = ticket.get("text", "") if isinstance(ticket, dict) else str(ticket)
        models = models or {}
        slm = models.get("slm1") or models.get("slm2")

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

        raw_output = _call_model(slm, prompt) if slm is not None else ""
        json_contract = JSONContract()
        contract = SchemaContract()
        self_repair = SelfRepair()
        ok_json, parsed = (
            json_contract.validate(raw_output)
            if isinstance(raw_output, str)
            else (True, raw_output if isinstance(raw_output, dict) else None)
        )
        final_output = normalize_output(parsed if parsed else raw_output)
        schema_valid = bool(ok_json and contract.validate(final_output)[0])
        if not schema_valid and slm is not None:
            repaired = self_repair.repair(slm, prompt, raw_output, contract)
            final_output = normalize_output(repaired)
            schema_valid, _ = contract.validate(final_output)

        if not final_output.get("predicted_category"):
            final_output["predicted_category"] = infer_category_from_text(text)

        classifier_confidence = float(ticket.get("classifier_confidence", 1.0))
        routing_decision = build_routing_decision(
            router_mode="slm_only",
            text=text,
            base_model_name="slm",
            category=str(ticket.get("category") or ""),
            classifier_confidence=classifier_confidence,
            final_answer=str(final_output.get("final_answer", "")),
            schema_valid=schema_valid,
        )

        result = {
            "strategy": "slm_only",
            "target_model": "slm",
            "model_used": "slm",
            "steps": [{"stage": "generate", "model": "slm", "output": final_output}],
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


__all__ = ["SLMOnlyRouter"]
