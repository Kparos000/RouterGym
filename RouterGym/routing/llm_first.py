"""LLM-first routing policy.

This router serves as the baseline: it always prefers the strongest LLM
(except for trivial short/obvious tickets) to set an upper bound on
quality and a lower bound on cost/latency efficiency. It is intentionally
simple and easy to explain for benchmarking.
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
        return "access"
    if "password" in lower or "login" in lower or "access" in lower:
        return "access"
    if "hr" in lower or "leave" in lower:
        return "hr support"
    if "printer" in lower or "laptop" in lower or "hardware" in lower:
        return "hardware"
    return default or "miscellaneous"


class LLMFirstRouter(BaseRouter):
    """Always prefer LLM, with optional downshift hooks."""

    def route(
        self,
        ticket: Dict[str, Any],
        kb: Optional[Any] = None,
        models: Optional[Dict[str, Any]] = None,
        memory: Optional[Any] = None,
        force_llm: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Route using LLM by default; downshift to SLM for very short or trivial tickets."""
        text = ticket.get("text", "") if isinstance(ticket, dict) else str(ticket)
        tokens = len(text.split())
        category = ticket.get("category") if isinstance(ticket, dict) else None

        # choose models
        models = models or {}
        llm = models.get("llm1") or models.get("llm2")
        slm = models.get("slm1") or models.get("slm2")

        use_slm = (tokens < 40 or (category and str(category).lower() in {"access", "hardware", "hr"})) and not force_llm
        chosen_model = llm if force_llm else (slm if use_slm and slm is not None else llm or slm)
        router_confidence_score = 0.0
        router_decision_reason = "llm_first_baseline"

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
            f"Use predicted_category from: {', '.join(CLASS_LABELS)}. Respond with JSON only."
        )
        prompt = "\n\n".join(prompt_parts)

        raw_output = _call_model(chosen_model, prompt) if chosen_model else ""
        contract = SchemaContract()
        jc = JSONContract()
        sr = SelfRepair()
        ok_json, parsed = jc.validate(raw_output) if isinstance(raw_output, str) else (True, raw_output if isinstance(raw_output, dict) else None)
        ok_schema = False
        final_output = normalize_output(raw_output)
        if ok_json and parsed:
            ok_schema, _ = contract.validate(normalize_output(parsed))
        if not (ok_json and ok_schema):
            repaired = sr.repair(chosen_model or slm, prompt, raw_output, contract) if chosen_model else final_output
            final_output = normalize_output(repaired)
        else:
            final_output = normalize_output(parsed)

        if not final_output.get("predicted_category"):
            final_output["predicted_category"] = infer_category_from_text(text)

        steps = [
            {"stage": "select_model", "model": "slm" if use_slm else "llm"},
            {"stage": "generate", "prompt": prompt, "output": final_output},
        ]
        return {
            "strategy": "llm_first",
            "target_model": "slm" if use_slm else "llm",
            "model_used": "slm" if use_slm else "llm",
            "steps": steps,
            "final_output": final_output,
            "json_valid": bool(ok_json),
            "schema_valid": bool(ok_schema),
            "predicted_category": final_output.get("predicted_category", ""),
            "kb_attached": bool(kb_snippets),
            "kb_snippets": kb_snippets,
            "prompt": prompt,
            "router_confidence_score": router_confidence_score,
            "router_decision_reason": router_decision_reason,
        }
