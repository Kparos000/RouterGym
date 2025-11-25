"""Hybrid specialist routing policy."""

import json
from typing import Any, Dict, Optional

from RouterGym.routing.base import BaseRouter
from RouterGym.agents.generator import SchemaContract, SelfRepair, normalize_output, _call_model
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
    return default or "general"


class HybridSpecialistRouter(BaseRouter):
    """Route to specialized SLMs by domain; fallback to LLM as needed."""

    def __init__(self) -> None:
        self.category_to_model = {
            "access": "slm",
            "hardware": "slm",
            "hr_support": "slm",
        }

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

        # Stage 1: classification
        classify_prompt = f"[Classify] {text}\nReturn JSON with final_answer and reasoning."
        classify_output = _call_model(llm if force_llm else slm, classify_prompt) if (llm if force_llm else slm) else ""

        # Stage 2: snippet retrieval
        snippet_text = ""
        if kb is not None:
            try:
                hits = coerce_kb_hits(kb.retrieve(text, top_k=3) if hasattr(kb, "retrieve") else [])
                if hits:
                    snippet_text = "\n".join([h["text"] for h in hits if h["text"]])
            except Exception:
                snippet_text = ""

        # Stage 3: SLM draft
        draft_prompt = "\n\n".join(
            [
                text,
                f"[Snippet]\n{snippet_text}" if snippet_text else "",
                "Draft JSON with final_answer and reasoning.",
            ]
        )
        draft_raw = _call_model(llm if force_llm else slm, draft_prompt) if (llm if force_llm else slm) else ""

        contract = SchemaContract()
        jc = JSONContract()
        sr = SelfRepair()
        ok_json, parsed = jc.validate(draft_raw) if isinstance(draft_raw, str) else (True, draft_raw if isinstance(draft_raw, dict) else None)
        draft_norm = normalize_output(parsed if parsed else draft_raw)
        if not (ok_json and contract.validate(draft_norm)[0]):
            repaired = sr.repair(llm if force_llm else slm, draft_prompt, draft_raw, contract) if (llm or slm) else draft_norm
            draft_norm = normalize_output(repaired)
        if not draft_norm.get("predicted_category"):
            draft_norm["predicted_category"] = _infer_category(text, str(ticket.get("category", "")))

        # Stage 4: LLM rewrite
        rewrite_prompt = f"Rewrite for clarity keeping JSON structure:\n{draft_norm}"
        final_output_raw = _call_model(llm, rewrite_prompt) if llm else json.dumps(draft_norm)
        ok_json, parsed = jc.validate(final_output_raw) if isinstance(final_output_raw, str) else (True, final_output_raw if isinstance(final_output_raw, dict) else None)
        final_output = normalize_output(parsed if parsed else final_output_raw)
        if not (ok_json and contract.validate(final_output)[0]):
            final_output = draft_norm
        if not final_output.get("predicted_category"):
            final_output["predicted_category"] = draft_norm.get("predicted_category", _infer_category(text, str(ticket.get("category", ""))))

        steps = [
            {"stage": "classify_slm", "output": normalize_output(classify_output)},
            {"stage": "retrieve_snippet", "snippet": snippet_text},
            {"stage": "draft_slm", "output": draft_norm},
            {"stage": "rewrite_llm", "output": final_output},
        ]
        return {
            "strategy": "hybrid_specialist",
            "target_model": "llm",
            "model_used": "llm",
            "steps": steps,
            "final_output": final_output,
            "json_valid": bool(ok_json),
            "schema_valid": contract.validate(final_output)[0],
            "predicted_category": final_output.get("predicted_category", ""),
            "kb_attached": bool(snippet_text),
        }
