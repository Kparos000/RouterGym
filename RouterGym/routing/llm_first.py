"""LLM-first routing policy."""

from typing import Any, Dict, Optional

from RouterGym.routing.base import BaseRouter
from RouterGym.agents.generator import SchemaContract, SelfRepair, normalize_output, _call_model
from RouterGym.contracts.json_contract import JSONContract
from RouterGym.utils.kb_utils import coerce_kb_hits


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
        slm = models.get("slm_phi3") or models.get("slm_phi2")

        use_slm = (tokens < 40 or (category and str(category).lower() in {"access", "hardware", "hr"})) and not force_llm
        chosen_model = llm if force_llm else (slm if use_slm and slm is not None else llm or slm)

        if memory:
            memory.add(text)
        memory_context = memory.get_context() if memory else ""
        kb_snippets = ""
        if kb is not None:
            try:
                hits = coerce_kb_hits(kb.retrieve(text, top_k=1) if hasattr(kb, "retrieve") else [])
                if hits:
                    kb_snippets = hits[0]["text"]
            except Exception:
                kb_snippets = ""

        prompt_parts = [text]
        if memory_context:
            prompt_parts.append(f"[Memory]\n{memory_context}")
        if kb_snippets:
            prompt_parts.append(f"[KB]\n{kb_snippets}")
        prompt_parts.append("Return JSON with fields final_answer, reasoning.")
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
            "json_valid": ok_json,
            "schema_valid": ok_schema,
        }
