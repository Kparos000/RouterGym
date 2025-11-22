"""SLM-dominant routing policy."""

from typing import Any, Dict, Optional

from RouterGym.routing.base import BaseRouter
from RouterGym.agents.generator import SchemaContract, SelfRepair, normalize_output, _call_model
from RouterGym.contracts.json_contract import JSONContract


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
        slm = models.get("slm_phi3") or models.get("slm_qwen3b")
        llm = models.get("llm1") or models.get("llm2")
        steps = []

        if memory:
            memory.add(text)
        memory_context = memory.get_context() if memory else ""
        kb_snippets = ""
        if kb is not None:
            try:
                hits = kb.retrieve(text, top_k=1) if hasattr(kb, "retrieve") else []
                if hits:
                    kb_snippets = hits[0].get("text") or hits[0].get("chunk", "")
            except Exception:
                kb_snippets = ""

        prompt_parts = [text]
        if memory_context:
            prompt_parts.append(f"[Memory]\n{memory_context}")
        if kb_snippets:
            prompt_parts.append(f"[KB]\n{kb_snippets}")
        prompt_parts.append("Return JSON with final_answer, reasoning.")
        prompt = "\n\n".join(prompt_parts)

        confidence = 0.8 if ticket.get("category") else 0.4
        contract = SchemaContract()

        initial_model = llm if force_llm else slm
        raw_output = _call_model(initial_model, prompt) if initial_model else ""
        jc = JSONContract()
        sr = SelfRepair()
        ok_json, parsed = jc.validate(raw_output) if isinstance(raw_output, str) else (True, raw_output if isinstance(raw_output, dict) else None)
        schema_ok = False
        final_output = normalize_output(raw_output)
        if ok_json and parsed:
            schema_ok, _ = contract.validate(normalize_output(parsed))
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
            fallback = False
            final_output = normalize_output(parsed if parsed else raw_output)

        steps.append({"stage": "generate", "output": final_output, "confidence": confidence})
        if fallback:
            steps.append({"stage": "fallback_llm", "output": final_output})

        return {
            "strategy": "slm_dominant",
            "target_model": "llm" if fallback else "slm",
            "model_used": "llm" if fallback else "slm",
            "steps": steps,
            "final_output": final_output,
            "json_valid": ok_json,
            "schema_valid": schema_ok,
        }
