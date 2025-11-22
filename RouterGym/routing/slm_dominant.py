"""SLM-dominant routing policy."""

from typing import Any, Dict, Optional

from RouterGym.routing.base import BaseRouter
from RouterGym.agents.generator import SchemaContract, SelfRepair
from RouterGym.contracts.json_contract import JSONContract


def _run_generation(model: Any, prompt: str) -> str:
    if hasattr(model, "generate"):
        out = model.generate(prompt, max_new_tokens=256, temperature=0.2)
        if isinstance(out, str):
            return out
        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            return out[0]["generated_text"]
    if callable(model):
        return model(prompt)  # type: ignore
    return str(prompt)


class SLMDominantRouter(BaseRouter):
    """Prefer SLM routes and escalate on contract or confidence failures."""

    def route(
        self,
        ticket: Dict[str, Any],
        kb: Optional[Any] = None,
        models: Optional[Dict[str, Any]] = None,
        memory: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        text = ticket.get("text", "") if isinstance(ticket, dict) else str(ticket)
        models = models or {}
        slm = models.get("slm_qwen_7b") or models.get("slm_llama_8b")
        llm = models.get("llm_deepseek_r1") or models.get("llm_qwen_72b")
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
        prompt_parts.append("Return JSON with classification, answer, reasoning.")
        prompt = "\n\n".join(prompt_parts)

        confidence = 0.8 if ticket.get("category") else 0.4
        contract = SchemaContract()

        raw_output = _run_generation(slm, prompt) if slm else ""
        jc = JSONContract()
        sr = SelfRepair()
        ok_json, parsed = jc.validate(raw_output)
        retries = 0
        while retries < 2 and not (ok_json and contract.validate(parsed)[0]):
            raw_output = _run_generation(slm, prompt) if slm else raw_output
            ok_json, parsed = jc.validate(raw_output)
            retries += 1

        fallback = False
        missing_kb = kb is not None and not kb_snippets
        if (confidence < 0.55 or missing_kb or not (ok_json and contract.validate(parsed)[0])) and llm is not None:
            fallback = True
            raw_output = _run_generation(llm, prompt) if llm else raw_output
            ok_json, parsed = jc.validate(raw_output)
            if not (ok_json and contract.validate(parsed)[0]) and llm is not None:
                raw_output = sr.repair(llm, prompt, raw_output, contract)

        steps.append({"stage": "generate_slm", "output": raw_output, "confidence": confidence})
        if fallback:
            steps.append({"stage": "fallback_llm", "output": raw_output})

        return {
            "strategy": "slm_dominant",
            "target_model": "llm" if fallback else "slm",
            "model_used": "llm" if fallback else "slm",
            "steps": steps,
            "final_output": raw_output,
        }
