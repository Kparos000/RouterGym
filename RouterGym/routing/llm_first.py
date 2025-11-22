"""LLM-first routing policy."""

from typing import Any, Dict, Optional

from RouterGym.routing.base import BaseRouter
from RouterGym.agents.generator import SchemaContract, SelfRepair
from RouterGym.contracts.json_contract import JSONContract


def _run_generation(model: Any, prompt: str) -> str:
    if hasattr(model, "generate"):
        out = model.generate(prompt, max_new_tokens=256, temperature=0.2)
        if isinstance(out, str):
            return out
        # HF pipeline returns list of dict
        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            return out[0]["generated_text"]
    if callable(model):
        return model(prompt)  # type: ignore
    return str(prompt)


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
        llm = models.get("llm_qwen_72b") or models.get("llm_llama_70b")
        slm = models.get("slm_qwen_1_5b") or models.get("slm_tiny_llama")

        use_slm = (tokens < 40 or (category and str(category).lower() in {"access", "hardware", "hr"})) and not force_llm
        chosen_model = slm if use_slm and slm is not None else llm or slm

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
        prompt_parts.append("Return JSON with fields classification, answer, reasoning.")
        prompt = "\n\n".join(prompt_parts)

        raw_output = _run_generation(chosen_model, prompt) if chosen_model else ""
        contract = SchemaContract()
        jc = JSONContract()
        sr = SelfRepair()
        ok_json, parsed = jc.validate(raw_output)
        ok_schema = False
        if ok_json:
            ok_schema, _ = contract.validate(parsed)
        if not (ok_json and ok_schema):
            repaired = sr.repair(chosen_model, prompt, raw_output, contract) if chosen_model else raw_output
            raw_output = repaired

        steps = [
            {"stage": "select_model", "model": "slm" if use_slm else "llm"},
            {"stage": "generate", "prompt": prompt, "output": raw_output},
        ]
        return {
            "strategy": "llm_first",
            "target_model": "slm" if use_slm else "llm",
            "model_used": "slm" if use_slm else "llm",
            "steps": steps,
            "final_output": raw_output,
        }
