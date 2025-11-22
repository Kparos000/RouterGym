"""Hybrid specialist routing policy."""

from typing import Any, Dict, Optional

from RouterGym.routing.base import BaseRouter
from RouterGym.agents.generator import SchemaContract


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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        text = ticket.get("text", "") if isinstance(ticket, dict) else str(ticket)
        category = ticket.get("category") if isinstance(ticket, dict) else None

        models = models or {}
        slm = models.get("slm_qwen_7b") or models.get("slm_llama_8b")
        llm = models.get("llm_deepseek_r1") or models.get("llm_qwen_72b")

        # Stage 1: classification
        classify_prompt = f"[Classify] {text}\nReturn JSON with classification."
        classify_output = _run_generation(slm, classify_prompt) if slm else ""

        # Stage 2: snippet retrieval
        snippet_text = ""
        if kb is not None:
            try:
                hits = kb.retrieve(text, top_k=1) if hasattr(kb, "retrieve") else []
                if hits:
                    snippet_text = hits[0].get("text") or hits[0].get("chunk", "")
            except Exception:
                snippet_text = ""

        # Stage 3: SLM draft
        draft_prompt = "\n\n".join(
            [
                text,
                f"[Snippet]\n{snippet_text}" if snippet_text else "",
                "Draft JSON with classification, answer, reasoning.",
            ]
        )
        draft_output = _run_generation(slm, draft_prompt) if slm else ""

        contract = SchemaContract()
        if not contract.validate(draft_output):
            draft_output = (
                '{"classification": "%s", "answer": "%s", "reasoning": "%s"}'
                % (category or "general", draft_output, "Hybrid draft repair")
            )

        # Stage 4: LLM rewrite
        rewrite_prompt = f"Rewrite for clarity keeping JSON structure:\n{draft_output}"
        final_output = _run_generation(llm, rewrite_prompt) if llm else draft_output
        if not contract.validate(final_output):
            final_output = draft_output

        steps = [
            {"stage": "classify_slm", "output": classify_output},
            {"stage": "retrieve_snippet", "snippet": snippet_text},
            {"stage": "draft_slm", "output": draft_output},
            {"stage": "rewrite_llm", "output": final_output},
        ]
        return {
            "strategy": "hybrid_specialist",
            "target_model": "llm",
            "model_used": "llm",
            "steps": steps,
            "final_output": final_output,
        }
