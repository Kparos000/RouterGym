"""Model registry with local SLMs and remote LLMs via HF Inference API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from huggingface_hub import InferenceClient  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
from torch import float32  # type: ignore


@dataclass
class ModelEntry:
    """Model entry describing HF identifiers."""

    name: str
    hf_id: str
    kind: str  # slm or llm


MODEL_DEFS: Dict[str, ModelEntry] = {
    "slm_qwen_7b": ModelEntry(name="slm_qwen_7b", hf_id="Qwen/Qwen2-7B-Instruct", kind="slm"),
    "slm_llama_8b": ModelEntry(name="slm_llama_8b", hf_id="meta-llama/Llama-3.1-8B-Instruct", kind="slm"),
    "llm_deepseek_r1": ModelEntry(name="llm_deepseek_r1", hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", kind="llm"),
    "llm_qwen_72b": ModelEntry(name="llm_qwen_72b", hf_id="Qwen/Qwen2-72B-Instruct", kind="llm"),
}


class RemoteLLMWrapper:
    """Wrap HF InferenceClient to expose a .generate(prompt) method."""

    def __init__(self, model_id: str) -> None:
        self.client = InferenceClient(model=model_id)

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, **kwargs: Any) -> str:
        resp = self.client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return resp


def _load_slm(entry: ModelEntry) -> Any:
    tok = AutoTokenizer.from_pretrained(entry.hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(entry.hf_id, torch_dtype=float32)
    return pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)


def _load_llm(entry: ModelEntry) -> Any:
    return RemoteLLMWrapper(entry.hf_id)


def load_models(sanity: bool = False) -> Dict[str, Any]:
    """Load models: SLMs locally, LLMs via HF Inference API. Sanity loads only one SLM."""
    if sanity:
        entry = MODEL_DEFS["slm_qwen_7b"]
        return {entry.name: _load_slm(entry)}

    models: Dict[str, Any] = {}
    for name, entry in MODEL_DEFS.items():
        if entry.kind == "slm":
            models[name] = _load_slm(entry)
        else:
            models[name] = _load_llm(entry)
    return models


__all__ = ["load_models", "MODEL_DEFS"]
