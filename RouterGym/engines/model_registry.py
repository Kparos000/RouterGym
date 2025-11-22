"""Model registry using local HuggingFace pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

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
    "llm_qwen_14b": ModelEntry(name="llm_qwen_14b", hf_id="Qwen/Qwen2-14B-Instruct", kind="llm"),
    "llm_deepseek_16b": ModelEntry(name="llm_deepseek_16b", hf_id="deepseek-ai/deepseek-llm-16b-chat", kind="llm"),
}


def load_models(sanity: bool = False) -> Dict[str, Any]:
    """Load models as HF pipelines. In sanity mode, load a tiny model."""
    if sanity:
        tiny_id = "google/flan-t5-small"
        tok = AutoTokenizer.from_pretrained(tiny_id)
        mdl = AutoModelForCausalLM.from_pretrained(tiny_id)
        return {"sanity_model": pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)}

    models: Dict[str, Any] = {}
    for name, entry in MODEL_DEFS.items():
        tok = AutoTokenizer.from_pretrained(entry.hf_id)
        mdl = AutoModelForCausalLM.from_pretrained(entry.hf_id, torch_dtype=float32)
        models[name] = pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)
    return models


__all__ = ["load_models", "MODEL_DEFS"]
