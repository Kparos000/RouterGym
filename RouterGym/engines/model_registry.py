"""Model registry with small local pipelines and remote LLM inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

from huggingface_hub import InferenceClient  # type: ignore
from transformers import pipeline  # type: ignore

# Disable downloads where possible and silence hub warnings
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@dataclass
class ModelEntry:
    """Model entry describing HF identifiers."""

    name: str
    hf_id: str
    kind: str  # slm or llm


SMALL_MODELS: Dict[str, ModelEntry] = {
    "slm_qwen_1_5b": ModelEntry("slm_qwen_1_5b", "Qwen/Qwen2-1.5B-Instruct", "slm"),
    "slm_tiny_llama": ModelEntry("slm_tiny_llama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "slm"),
}

LARGE_MODELS: Dict[str, ModelEntry] = {
    "llm_qwen_72b": ModelEntry("llm_qwen_72b", "Qwen/Qwen2-72B-Instruct", "llm"),
    "llm_llama_70b": ModelEntry("llm_llama_70b", "meta-llama/Meta-Llama-3-70B-Instruct", "llm"),
}


class RemoteLLMEngine:
    """Wrapper around HF InferenceClient to expose a .generate interface."""

    def __init__(self, model_id: str) -> None:
        self.client = InferenceClient(model=model_id)

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, **_: Any) -> str:
        return self.client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=temperature)


def _load_slm(entry: ModelEntry) -> Any:
    # Use transformers pipeline directly; keep on CPU.
    return pipeline("text-generation", model=entry.hf_id, device=-1)


def _load_llm(entry: ModelEntry) -> Any:
    # Remote client only; no local downloads.
    return RemoteLLMEngine(entry.hf_id)


def load_models(sanity: bool = False) -> Dict[str, Any]:
    """Load small models locally; large models via remote inference."""
    models: Dict[str, Any] = {}
    if sanity:
        entry = SMALL_MODELS["slm_qwen_1_5b"]
        models[entry.name] = _load_slm(entry)
        return models

    for entry in SMALL_MODELS.values():
        models[entry.name] = _load_slm(entry)
    for entry in LARGE_MODELS.values():
        models[entry.name] = _load_llm(entry)
    return models


__all__ = ["load_models", "RemoteLLMEngine", "SMALL_MODELS", "LARGE_MODELS"]
