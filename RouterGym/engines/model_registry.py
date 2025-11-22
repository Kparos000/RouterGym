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
        self.model_name = model_id
        self.client = InferenceClient(model=model_id)

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, **_: Any) -> str:
        """Call chat_completion endpoint and normalize the response to string."""
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if isinstance(choice, dict):
                    return str(choice.get("message", {}).get("content", ""))
                # OpenAI-like object: choice.message.content
                msg = getattr(choice, "message", None)
                if isinstance(msg, dict):
                    return str(msg.get("content", ""))
                if msg is not None and hasattr(msg, "__getitem__"):
                    try:
                        return str(msg["content"])
                    except Exception:
                        pass
            return ""
        except Exception:
            return "[repair-response]"


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


def get_repair_model() -> RemoteLLMEngine:
    """Return the strongest available LLM engine for repair prompts."""
    # Prefer Qwen 72B then Llama 70B
    if "llm_qwen_72b" in LARGE_MODELS:
        return _load_llm(LARGE_MODELS["llm_qwen_72b"])
    return _load_llm(LARGE_MODELS["llm_llama_70b"])
