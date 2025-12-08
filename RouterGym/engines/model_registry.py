"""Model registry supporting HF Inference (default) and optional local vLLM."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from os import PathLike
from typing import IO, Any, Callable, Dict, Optional

from huggingface_hub import InferenceClient  # type: ignore

try:
    from RouterGym.engines.vllm_local import LocalVLLMEngine  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LocalVLLMEngine = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def _load_dotenv(
        dotenv_path: Optional[str | PathLike[str]] = None,
        stream: Optional[IO[str]] = None,
        verbose: bool = False,
        override: bool = False,
        interpolate: bool = True,
        encoding: Optional[str] = None,
    ) -> bool:
        return False
DotenvCallable = Callable[..., bool]
_dotenv_loader: DotenvCallable = _load_dotenv

# Load environment variables from .env if present
if callable(_dotenv_loader):
    _dotenv_loader()


@dataclass
class ModelEntry:
    """Model entry describing HF identifiers."""

    name: str
    hf_id: str
    kind: str  # slm or llm


SLM_MODELS: Dict[str, ModelEntry] = {
    "slm1": ModelEntry("slm1", "mistralai/Mistral-7B-Instruct-v0.3", "slm"),
    "slm2": ModelEntry("slm2", "meta-llama/Meta-Llama-3-8B-Instruct", "slm"),
}

LLM_MODELS: Dict[str, ModelEntry] = {
    "llm1": ModelEntry("llm1", "Qwen/Qwen2-72B-Instruct", "llm"),
    "llm2": ModelEntry("llm2", "meta-llama/Meta-Llama-3-70B-Instruct", "llm"),
}


class RemoteInferenceEngine:
    """Remote HF InferenceClient wrapper with chat_completion + retries."""

    def __init__(
        self,
        model_id: str,
        kind: str = "llm",
        token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 1,
    ) -> None:
        self.model_name = model_id
        self.kind = kind
        self.client = InferenceClient(model=model_id, token=token, timeout=timeout)
        self.timeout = timeout
        self.max_retries = max_retries

    def _extract_content(self, response: Any) -> Optional[str]:
        if response is None or not hasattr(response, "choices"):
            return None
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message") or {}
            return str(msg.get("content", ""))
        msg = getattr(first, "message", None)
        if isinstance(msg, dict):
            return str(msg.get("content", ""))
        if msg is not None and hasattr(msg, "__getitem__"):
            try:
                return str(msg["content"])
            except Exception:
                return None
        return None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        """Call chat_completion endpoint with retries and normalize the response to string."""
        fallback = json.dumps(
            {
                "final_answer": "LLM unavailable",
                "reasoning": "timeout or error",
                "predicted_category": "unknown",
            }
        )
        for _attempt in range(max(1, self.max_retries + 1)):
            try:
                response = self.client.chat_completion(  # type: ignore[call-overload]
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = self._extract_content(response)
                if content is not None:
                    return str(content)
            except Exception:
                continue
        return fallback

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)


def _get_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def get_model_backend() -> str:
    """Return configured model backend (hf_inference or vllm_local)."""
    backend = os.getenv("ROUTERGYM_MODEL_BACKEND", "").strip().lower()
    if backend in {"vllm_local"}:
        return "vllm_local"
    return "hf_inference"


def _filter_entries(entries: Dict[str, ModelEntry], subset: Optional[list[str]]) -> list[ModelEntry]:
    if subset:
        allowed = set(subset)
        return [entry for entry in entries.values() if entry.name in allowed]
    return list(entries.values())


def _build_engine(entry: ModelEntry, token: Optional[str]) -> RemoteInferenceEngine:
    return RemoteInferenceEngine(entry.hf_id, kind=entry.kind, token=token)


def load_models(sanity: bool = False, slm_subset: Optional[list[str]] = None, force_llm: bool = False) -> Dict[str, Any]:
    """Load all models using the configured backend (HF Inference or vLLM local)."""
    backend = get_model_backend()
    token = _get_token()
    models: Dict[str, Any] = {}
    subset = slm_subset or None

    slm_entries = _filter_entries(SLM_MODELS, subset)
    llm_entries = _filter_entries(LLM_MODELS, subset)

    if sanity:
        if not slm_entries:
            slm_entries = _filter_entries(SLM_MODELS, None)
        if not llm_entries:
            llm_entries = _filter_entries(LLM_MODELS, None)

        if slm_entries:
            slm_entry = slm_entries[0]
            models[slm_entry.name] = _build_engine(slm_entry, token)
        if llm_entries:
            llm_entry = llm_entries[0]
            models[llm_entry.name] = _build_engine(llm_entry, token)
        return models

    if force_llm and not llm_entries:
        llm_entries = _filter_entries(LLM_MODELS, None)

    if backend == "vllm_local":
        if LocalVLLMEngine is None:
            raise ImportError("vllm_local backend selected but vllm is not installed.")
        if not force_llm:
            for entry in slm_entries:
                models[entry.name] = LocalVLLMEngine(entry.hf_id)
        for entry in llm_entries:
            models[entry.name] = LocalVLLMEngine(entry.hf_id)
    else:  # hf_inference default
        if not force_llm:
            for entry in slm_entries:
                models[entry.name] = _build_engine(entry, token)
        for entry in llm_entries:
            models[entry.name] = _build_engine(entry, token)

    return models


def get_repair_model() -> RemoteInferenceEngine:
    """Return the strongest available LLM engine for repair prompts."""
    token = _get_token()
    backend = get_model_backend()
    if backend == "vllm_local":
        if LocalVLLMEngine is None:
            raise ImportError("vllm_local backend selected but vllm is not installed.")
        target = LLM_MODELS.get("llm1") or LLM_MODELS.get("llm2")
        return LocalVLLMEngine(target.hf_id if target else "unknown_llm")  # type: ignore[return-value]
    if "llm1" in LLM_MODELS:
        return _build_engine(LLM_MODELS["llm1"], token)
    return _build_engine(LLM_MODELS["llm2"], token)


__all__ = [
    "load_models",
    "RemoteInferenceEngine",
    "SLM_MODELS",
    "LLM_MODELS",
    "get_repair_model",
    "get_model_backend",
]
