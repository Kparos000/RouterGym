"""Model registry supporting HF Inference (default) and optional local vLLM."""

from __future__ import annotations

import json
import importlib
import importlib.util
import os
import traceback
from dataclasses import dataclass
from os import PathLike
from typing import IO, Any, Callable, Dict, Optional

from huggingface_hub import InferenceClient  # type: ignore

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
_ENV_LOADED = False


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
    "llm1": ModelEntry("llm1", "openai/gpt-oss-20b", "llm"),
    "llm2": ModelEntry("llm2", "Qwen/Qwen2.5-14B-Instruct", "llm"),
}
# These benchmark models are routed through HF providers that expose
# conversational/chat interfaces only. Keep them off text_generation so
# provider task-mismatch errors stay out of the main pipeline path.
HF_CHAT_ONLY_MODEL_KEYS = frozenset({"slm1", "slm2", "llm1", "llm2"})


class RemoteInferenceEngine:
    """Remote HF InferenceClient wrapper with chat_completion + retries."""

    def __init__(
        self,
        model_id: str,
        model_key: Optional[str] = None,
        kind: str = "llm",
        token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 1,
    ) -> None:
        self.model_key = model_key or model_id
        self.model_name = model_id
        self.kind = kind
        self.backend_used = "hf_inference"
        self.last_usage: Optional[Dict[str, int]] = None
        self.last_endpoint_path = ""
        self.last_error: Optional[Dict[str, str]] = None
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
            content = str(msg.get("content", ""))
            return content or None
        msg = getattr(first, "message", None)
        if isinstance(msg, dict):
            content = str(msg.get("content", ""))
            return content or None
        if msg is not None and hasattr(msg, "__getitem__"):
            try:
                content = str(msg["content"])
                return content or None
            except Exception:
                return None
        return None

    def _extract_text_generation_content(self, response: Any) -> Optional[str]:
        if response is None:
            return None
        if isinstance(response, str):
            return response
        generated_text = getattr(response, "generated_text", None)
        if isinstance(generated_text, str):
            return generated_text
        if isinstance(response, dict):
            value = response.get("generated_text", response.get("text"))
            if isinstance(value, str):
                return value
        return None

    def _extract_usage(self, response: Any) -> Optional[Dict[str, int]]:
        if response is None:
            return None
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return None

        def _coerce(value: Any) -> Optional[int]:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return max(int(value), 0)
            return None

        if isinstance(usage, dict):
            prompt_tokens = _coerce(usage.get("prompt_tokens", usage.get("input_tokens")))
            completion_tokens = _coerce(
                usage.get("completion_tokens", usage.get("output_tokens"))
            )
            total_tokens = _coerce(usage.get("total_tokens"))
        else:
            prompt_tokens = _coerce(getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None)))
            completion_tokens = _coerce(
                getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None))
            )
            total_tokens = _coerce(getattr(usage, "total_tokens", None))

        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return None
        safe_prompt = prompt_tokens or 0
        safe_completion = completion_tokens or 0
        safe_total = total_tokens if total_tokens is not None else safe_prompt + safe_completion
        if safe_total == 0:
            safe_total = safe_prompt + safe_completion
        return {
            "input_tokens": safe_prompt,
            "output_tokens": safe_completion,
            "total_tokens": safe_total,
        }

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
        self.last_usage = None
        self.last_endpoint_path = ""
        self.last_error = None
        for _attempt in range(max(1, self.max_retries + 1)):
            try:
                response = self.client.chat_completion(  # type: ignore[call-overload]
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                self.last_usage = self._extract_usage(response)
                self.last_endpoint_path = "chat_completion"
                self.last_error = None
                content = self._extract_content(response)
                if content is not None:
                    return str(content)
            except Exception as exc:
                self.last_usage = None
                self.last_endpoint_path = ""
                self.last_error = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "stack_trace": traceback.format_exc(),
                    "phase": "chat_completion",
                }
            try:
                response = self.client.chat_completion(  # type: ignore[call-overload]
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max(max_new_tokens, 1024),
                    temperature=temperature,
                )
                self.last_usage = self._extract_usage(response)
                self.last_endpoint_path = "chat_completion_plain"
                self.last_error = None
                content = self._extract_content(response)
                if content is not None:
                    return str(content)
            except Exception as exc:
                self.last_usage = None
                self.last_endpoint_path = ""
                self.last_error = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "stack_trace": traceback.format_exc(),
                    "phase": "chat_completion_plain",
                }
            if self.model_key in HF_CHAT_ONLY_MODEL_KEYS:
                continue
            try:
                response = self.client.text_generation(
                    prompt,
                    model=self.model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    return_full_text=False,
                )
                self.last_usage = self._extract_usage(response)
                self.last_endpoint_path = "text_generation"
                self.last_error = None
                content = self._extract_text_generation_content(response)
                if content is not None:
                    return str(content)
            except Exception as exc:
                self.last_usage = None
                self.last_endpoint_path = ""
                self.last_error = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "stack_trace": traceback.format_exc(),
                    "phase": "text_generation",
                }
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
    _ensure_env_loaded()
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _get_openai_compatible_base_url() -> str:
    _ensure_env_loaded()
    return (
        os.getenv("ROUTERGYM_OPENAI_BASE_URL")
        or os.getenv("ROUTERGYM_VLLM_BASE_URL")
        or "http://localhost:8000/v1"
    )


def _get_openai_compatible_api_key() -> str:
    _ensure_env_loaded()
    return (
        os.getenv("ROUTERGYM_OPENAI_API_KEY")
        or os.getenv("ROUTERGYM_VLLM_API_KEY")
        or "EMPTY"
    )


def _ensure_env_loaded() -> None:
    """Load environment variables lazily to keep module import side-effect free."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if callable(_dotenv_loader):
        _dotenv_loader()
    _ENV_LOADED = True


def get_model_backend() -> str:
    """Return configured model backend."""
    backend = os.getenv("ROUTERGYM_MODEL_BACKEND", "").strip().lower()
    if backend in {"openai_compatible", "vllm_openai"}:
        return "openai_compatible"
    if backend in {"vllm_local"}:
        return "vllm_local"
    return "hf_inference"


def _get_local_vllm_engine_class():
    """Import LocalVLLMEngine only when the backend is actually requested."""
    try:
        module = importlib.import_module("RouterGym.engines.vllm_local")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("vllm_local backend selected but vllm is not installed.") from exc
    engine_cls = getattr(module, "LocalVLLMEngine", None)
    if engine_cls is None:
        raise ImportError("vllm_local backend selected but vllm is not installed.")
    return engine_cls


def has_local_vllm_backend() -> bool:
    """Return whether the optional vLLM package appears importable."""
    return importlib.util.find_spec("vllm") is not None


def _get_openai_compatible_engine_class():
    from RouterGym.engines.openai_compatible import OpenAICompatibleEngine

    return OpenAICompatibleEngine


def _filter_entries(entries: Dict[str, ModelEntry], subset: Optional[list[str]]) -> list[ModelEntry]:
    if subset:
        allowed = set(subset)
        return [entry for entry in entries.values() if entry.name in allowed]
    return list(entries.values())


def _build_engine(entry: ModelEntry, token: Optional[str]) -> RemoteInferenceEngine:
    return RemoteInferenceEngine(entry.hf_id, model_key=entry.name, kind=entry.kind, token=token)


def _build_openai_compatible_engine(entry: ModelEntry) -> Any:
    engine_cls = _get_openai_compatible_engine_class()
    return engine_cls(
        entry.hf_id,
        model_key=entry.name,
        kind=entry.kind,
        base_url=_get_openai_compatible_base_url(),
        api_key=_get_openai_compatible_api_key(),
    )


def _tag_local_engine(engine: Any, entry: ModelEntry) -> Any:
    setattr(engine, "model_key", entry.name)
    setattr(engine, "kind", entry.kind)
    setattr(engine, "backend_used", "vllm_local")
    setattr(engine, "last_usage", None)
    return engine


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
        local_vllm_engine = _get_local_vllm_engine_class()
        if not force_llm:
            for entry in slm_entries:
                models[entry.name] = _tag_local_engine(local_vllm_engine(entry.hf_id), entry)
        for entry in llm_entries:
            models[entry.name] = _tag_local_engine(local_vllm_engine(entry.hf_id), entry)
    elif backend == "openai_compatible":
        if not force_llm:
            for entry in slm_entries:
                models[entry.name] = _build_engine(entry, token)
        for entry in llm_entries:
            models[entry.name] = _build_openai_compatible_engine(entry)
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
        local_vllm_engine = _get_local_vllm_engine_class()
        target = LLM_MODELS.get("llm1") or LLM_MODELS.get("llm2")
        return local_vllm_engine(target.hf_id if target else "unknown_llm")  # type: ignore[return-value]
    if backend == "openai_compatible":
        target = LLM_MODELS.get("llm1") or LLM_MODELS.get("llm2")
        if target is None:
            raise RuntimeError("No LLM entries are configured for the openai_compatible backend.")
        return _build_openai_compatible_engine(target)  # type: ignore[return-value]
    if "llm1" in LLM_MODELS:
        return _build_engine(LLM_MODELS["llm1"], token)
    return _build_engine(LLM_MODELS["llm2"], token)


__all__ = [
    "load_models",
    "RemoteInferenceEngine",
    "SLM_MODELS",
    "LLM_MODELS",
    "_get_openai_compatible_api_key",
    "_get_openai_compatible_base_url",
    "get_repair_model",
    "get_model_backend",
    "has_local_vllm_backend",
]
