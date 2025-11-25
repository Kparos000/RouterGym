"""Model registry with local SLM pipelines and remote LLM inference."""

from __future__ import annotations

import os
from os import PathLike
from dataclasses import dataclass
from typing import IO, Any, Callable, Dict, Optional

import torch  # type: ignore
from huggingface_hub import InferenceClient  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

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


SMALL_MODELS: Dict[str, ModelEntry] = {
    "slm_phi3": ModelEntry("slm_phi3", "microsoft/Phi-3-mini-4k-instruct", "slm"),
    "slm_phi2": ModelEntry("slm_phi2", "microsoft/phi-2", "slm"),
}

LARGE_MODELS: Dict[str, ModelEntry] = {
    "llm1": ModelEntry("llm1", "Qwen/Qwen2-72B-Instruct", "llm"),
    "llm2": ModelEntry("llm2", "meta-llama/Meta-Llama-3-70B-Instruct", "llm"),
}


_LOCAL_CACHE: Dict[str, Any] = {}


class LocalPipelineEngine:
    """Lazy local HF pipeline wrapper with caching."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.pipeline: Optional[Any] = _LOCAL_CACHE.get(model_id)

    def _ensure_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype)
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
        )
        _LOCAL_CACHE[self.model_id] = self.pipeline

    def __call__(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, do_sample: bool = False, **_: Any) -> str:
        self._ensure_pipeline()
        if self.pipeline is None:  # Safety for type-checkers
            raise RuntimeError("Pipeline failed to initialize")
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            return_full_text=False,
        )
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            return str(outputs[0].get("generated_text", ""))
        return str(outputs)

    def unload(self) -> None:
        """Release cached pipeline to free memory."""
        if self.pipeline is not None:
            self.pipeline = None
        _LOCAL_CACHE.pop(self.model_id, None)


class RemoteLLMEngine:
    """Wrapper around HF InferenceClient to expose a .generate interface with timeout/retries."""

    def __init__(self, model_id: str, token: Optional[str] = None, timeout: int = 30, max_retries: int = 2) -> None:
        self.model_name = model_id
        self.client = InferenceClient(model=model_id, token=token, timeout=timeout)
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, **_: Any) -> str:
        """Call chat_completion endpoint with retries and normalize the response to string."""
        last_error: Optional[Exception] = None
        response_schema = {
            "name": "RouterGymResponse",
            "schema": {
                "type": "object",
                "properties": {
                    "final_answer": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["final_answer", "reasoning"],
            },
            "strict": True,
        }
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat_completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    response_format={"type": "json_schema", "json_schema": response_schema},
                )
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    if isinstance(choice, dict):
                        return str(choice.get("message", {}).get("content", ""))
                    msg = getattr(choice, "message", None)
                    if isinstance(msg, dict):
                        return str(msg.get("content", ""))
                    if msg is not None and hasattr(msg, "__getitem__"):
                        try:
                            return str(msg["content"])
                        except Exception:
                            pass
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                continue
        if last_error:
            # Fallback JSON to avoid downstream crashes
            return '{"final_answer":"LLM unavailable","reasoning":"timeout or error"}'
        return ""

    def __call__(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, **kwargs: Any) -> str:
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)


class RemoteGoogleSLMEngine:
    """Minimal remote SLM engine using a Google/Gemini-style endpoint."""

    def __init__(self, api_key: str, model: str = "gemini-pro") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, **kwargs: Any) -> str:
        try:
            import requests  # type: ignore
        except Exception:
            return "[google-slm-unavailable]"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_new_tokens, "temperature": temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates") or []
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts") or []
                if parts and "text" in parts[0]:
                    return str(parts[0]["text"])
        except Exception:
            return "[google-slm-error]"
        return "[google-slm-empty]"

    def __call__(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, **kwargs: Any) -> str:
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)


def _load_slm(entry: ModelEntry) -> Any:
    return LocalPipelineEngine(entry.hf_id)


def _load_llm(entry: ModelEntry) -> Any:
    # Remote client only; no local downloads.
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    return RemoteLLMEngine(entry.hf_id, token=token)


def load_models(sanity: bool = False, slm_subset: Optional[list[str]] = None, force_llm: bool = False) -> Dict[str, Any]:
    """Load small models locally; large models via remote inference."""
    models: Dict[str, Any] = {}
    use_google_slm = (os.getenv("USE_GOOGLE_SLM") or "false").lower() == "true"
    google_key = os.getenv("GOOGLE_API_KEY")
    subset = set(slm_subset or [])
    slm_names = list(SMALL_MODELS.keys())
    if subset:
        slm_names = [name for name in slm_names if name in subset]
    sanity_target = slm_names[0] if slm_names else "slm_phi3"
    if sanity:
        entry = SMALL_MODELS.get(sanity_target, SMALL_MODELS["slm_phi3"])
        models[entry.name] = _load_slm(entry)
        return models

    if not force_llm:
        for name in slm_names:
            entry = SMALL_MODELS[name]
            if entry.name == "slm_phi2" and use_google_slm and google_key:
                models[entry.name] = RemoteGoogleSLMEngine(api_key=google_key)
            else:
                models[entry.name] = _load_slm(entry)
    for entry in LARGE_MODELS.values():
        models[entry.name] = _load_llm(entry)
    return models


__all__ = [
    "load_models",
    "RemoteLLMEngine",
    "LocalPipelineEngine",
    "RemoteGoogleSLMEngine",
    "SMALL_MODELS",
    "LARGE_MODELS",
]


def get_repair_model() -> RemoteLLMEngine:
    """Return the strongest available LLM engine for repair prompts."""
    if "llm1" in LARGE_MODELS:
        return _load_llm(LARGE_MODELS["llm1"])
    return _load_llm(LARGE_MODELS["llm2"])
