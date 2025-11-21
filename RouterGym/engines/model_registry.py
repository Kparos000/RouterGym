"""Unified model registry for local/remote vLLM engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import os

from RouterGym.utils.config import load_settings
from .vllm_local import LocalVLLMEngine
from .vllm_remote import RemoteVLLMEngine


@dataclass
class ModelConfig:
    """Model configuration entry."""

    name: str
    provider: str  # local or remote
    model: str
    endpoint: str | None = None
    path: str | None = None


def _build_registry() -> Dict[str, ModelConfig]:
    """Create the registry with environment-aware endpoints/paths."""
    settings = load_settings()
    engine_url = settings.model.engine_url or os.getenv("VLLM_ENGINE_URL", "")
    slm_path = settings.model.slm_model_path or os.getenv("VLLM_SLM_MODEL_PATH")
    llm_path = settings.model.llm_model_path or os.getenv("VLLM_LLM_MODEL_PATH")

    return {
        "llama70b": ModelConfig(
            name="llama70b",
            provider="remote",
            model="llama70b",
            endpoint=engine_url,
            path=llm_path,
        ),
        "mixtral46b": ModelConfig(
            name="mixtral46b",
            provider="remote",
            model="mixtral-46b",
            endpoint=engine_url,
            path=llm_path,
        ),
        "phi3mini": ModelConfig(
            name="phi3mini",
            provider="local",
            model="phi3-mini",
            path=slm_path or "phi3-mini",
        ),
        "qwen1.5b": ModelConfig(
            name="qwen1.5b",
            provider="local",
            model="qwen1.5b",
            path=slm_path or "qwen1.5b",
        ),
    }


_REGISTRY: Dict[str, ModelConfig] = _build_registry()


def list_models() -> List[str]:
    """List available model names."""
    return sorted(_REGISTRY.keys())


def get_model_config(name: str) -> ModelConfig:
    """Return model config by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model name: {name}")
    return _REGISTRY[name]


def get_engine(name: str):
    """Get an initialized engine instance for a model."""
    cfg = get_model_config(name)
    if cfg.provider == "remote":
        if not cfg.endpoint:
            raise RuntimeError(f"Missing endpoint for remote model {name}")
        return RemoteVLLMEngine(model=cfg.model, endpoint=cfg.endpoint)
    if cfg.provider == "local":
        model_or_path = cfg.path or cfg.model
        return LocalVLLMEngine(model=model_or_path, enforce_cpu=False)
    raise RuntimeError(f"Unsupported provider for model {name}: {cfg.provider}")
