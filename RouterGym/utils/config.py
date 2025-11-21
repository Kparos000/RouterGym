"""Environment and configuration loader for RouterGym."""

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Optional


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Fetch an environment variable with optional default and required enforcement."""
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass
class ModelSettings:
    slm_model: Optional[str]
    llm_model: Optional[str]
    slm_model_path: Optional[str]
    llm_model_path: Optional[str]
    engine_url: Optional[str]


@dataclass
class ProviderKeys:
    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str]
    azure_openai_endpoint: Optional[str]
    azure_openai_key: Optional[str]
    together_api_key: Optional[str]


@dataclass
class Settings:
    env: str
    model: ModelSettings
    keys: ProviderKeys


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load environment-backed settings once and cache the result."""
    env = _get_env("ROUTER_GYM_ENV", "local")
    model = ModelSettings(
        slm_model=_get_env("VLLM_SLM_MODEL"),
        llm_model=_get_env("VLLM_LLM_MODEL"),
        slm_model_path=_get_env("VLLM_SLM_MODEL_PATH"),
        llm_model_path=_get_env("VLLM_LLM_MODEL_PATH"),
        engine_url=_get_env("VLLM_ENGINE_URL"),
    )
    keys = ProviderKeys(
        openai_api_key=_get_env("OPENAI_API_KEY"),
        anthropic_api_key=_get_env("ANTHROPIC_API_KEY"),
        azure_openai_endpoint=_get_env("AZURE_OPENAI_ENDPOINT"),
        azure_openai_key=_get_env("AZURE_OPENAI_KEY"),
        together_api_key=_get_env("TOGETHER_API_KEY"),
    )
    return Settings(env=env, model=model, keys=keys)


__all__ = ["Settings", "ModelSettings", "ProviderKeys", "load_settings"]
