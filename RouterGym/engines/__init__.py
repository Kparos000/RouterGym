"""Engine package exports via lazy imports.

Keeping this module lazy prevents package import from pulling the model
registry, optional backend probes, or environment loading into unrelated code
paths such as test collection.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LLM_MODELS",
    "SLM_MODELS",
    "RemoteInferenceEngine",
    "get_model_backend",
    "get_repair_model",
    "load_models",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from RouterGym.engines import model_registry

        return getattr(model_registry, name)
    raise AttributeError(name)
