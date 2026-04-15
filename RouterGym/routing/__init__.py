"""Routing package exports.

This package uses lazy attribute loading so importing ``RouterGym.routing`` or a
single routing submodule does not eagerly import the full routing stack and its
generator dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseRouter",
    "CLASSIFIER_MODES",
    "HybridSpecialistRouter",
    "LLMFirstRouter",
    "RouterEngine",
    "SLMDominantRouter",
]


def __getattr__(name: str) -> Any:
    if name == "BaseRouter":
        from RouterGym.routing.base import BaseRouter

        return BaseRouter
    if name in {"RouterEngine", "CLASSIFIER_MODES"}:
        from RouterGym.routing.router_engine import CLASSIFIER_MODES, RouterEngine

        return {"RouterEngine": RouterEngine, "CLASSIFIER_MODES": CLASSIFIER_MODES}[name]
    if name == "LLMFirstRouter":
        from RouterGym.routing.llm_first import LLMFirstRouter

        return LLMFirstRouter
    if name == "SLMDominantRouter":
        from RouterGym.routing.slm_dominant import SLMDominantRouter

        return SLMDominantRouter
    if name == "HybridSpecialistRouter":
        from RouterGym.routing.hybrid_specialist import HybridSpecialistRouter

        return HybridSpecialistRouter
    raise AttributeError(name)
