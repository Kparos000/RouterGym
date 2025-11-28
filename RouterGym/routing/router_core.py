"""Router helpers for optional classifier loading."""

from __future__ import annotations

from typing import Any, Optional

from RouterGym.engines.model_registry import load_classifier


def load_classifier_by_name(name: Optional[str]) -> Optional[Any]:
    """Load a classifier from the registry if requested."""
    if not name:
        return None
    if name == "tfidf":
        return load_classifier("classifier_tfidf")
    return load_classifier(name)


__all__ = ["load_classifier_by_name"]
