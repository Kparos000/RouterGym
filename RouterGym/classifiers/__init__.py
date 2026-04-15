"""Classifier exports with lazy registration.

Importing the classifiers package should not eagerly import the SLM classifier,
because that pulls the generator/model stack into unrelated imports. Default
classifier registration is deferred until the registry is actually queried.
"""

from __future__ import annotations

from typing import Any

from RouterGym.classifiers.paths import HEAD_PATH
from RouterGym.classifiers.utils import (
    ClassifierMetadata,
    ClassifierProtocol,
    available_classifiers as _available_classifiers,
    get_classifier as _get_classifier,
    register_classifier,
)

CLASSIFIER_MODES = ["tfidf", "encoder", "slm_finetuned"]
_REGISTRY_INITIALIZED = False


def _ensure_registry_initialized() -> None:
    global _REGISTRY_INITIALIZED
    if _REGISTRY_INITIALIZED:
        return
    from RouterGym.classifiers.encoder_classifier import EncoderClassifier
    from RouterGym.classifiers.slm_classifier import SLMClassifier
    from RouterGym.classifiers.tfidf_classifier import TFIDFClassifier

    register_classifier("tfidf", TFIDFClassifier)
    register_classifier("encoder", EncoderClassifier)
    register_classifier("slm_finetuned", SLMClassifier)
    _REGISTRY_INITIALIZED = True


def get_classifier_instance(name: str):
    """Instantiate a classifier by registry key."""
    _ensure_registry_initialized()
    return _get_classifier(name)


def available_classifiers() -> list[str]:
    """Return registered classifier names after lazy initialization."""
    _ensure_registry_initialized()
    return _available_classifiers()


def __getattr__(name: str) -> Any:
    if name == "EncoderClassifier":
        from RouterGym.classifiers.encoder_classifier import EncoderClassifier

        return EncoderClassifier
    if name == "SLMClassifier":
        from RouterGym.classifiers.slm_classifier import SLMClassifier

        return SLMClassifier
    if name == "TFIDFClassifier":
        from RouterGym.classifiers.tfidf_classifier import TFIDFClassifier

        return TFIDFClassifier
    raise AttributeError(name)


__all__ = [
    "CLASSIFIER_MODES",
    "ClassifierMetadata",
    "ClassifierProtocol",
    "EncoderClassifier",
    "SLMClassifier",
    "TFIDFClassifier",
    "available_classifiers",
    "get_classifier_instance",
    "HEAD_PATH",
]
