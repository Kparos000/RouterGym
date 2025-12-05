"""Classifier suite exports and registry setup."""

from __future__ import annotations

from RouterGym.classifiers.encoder_classifier import EncoderClassifier
from RouterGym.classifiers.slm_classifier import SLMClassifier
from RouterGym.classifiers.tfidf_classifier import TFIDFClassifier
from RouterGym.classifiers.utils import (
    ClassifierMetadata,
    ClassifierProtocol,
    available_classifiers,
    get_classifier,
    register_classifier,
)

# Register default classifiers in deterministic order.
register_classifier("tfidf", TFIDFClassifier)
register_classifier("encoder", EncoderClassifier)
register_classifier("slm_finetuned", SLMClassifier)

CLASSIFIER_MODES = available_classifiers()


def get_classifier_instance(name: str):
    """Instantiate a classifier by registry key."""
    return get_classifier(name)


__all__ = [
    "CLASSIFIER_MODES",
    "ClassifierMetadata",
    "ClassifierProtocol",
    "EncoderClassifier",
    "SLMClassifier",
    "TFIDFClassifier",
    "get_classifier_instance",
]
