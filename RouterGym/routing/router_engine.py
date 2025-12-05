"""RouterEngine orchestrates classifier inference and metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

from RouterGym.classifiers import CLASSIFIER_MODES as REGISTERED_MODES, get_classifier_instance
from RouterGym.classifiers.utils import ClassifierMetadata, ClassifierProtocol, canonical_label

CLASSIFIER_MODES = REGISTERED_MODES


@dataclass(slots=True)
class ClassificationSummary:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    token_cost: float
    accuracy: float
    efficiency: float
    metadata: Dict[str, Any]

    def as_dict(self, classifier_mode: str) -> Dict[str, Any]:
        return {
            "classifier_mode": classifier_mode,
            "classifier_label": self.label,
            "classifier_confidence": self.confidence,
            "classifier_probabilities": self.probabilities,
            "classifier_latency_ms": self.latency_ms,
            "classifier_token_cost": self.token_cost,
            "classifier_accuracy": self.accuracy,
            "classifier_efficiency_score": self.efficiency,
            "classifier_metadata": self.metadata,
        }


class RouterEngine:
    """Coordinates classifier selection and summary generation."""

    def __init__(self, classifier_mode: str = "tfidf") -> None:
        self.classifier_mode = canonical_label(classifier_mode)
        self._classifier = self._init_classifier(self.classifier_mode)

    def _init_classifier(self, mode: str) -> ClassifierProtocol:
        classifier = get_classifier_instance(mode)
        if not isinstance(classifier, ClassifierProtocol):
            raise TypeError(f"Classifier '{mode}' does not implement ClassifierProtocol")
        return classifier

    @property
    def metadata(self) -> ClassifierMetadata:
        return self._classifier.metadata

    def classify_ticket(self, ticket: Dict[str, Any]) -> ClassificationSummary:
        if isinstance(ticket, dict):
            text = str(ticket.get("text", ""))
            gold_label = canonical_label(ticket.get("gold_category") or ticket.get("category"))
        else:
            text = str(ticket)
            gold_label = ""
        start = time.perf_counter()
        probabilities = self._classifier.predict_proba(text)
        label = self._classifier.predict_label(text)
        latency_ms = (time.perf_counter() - start) * 1000
        confidence = float(probabilities.get(label, 0.0))
        token_cost = float(self._classifier.metadata.token_cost)
        denom = token_cost if token_cost > 0 else 1.0
        accuracy = 1.0 if gold_label and label == gold_label else 0.0
        efficiency = accuracy / denom
        metadata = self._classifier.metadata.as_dict()
        return ClassificationSummary(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
            token_cost=token_cost,
            accuracy=accuracy,
            efficiency=efficiency,
            metadata=metadata,
        )

    def set_mode(self, classifier_mode: str) -> None:
        mode = canonical_label(classifier_mode)
        if mode == self.classifier_mode:
            return
        self.classifier_mode = mode
        self._classifier = self._init_classifier(mode)


__all__ = ["RouterEngine", "ClassificationSummary", "CLASSIFIER_MODES"]
