"""RouterEngine orchestrates classifier inference and metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from RouterGym.classifiers import CLASSIFIER_MODES as REGISTERED_MODES, get_classifier_instance
from RouterGym.classifiers.utils import ClassifierMetadata, ClassifierProtocol, canonical_label, canonical_mode
from RouterGym.memory.base import MemoryRetrieval

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
    memory_context_used: str
    memory_relevance_score: float
    memory_cost_tokens: int
    memory_mode: str
    retrieval_latency_ms: float
    retrieved_context_length: int

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
            "memory_context_used": self.memory_context_used,
            "memory_relevance_score": self.memory_relevance_score,
            "memory_cost_tokens": self.memory_cost_tokens,
            "memory_mode": self.memory_mode,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "retrieved_context_length": self.retrieved_context_length,
        }


class RouterEngine:
    """Coordinates classifier selection and summary generation."""

    def __init__(self, classifier_mode: str = "tfidf") -> None:
        self.classifier_mode = canonical_mode(classifier_mode)
        self._classifier = self._init_classifier(self.classifier_mode)

    def _init_classifier(self, mode: str) -> ClassifierProtocol:
        classifier = get_classifier_instance(mode)
        if not isinstance(classifier, ClassifierProtocol):
            raise TypeError(f"Classifier '{mode}' does not implement ClassifierProtocol")
        return classifier

    @property
    def metadata(self) -> ClassifierMetadata:
        return self._classifier.metadata

    def classify_ticket(
        self,
        ticket: Dict[str, Any],
        memory_result: Optional[MemoryRetrieval] = None,
        memory_mode: str = "none",
    ) -> ClassificationSummary:
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
        memory_context = memory_result.retrieved_context if memory_result else ""
        memory_relevance = memory_result.relevance_score if memory_result else 0.0
        memory_cost = memory_result.retrieval_cost_tokens if memory_result else 0
        memory_latency = memory_result.retrieval_latency_ms if memory_result else 0.0
        context_length = memory_result.retrieved_context_length if memory_result else 0
        metadata.update(
            {
                "memory_available": bool(memory_context),
                "memory_mode": memory_mode,
                "memory_relevance_score": memory_relevance,
                "retrieval_latency_ms": memory_latency,
                "retrieved_context_length": context_length,
            }
        )
        return ClassificationSummary(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
            token_cost=token_cost,
            accuracy=accuracy,
            efficiency=efficiency,
            metadata=metadata,
            memory_context_used=memory_context,
            memory_relevance_score=memory_relevance,
            memory_cost_tokens=memory_cost,
            memory_mode=memory_mode,
            retrieval_latency_ms=memory_latency,
            retrieved_context_length=context_length,
        )

    def set_mode(self, classifier_mode: str) -> None:
        mode = canonical_mode(classifier_mode)
        if mode == self.classifier_mode:
            return
        self.classifier_mode = mode
        self._classifier = self._init_classifier(mode)


__all__ = ["RouterEngine", "ClassificationSummary", "CLASSIFIER_MODES"]
