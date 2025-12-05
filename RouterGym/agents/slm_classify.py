"""Adapter exposing the SLM classifier through the unified interface."""

from __future__ import annotations

from typing import Any, Dict

from RouterGym.classifiers import get_classifier_instance
from RouterGym.classifiers.utils import ClassifierProtocol, canonical_label


class SLMClassifierAgent:
    """Wrapper that keeps backward compatibility for legacy agent callers."""

    def __init__(self, model_id: str | None = None) -> None:
        self.mode = "slm_finetuned"
        self._classifier: ClassifierProtocol = get_classifier_instance(self.mode)
        if model_id:
            # Shadow reference to the requested model so downstream logging can expose it.
            self._classifier.metadata.model_reference = model_id  # type: ignore[misc]

    def classify(self, prompt: str, gold_category: str | None = None) -> Dict[str, Any]:
        probabilities = self._classifier.predict_proba(prompt)
        label = self._classifier.predict_label(prompt)
        payload: Dict[str, Any] = {
            "predicted_category": label,
            "confidence": probabilities.get(label, 0.0),
            "probabilities": probabilities,
            "classifier_mode": self.mode,
            "metadata": self._classifier.metadata.as_dict(),
        }
        if gold_category:
            payload["accuracy"] = 1.0 if canonical_label(gold_category) == label else 0.0
        return payload


def classify_text(prompt: str, gold_category: str | None = None) -> Dict[str, Any]:
    """Functional helper mirroring the agent interface."""
    agent = SLMClassifierAgent()
    return agent.classify(prompt, gold_category=gold_category)


__all__ = ["SLMClassifierAgent", "classify_text"]
