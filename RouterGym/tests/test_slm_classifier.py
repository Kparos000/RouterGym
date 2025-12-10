"""Tests for SLM classifier behaviour and misc overrides."""

from __future__ import annotations

from typing import Any, Dict

from RouterGym.classifiers.slm_classifier import SLMClassifier


def test_misc_overridden_by_purchase_prior(monkeypatch: Any) -> None:
    """When SLM predicts misc but purchase keywords dominate, override to purchase."""
    clf = SLMClassifier()

    # Force predict_proba to return misc as top
    def fake_predict_proba(text: str) -> Dict[str, float]:
        return {label: (1.0 if label == "miscellaneous" else 0.0) for label in clf.labels}

    monkeypatch.setattr(clf, "predict_proba", fake_predict_proba)

    text = "Need to renew my software subscription and pay the invoice to the vendor."
    label = clf.predict_label(text)
    assert label == "purchase"


def test_misc_stays_when_prior_is_weak(monkeypatch: Any) -> None:
    """If prior is weak/generic, keep miscellaneous."""
    clf = SLMClassifier()

    def fake_predict_proba(text: str) -> Dict[str, float]:
        return {label: (1.0 if label == "miscellaneous" else 0.0) for label in clf.labels}

    monkeypatch.setattr(clf, "predict_proba", fake_predict_proba)

    text = "General inquiry with no clear signal"
    label = clf.predict_label(text)
    assert label == "miscellaneous"
