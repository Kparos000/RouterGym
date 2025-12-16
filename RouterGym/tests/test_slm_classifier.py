"""Tests for SLM classifier behaviour and misc overrides."""

from __future__ import annotations

from typing import Any, Dict

from RouterGym.agents import generator
from RouterGym.classifiers.slm_classifier import SLMClassifier


def test_misc_overridden_by_purchase_prior(monkeypatch: Any) -> None:
    """When SLM predicts misc but purchase keywords dominate, override to purchase."""
    clf = SLMClassifier()

    # Force predict_proba to return misc as top
    def fake_predict_proba(text: str) -> Dict[str, float]:
        return {label: (1.0 if label == "Miscellaneous" else 0.0) for label in clf.labels}

    monkeypatch.setattr(clf, "predict_proba", fake_predict_proba)

    text = "Need to renew my software subscription and pay the invoice to the vendor."
    label = clf.predict_label(text)
    assert label == "Purchase"


def test_misc_stays_when_prior_is_weak(monkeypatch: Any) -> None:
    """If prior is weak/generic, keep miscellaneous."""
    clf = SLMClassifier()

    def fake_predict_proba(text: str) -> Dict[str, float]:
        return {label: (1.0 if label == "Miscellaneous" else 0.0) for label in clf.labels}

    monkeypatch.setattr(clf, "predict_proba", fake_predict_proba)

    text = "General inquiry with no clear signal"
    label = clf.predict_label(text)
    assert label == "Miscellaneous"


def test_hr_keywords_override_misc(monkeypatch: Any) -> None:
    """Strong HR cues should override a misc guess."""
    clf = SLMClassifier()

    def fake_predict_proba(text: str) -> Dict[str, float]:
        return {label: (1.0 if label == "Miscellaneous" else 0.0) for label in clf.labels}

    monkeypatch.setattr(clf, "predict_proba", fake_predict_proba)

    text = "My payroll and benefits need correction and HR must fix it."
    label = clf.predict_label(text)
    assert label == "HR Support"


def test_classification_prompt_contains_labels() -> None:
    prompt = generator.classification_instruction()
    for label in [
        "Access",
        "Administrative rights",
        "Hardware",
        "HR Support",
        "Purchase",
        "Internal Project",
        "Storage",
        "Miscellaneous",
    ]:
        assert label in prompt
    assert "strict json" in prompt.lower()
    assert "only use 'miscellaneous'" in prompt.lower()


def test_model_retry_and_parsing(monkeypatch: Any) -> None:
    calls: list[str] = []

    def fake_call_model(model: Any, prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "not json"
        return '{"category": "Purchase", "reasoning": "buy hardware"}'

    monkeypatch.setattr(generator, "_call_model", fake_call_model)
    clf = SLMClassifier(model="dummy-model")
    label = clf.predict_label("Need to buy new monitors and pay invoice")
    assert label == "Purchase"
    assert len(calls) >= 2


def test_model_misc_overridden_by_prior(monkeypatch: Any) -> None:
    def fake_call_model(model: Any, prompt: str) -> str:
        return '{"category": "miscellaneous", "reasoning": "unsure"}'

    monkeypatch.setattr(generator, "_call_model", fake_call_model)
    clf = SLMClassifier(model="dummy-model")
    label = clf.predict_label("renew software subscription and vendor invoice payment")
    assert label == "Purchase"
