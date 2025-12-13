"""Tests covering the RouterGym classifier suite."""

from __future__ import annotations

import math

import os

from RouterGym.agents.generator import CLASS_LABELS
from RouterGym.agents.slm_classify import SLMClassifierAgent
from RouterGym.classifiers import CLASSIFIER_MODES, get_classifier_instance
from RouterGym.routing.router_engine import RouterEngine

SAMPLE_TEXT = "Need help resetting my password and unlocking my account."
LABEL_SET = {label for label in CLASS_LABELS}


def test_classifier_modes_match_expected_order() -> None:
    assert CLASSIFIER_MODES == ["tfidf", "encoder", "slm_finetuned"]


def test_each_classifier_returns_normalized_distribution() -> None:
    original = os.environ.get("ROUTERGYM_ALLOW_ENCODER_FALLBACK")
    os.environ["ROUTERGYM_ALLOW_ENCODER_FALLBACK"] = "1"
    for mode in CLASSIFIER_MODES:
        classifier = get_classifier_instance(mode)
        probabilities = classifier.predict_proba(SAMPLE_TEXT)
        assert set(probabilities).issubset(LABEL_SET)
        total = sum(probabilities.values())
        assert math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3)
        predicted = classifier.predict_label(SAMPLE_TEXT)
        assert predicted in probabilities
    if original is None:
        os.environ.pop("ROUTERGYM_ALLOW_ENCODER_FALLBACK", None)
    else:
        os.environ["ROUTERGYM_ALLOW_ENCODER_FALLBACK"] = original


def test_router_engine_emits_metrics_payload() -> None:
    ticket = {"text": SAMPLE_TEXT, "gold_category": "access"}
    engine = RouterEngine("tfidf")
    summary = engine.classify_ticket(ticket)
    payload = summary.as_dict("tfidf")
    assert payload["classifier_mode"] == "tfidf"
    assert 0.0 <= payload["classifier_confidence"] <= 1.0
    assert payload["classifier_accuracy"] in {0.0, 1.0}
    assert "classifier_metadata" in payload


def test_slm_classifier_agent_wraps_registry() -> None:
    agent = SLMClassifierAgent()
    result = agent.classify(SAMPLE_TEXT, gold_category="access")
    assert result["classifier_mode"] == "slm_finetuned"
    assert result["predicted_category"] in LABEL_SET
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
