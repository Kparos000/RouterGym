"""Tests for EncoderClassifier with optional centroid file."""

from __future__ import annotations

from pathlib import Path

import pytest

from RouterGym.classifiers.encoder_classifier import EncoderClassifier
CENTROID_FILE = Path(__file__).resolve().parents[1] / "classifiers" / "encoder_centroids.npz"


@pytest.mark.skipif(not CENTROID_FILE.exists(), reason="encoder centroids not trained yet")
def test_centroid_probabilities_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", "1")
    clf = EncoderClassifier()
    probs = clf.predict_proba("purchase order for monitors")
    total = sum(probs.values())
    assert 0.99 <= total <= 1.01
    assert set(probs.keys()) == set(clf.labels)


@pytest.mark.skipif(not CENTROID_FILE.exists(), reason="encoder centroids not trained yet")
def test_centroid_prefers_relevant_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", "1")
    clf = EncoderClassifier()
    probs = clf.predict_proba("purchase order for monitors")
    assert probs.get("Purchase", 0.0) >= probs.get("Access", 0.0)
    assert probs.get("Purchase", 0.0) >= probs.get("HR Support", 0.0)
    probs_hw = clf.predict_proba("laptop screen broken replace device")
    assert probs_hw.get("Hardware", 0.0) >= probs_hw.get("Purchase", 0.0)
