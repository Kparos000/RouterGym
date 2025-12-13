"""Tests for EncoderClassifier pure centroid mode (no lexical prior)."""

from __future__ import annotations

import pytest

from RouterGym.classifiers.encoder_classifier import EncoderClassifier
from RouterGym.classifiers import utils as utils_mod


def test_pure_mode_skips_lexical(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force lexical prior to raise if called
    monkeypatch.setattr(utils_mod, "apply_lexical_prior", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("lexical called")))
    monkeypatch.setenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", "1")
    clf = EncoderClassifier(use_lexical_prior=False)
    probs = clf.predict_proba("generic text with no centroids")
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert set(probs.keys()).issuperset({"access", "hardware", "purchase", "hr support", "miscellaneous"})
