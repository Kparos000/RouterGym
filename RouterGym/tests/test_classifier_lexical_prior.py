"""Tests for lexical prior utility on classifier probabilities."""

from __future__ import annotations

from math import isclose

from RouterGym.classifiers.utils import apply_lexical_prior


BASE = {
    "access": 0.2,
    "hardware": 0.2,
    "hr support": 0.2,
    "purchase": 0.2,
    "miscellaneous": 0.2,
}


def test_purchase_text_boosts_purchase() -> None:
    text = "We need to raise a purchase order for new licenses"
    adjusted = apply_lexical_prior(text, BASE)
    assert adjusted["purchase"] > BASE["purchase"]
    assert adjusted["purchase"] > adjusted["hardware"]


def test_hardware_text_boosts_hardware() -> None:
    text = "My laptop screen is broken and needs replacement"
    adjusted = apply_lexical_prior(text, BASE)
    assert adjusted["hardware"] > BASE["hardware"]
    assert adjusted["hardware"] > adjusted["hr support"]


def test_neutral_text_stays_close() -> None:
    text = "Hello there with no clear signal"
    adjusted = apply_lexical_prior(text, BASE)
    diff = sum(abs(adjusted[k] - BASE[k]) for k in BASE)
    assert diff < 1e-3
    assert isclose(sum(adjusted.values()), 1.0, rel_tol=1e-6)
