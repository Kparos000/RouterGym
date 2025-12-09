"""Tests for lexical prior utility on classifier probabilities."""

from __future__ import annotations

from math import isclose

from RouterGym.agents.generator import classification_instruction
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


def test_misc_prior_is_downweighted() -> None:
    text = "password reset other category maybe general other"
    adjusted = apply_lexical_prior(text, BASE)
    assert adjusted["access"] >= adjusted["miscellaneous"]


def test_classification_instruction_mentions_all_labels_and_misc_rule() -> None:
    prompt = classification_instruction()
    for label in ["access", "administrative rights", "hardware", "hr support", "purchase", "miscellaneous"]:
        assert label in prompt
    assert "miscellaneous" in prompt.lower()
    assert "only" in prompt.lower()
