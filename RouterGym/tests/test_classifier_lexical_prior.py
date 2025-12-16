"""Tests for lexical prior utility on classifier probabilities."""

from __future__ import annotations

from math import isclose

from RouterGym.agents.generator import classification_instruction
from RouterGym.classifiers.utils import apply_lexical_prior


BASE = {
    "Access": 0.125,
    "Administrative rights": 0.125,
    "Hardware": 0.125,
    "HR Support": 0.125,
    "Purchase": 0.125,
    "Internal Project": 0.125,
    "Storage": 0.125,
    "Miscellaneous": 0.125,
}


def test_purchase_text_boosts_purchase() -> None:
    text = "We need to raise a purchase order for new licenses"
    adjusted = apply_lexical_prior(text, BASE)
    assert adjusted["Purchase"] > BASE["Purchase"]
    assert adjusted["Purchase"] > adjusted["Hardware"]


def test_hardware_text_boosts_hardware() -> None:
    text = "My laptop screen is broken and needs replacement"
    adjusted = apply_lexical_prior(text, BASE)
    assert adjusted["Hardware"] > BASE["Hardware"]
    assert adjusted["Hardware"] > adjusted["HR Support"]


def test_neutral_text_stays_close() -> None:
    text = "Hello there with no clear signal"
    adjusted = apply_lexical_prior(text, BASE)
    diff = sum(abs(adjusted[k] - BASE[k]) for k in BASE)
    assert diff < 1e-3
    assert isclose(sum(adjusted.values()), 1.0, rel_tol=1e-6)


def test_misc_prior_is_downweighted() -> None:
    text = "password reset other category maybe general other"
    adjusted = apply_lexical_prior(text, BASE)
    assert adjusted["Access"] >= adjusted["Miscellaneous"]


def test_classification_instruction_mentions_all_labels_and_misc_rule() -> None:
    prompt = classification_instruction()
    for label in ["Access", "Administrative rights", "Hardware", "HR Support", "Purchase", "Internal Project", "Storage", "Miscellaneous"]:
        assert label in prompt
    assert "miscellaneous" in prompt.lower()
    assert "only" in prompt.lower()


def test_hr_support_keywords_dominate_misc() -> None:
    text = "My payroll and vacation hours are incorrect; HR needs to fix my benefits."
    adjusted = apply_lexical_prior(text, BASE)
    assert adjusted["HR Support"] > adjusted["Miscellaneous"]
    assert adjusted["HR Support"] > adjusted["Hardware"]


def test_purchase_keywords_dominate_misc() -> None:
    text = "Need to renew my software subscription and get a new invoice from the vendor."
    adjusted = apply_lexical_prior(text, BASE)
    assert adjusted["Purchase"] > adjusted["Miscellaneous"]
    assert adjusted["Purchase"] > adjusted["Access"]


def test_generic_text_keeps_misc_relevant() -> None:
    text = "I have a general question."
    adjusted = apply_lexical_prior(text, BASE)
    assert "Miscellaneous" in adjusted
    # Generic text should not strongly favor concrete labels; misc should be competitive.
    assert adjusted["Miscellaneous"] >= 0.10
