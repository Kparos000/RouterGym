from __future__ import annotations

import logging

import pytest

from RouterGym import label_space


def test_canonicalize_label_variants() -> None:
    assert label_space.canonicalize_label("HR Support") == "hr support"
    assert label_space.canonicalize_label("hr_support") == "hr support"
    assert label_space.canonicalize_label("  Admin Rights ") == "administrative rights"
    assert label_space.canonicalize_label("vpn access issue") == "access"


def test_canonicalize_label_unknown_maps_to_misc(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert label_space.canonicalize_label("randomjunk") == "miscellaneous"
    assert any("Unexpected label" in rec.message for rec in caplog.records)
