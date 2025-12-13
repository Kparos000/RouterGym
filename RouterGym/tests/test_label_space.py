from __future__ import annotations

import logging

import numpy as np
import pytest

from RouterGym import label_space
from RouterGym.scripts import train_encoder_calibrated_head as trainer


def test_canonical_label_maps_variants_and_unknowns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert label_space.canonical_label("HR Support") == "hr support"
        assert label_space.canonical_label("  Admin Rights ") == "administrative rights"
        assert label_space.canonical_label("benefits question") == "hr support"
        assert label_space.canonical_label("internal project work") == "miscellaneous"
        assert label_space.canonical_label("totally_unknown_label") == "miscellaneous"
    # Unknown labels should trigger a warning and map to miscellaneous
    assert any("Unexpected label" in rec.message for rec in caplog.records)


def test_compute_class_weights_strictness() -> None:
    labels = np.array(label_space.CANONICAL_LABELS, dtype=object)
    weights = trainer._compute_class_weights(labels)
    assert set(weights.keys()) == set(label_space.CANONICAL_LABELS)
    # Inject a bad label and expect failure
    bad = np.array(list(label_space.CANONICAL_LABELS) + ["unknown_label"], dtype=object)
    with pytest.raises(RuntimeError):
        trainer._compute_class_weights(bad)
