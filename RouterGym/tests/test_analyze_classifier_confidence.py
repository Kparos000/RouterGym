"""Tests for analyze_classifier_confidence utility."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from RouterGym.scripts import analyze_classifier_confidence as confidence_script


def _sample_confidence_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"classifier_mode": "tfidf", "classifier_confidence": 0.2, "classifier_accuracy": 0},
            {"classifier_mode": "tfidf", "classifier_confidence": 0.8, "classifier_accuracy": 1},
            {"classifier_mode": "encoder", "classifier_confidence": 0.6, "classifier_accuracy": 0},
            {"classifier_mode": "encoder", "classifier_confidence": 0.95, "classifier_accuracy": 1},
        ]
    )


def test_confidence_bins_print(monkeypatch: Any, capsys: Any) -> None:
    data = _sample_confidence_results()

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        return data

    monkeypatch.setattr(confidence_script.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        confidence_script,
        "parse_args",
        lambda: argparse.Namespace(results_path="dummy.csv"),
    )

    confidence_script.main()

    captured = capsys.readouterr().out
    assert "=== Classifier: tfidf ===" in captured
    assert "=== Classifier: encoder ===" in captured
    assert "mean_conf=" in captured
    assert "empirical_acc=" in captured
