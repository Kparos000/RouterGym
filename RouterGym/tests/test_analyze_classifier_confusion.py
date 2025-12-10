"""Tests for analyze_classifier_confusion utility."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from RouterGym.scripts import analyze_classifier_confusion as confusion_script


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"classifier_mode": "tfidf", "gold_category": "access", "predicted_category": "access"},
            {"classifier_mode": "tfidf", "gold_category": "access", "predicted_category": "hardware"},
            {"classifier_mode": "encoder", "gold_category": "hardware", "predicted_category": "hardware"},
            {"classifier_mode": "encoder", "gold_category": "hardware", "predicted_category": "hardware"},
        ]
    )


def test_main_prints_confusion_and_accuracy(monkeypatch: Any, capsys: Any) -> None:
    data = _sample_results()

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        return data

    monkeypatch.setattr(confusion_script.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        confusion_script,
        "parse_args",
        lambda: argparse.Namespace(results_path="dummy.csv", output_dir=None),
    )

    confusion_script.main()

    captured = capsys.readouterr().out
    assert "=== Classifier: encoder ===" in captured
    assert "=== Classifier: tfidf ===" in captured
    assert "Overall accuracy" in captured
    assert "access" in captured
    assert "hardware" in captured
    assert "Headline encoder accuracy" in captured
    assert "Global average accuracy across classifier modes" in captured
    assert "mean_overall_accuracy=" in captured


def test_output_dir_writes_csvs(monkeypatch: Any, tmp_path: Path) -> None:
    data = _sample_results()

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        return data

    monkeypatch.setattr(confusion_script.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        confusion_script,
        "parse_args",
        lambda: argparse.Namespace(results_path="dummy.csv", output_dir=str(tmp_path)),
    )

    confusion_script.main()

    assert (tmp_path / "confusion_tfidf.csv").exists()
    assert (tmp_path / "per_class_tfidf.csv").exists()
