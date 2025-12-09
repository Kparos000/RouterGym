"""Tests for analyze_model_performance utility."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from RouterGym.scripts import analyze_model_performance as model_script


def _sample_model_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "slm1", "memory_mode": "none", "accuracy": 1},
            {"model": "slm1", "memory_mode": "rag_dense", "accuracy": 0},
            {"model": "llm1", "memory_mode": "none", "accuracy": 1},
            {"model": "llm1", "memory_mode": "rag_dense", "accuracy": 1},
        ]
    )


def test_model_performance_prints(monkeypatch: Any, capsys: Any) -> None:
    data = _sample_model_results()

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        return data

    monkeypatch.setattr(model_script.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        model_script,
        "parse_args",
        lambda: argparse.Namespace(results_path="dummy.csv"),
    )

    model_script.main()

    captured = capsys.readouterr().out
    assert "Per-model accuracy" in captured
    assert "Model x Memory accuracy" in captured
    assert "slm1" in captured
    assert "llm1" in captured
