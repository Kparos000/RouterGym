"""Tests for analyze_router_effectiveness utility."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from RouterGym.scripts import analyze_router_effectiveness as router_script


def _sample_router_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "router": "llm_first",
                "memory_mode": "none",
                "model_used": "slm",
                "accuracy": 1,
                "router_confidence_score": 0.8,
            },
            {
                "router": "llm_first",
                "memory_mode": "rag_dense",
                "model_used": "llm",
                "accuracy": 0,
                "router_confidence_score": 0.6,
            },
            {
                "router": "slm_dominant",
                "memory_mode": "none",
                "model_used": "slm",
                "accuracy": 1,
                "router_confidence_score": 0.9,
            },
        ]
    )


def test_router_effectiveness_prints(monkeypatch: Any, capsys: Any) -> None:
    data = _sample_router_results()

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        return data

    monkeypatch.setattr(router_script.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        router_script,
        "parse_args",
        lambda: argparse.Namespace(results_path="dummy.csv"),
    )

    router_script.main()

    captured = capsys.readouterr().out
    assert "llm_first" in captured
    assert "slm_dominant" in captured
    assert "model_used distribution" in captured
    assert "Router x Memory accuracy" in captured
