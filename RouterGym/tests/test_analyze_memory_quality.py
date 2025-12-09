"""Tests for analyze_memory_quality utility."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from RouterGym.scripts import analyze_memory_quality as memory_script


def _sample_memory_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "memory_mode": "rag_dense",
                "memory_relevance_score": 0.9,
                "retrieval_latency_ms": 10,
                "retrieved_context_length": 500,
            },
            {
                "memory_mode": "rag_dense",
                "memory_relevance_score": 0.8,
                "retrieval_latency_ms": 12,
                "retrieved_context_length": 600,
            },
            {
                "memory_mode": "rag_bm25",
                "memory_relevance_score": 0.2,
                "retrieval_latency_ms": 5,
                "retrieved_context_length": 200,
            },
        ]
    )


def test_prints_summary(monkeypatch: Any, capsys: Any) -> None:
    data = _sample_memory_results()

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        return data

    monkeypatch.setattr(memory_script.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        memory_script,
        "parse_args",
        lambda: argparse.Namespace(results_path="dummy.csv", output_path=None),
    )

    memory_script.main()

    captured = capsys.readouterr().out
    assert "rag_dense" in captured
    assert "rag_bm25" in captured
    assert "mean=" in captured


def test_writes_output_csv(monkeypatch: Any, tmp_path: Path) -> None:
    data = _sample_memory_results()

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        return data

    output_path = tmp_path / "memory_quality.csv"
    monkeypatch.setattr(memory_script.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        memory_script,
        "parse_args",
        lambda: argparse.Namespace(results_path="dummy.csv", output_path=str(output_path)),
    )

    memory_script.main()

    assert output_path.exists()
    with output_path.open("r", newline="", encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))
    modes = {row["memory_mode"] for row in reader}
    assert modes == {"rag_dense", "rag_bm25"}
