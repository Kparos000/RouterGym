"""Tests for analyze_prompt_compliance utility."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from RouterGym.scripts import analyze_prompt_compliance as compliance_script


def _sample_compliance_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "router": "llm_first",
                "model": "slm1",
                "json_valid": 1,
                "schema_valid": 1,
                "predicted_category": "access",
            },
            {
                "router": "llm_first",
                "model": "slm1",
                "json_valid": 0,
                "schema_valid": 0,
                "predicted_category": "miscellaneous",
            },
            {
                "router": "hybrid_specialist",
                "model": "slm2",
                "json_valid": 1,
                "schema_valid": 1,
                "predicted_category": "hardware",
            },
        ]
    )


def test_compliance_summary_prints(monkeypatch: Any, capsys: Any) -> None:
    data = _sample_compliance_results()

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        return data

    monkeypatch.setattr(compliance_script.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        compliance_script,
        "parse_args",
        lambda: argparse.Namespace(results_path="dummy.csv"),
    )

    compliance_script.main()

    captured = capsys.readouterr().out
    assert "=== Global ===" in captured
    assert "=== By model ===" in captured
    assert "=== By router ===" in captured
    assert "misc_rate" in captured
