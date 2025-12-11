"""Tests for eval_encoder_pure_slice diagnostic script (uses synthetic data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from RouterGym.scripts import eval_encoder_pure_slice as eval_script


class DummyClassifier:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[str] = []

    def predict_label(self, text: str) -> str:
        self.calls.append(text)
        if "purchase" in text:
            return "purchase"
        if "laptop" in text:
            return "hardware"
        return "access"


def test_eval_encoder_pure_slice_outputs(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    data = pd.DataFrame(
        {
            "Document": ["purchase order", "laptop broken", "login issue"],
            "Topic_group": ["purchase", "hardware", "access"],
        }
    )

    def fake_read_csv(*args: Any, **kwargs: Any):
        return data

    monkeypatch.setattr(eval_script, "pd", pd)
    monkeypatch.setattr(eval_script, "EncoderClassifier", lambda *args, **kwargs: DummyClassifier())
    monkeypatch.setattr(eval_script, "DATA_PATH", tmp_path / "tickets.csv")
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(eval_script, "_load_data", lambda path: data)

    eval_script.evaluate_slice(0, 3, "centroid")
    captured = capsys.readouterr().out
    assert "Overall accuracy on slice" in captured
    assert "purchase" in captured and "hardware" in captured and "access" in captured
