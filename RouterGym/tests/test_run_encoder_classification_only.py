from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from RouterGym.experiments import run_encoder_classification_only as classify_script


class FakeClassifier:
    def __init__(self, probs: dict[str, float]) -> None:
        self._probs = probs
        self.backend_name = "encoder_calibrated"

    def predict_proba(self, text: str) -> dict[str, float]:
        return self._probs


def test_classify_dataframe_and_write(tmp_path: Path, monkeypatch: Any) -> None:
    df = pd.DataFrame(
        {
            "text": ["reset password", "buy laptop"],
            "label": ["Access", "Purchase"],
        }
    )
    classifier = FakeClassifier({"Access": 0.9, "Purchase": 0.1})
    records = classify_script.classify_dataframe(df, classifier, start_offset=5)
    assert len(records) == 2
    assert records[0]["ticket_id"] == 5
    assert records[0]["predicted_label"] == "Access"
    assert records[0]["confidence_bucket"] == "high"
    assert records[0]["correct"] is True

    out_path = tmp_path / "out.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    loaded = pd.read_csv(out_path)
    assert list(loaded.columns[:6]) == [
        "ticket_id",
        "gold_label",
        "predicted_label",
        "predicted_confidence",
        "confidence_bucket",
        "correct",
    ]
    assert len(loaded) == 2
