"""Tests for dataset loader utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from RouterGym.data.tickets import dataset_loader


def test_load_dataset_success(tmp_path: Path, monkeypatch: Any) -> None:
    """Ensure dataset load applies column normalization and limits."""
    csv_path = tmp_path / "tickets.csv"
    df = pd.DataFrame({"Document": ["hello"], "Topic_group": ["greeting"]})
    df.to_csv(csv_path, index=False)
    # monkeypatch default path
    monkeypatch.setattr(dataset_loader, "DEFAULT_PATH", csv_path)
    loaded = dataset_loader.load_dataset(1)
    assert list(loaded.columns) == ["text", "label"]
    assert len(loaded) == 1


def test_load_dataset_missing_columns(tmp_path: Path, monkeypatch: Any) -> None:
    """Missing required columns raises ValueError."""
    csv_path = tmp_path / "tickets.csv"
    pd.DataFrame({"foo": ["bar"]}).to_csv(csv_path, index=False)
    monkeypatch.setattr(dataset_loader, "DEFAULT_PATH", csv_path)
    with pytest.raises(ValueError):
        dataset_loader.load_dataset(1)
