"""Tests for dataset loader utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from RouterGym.data import dataset_loader


def test_load_tickets_success(tmp_path: Path) -> None:
    """Ensure tickets load and columns are standardized."""
    csv_path = tmp_path / "tickets.csv"
    df = pd.DataFrame({"Text": ["hello"], "Label": ["greeting"]})
    df.to_csv(csv_path, index=False)
    loaded = dataset_loader.load_tickets(csv_path)
    assert list(loaded.columns) == ["text", "label"]
    assert len(loaded) == 1


def test_load_tickets_missing_columns(tmp_path: Path) -> None:
    """Missing required columns raises ValueError."""
    csv_path = tmp_path / "tickets.csv"
    pd.DataFrame({"foo": ["bar"]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        dataset_loader.load_tickets(csv_path)


def test_preprocess_ticket() -> None:
    """Preprocess row into ticket dict."""
    df = pd.DataFrame({"text": ["hello"], "label": ["greeting"]})
    ticket = dataset_loader.preprocess_ticket(df.iloc[0])
    assert ticket["id"] == 0
    assert ticket["text"] == "hello"
    assert ticket["category"] == "greeting"
    assert ticket["metadata"] == {}
