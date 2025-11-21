"""Dataset loader for tickets CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parent / "tickets.csv"


def load_dataset(limit: int | None = None) -> pd.DataFrame:
    """Load the tickets dataset with optional row limit."""
    df = pd.read_csv(DEFAULT_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "document" in df.columns:
        df = df.rename(columns={"document": "text"})
    if "topic_group" in df.columns:
        df = df.rename(columns={"topic_group": "label"})
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df[df["label"].astype(str).str.strip() != ""]
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def preprocess_ticket(row: pd.Series) -> Dict[str, object]:
    """Convert row into ticket dict."""
    return {
        "id": row.name,
        "text": str(row.get("text", "")).strip(),
        "category": row.get("label"),
        "metadata": {},
    }


def load_and_preprocess(limit: int | None = None) -> List[Dict[str, object]]:
    """Load and preprocess tickets into a list of dicts."""
    df = load_dataset(limit=limit)
    return [preprocess_ticket(row) for _, row in df.iterrows()]
