"""Dataset loader for RouterGym tickets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parent / "tickets.csv"


def load_dataset(n: Optional[int] = 5) -> pd.DataFrame:
    """Load tickets dataset and optionally limit rows."""
    df = pd.read_csv(DEFAULT_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "document" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"document": "text"})
    if "topic_group" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"topic_group": "label"})
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df[df["label"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    if n is not None:
        df = df.head(n)
    return df


__all__ = ["load_dataset"]
