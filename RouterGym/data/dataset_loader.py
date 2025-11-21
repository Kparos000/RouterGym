"""Dataset loader utilities for RouterGym tickets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_SPLIT_SEED = 42
DEFAULT_PATH = Path(__file__).resolve().parent / "tickets.csv"


def load_tickets(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load tickets CSV, standardize columns, drop empty rows, and validate schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Map actual columns
    if "document" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"document": "text"})
    if "topic_group" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"topic_group": "label"})

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df[df["label"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def load_dataset(limit: int | None = None) -> pd.DataFrame:
    """Convenience loader for tickets with optional limit."""
    df = load_tickets(DEFAULT_PATH)
    if limit is not None:
        df = df.head(limit)
    return df


def preprocess_ticket(row: pd.Series) -> Dict[str, object]:
    """Convert a ticket row into a structured dict."""
    text = str(row.get("text", "")).strip()
    category = row.get("label")
    ticket_id = row.name
    return {
        "id": ticket_id,
        "text": text,
        "category": category,
        "metadata": {},
    }


def load_and_preprocess(path: str | Path = DEFAULT_PATH, limit: int | None = None) -> List[Dict[str, object]]:
    """Load tickets, validate, and preprocess into a list of dicts."""
    df = load_tickets(path)
    if limit is not None:
        df = df.head(limit)
    return [preprocess_ticket(row) for _, row in df.iterrows()]


# Legacy helpers (retained for compatibility)
def load_kaggle_dataset(path: str | Path) -> pd.DataFrame:
    """Load a Kaggle-exported dataset from CSV/Parquet into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)

    if path.is_dir():
        candidates = list(path.glob("*.csv")) + list(path.glob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No CSV or Parquet files found in {path}")
        return load_kaggle_dataset(candidates[0])

    raise ValueError(f"Unsupported dataset format: {path}")


def preprocess_tickets(df: pd.DataFrame) -> List[dict]:
    """Preprocess ticket dataframe rows into dict records."""
    records: List[dict] = []
    for _, row in df.iterrows():
        records.append(
            {
                "id": row.get("id") or row.get("ticket_id"),
                "title": row.get("title") or row.get("subject"),
                "body": row.get("body") or row.get("description") or "",
                "priority": row.get("priority"),
                "category": row.get("category"),
            }
        )
    return records


def split_dataset(df: pd.DataFrame, train: float = 0.8, val: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test using simple random sampling."""
    if train + val >= 1.0:
        raise ValueError("train + val must be < 1.0")
    df_shuffled = df.sample(frac=1.0, random_state=DEFAULT_SPLIT_SEED).reset_index(drop=True)
    n = len(df_shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    train_df = df_shuffled.iloc[:n_train]
    val_df = df_shuffled.iloc[n_train : n_train + n_val]
    test_df = df_shuffled.iloc[n_train + n_val :]
    return train_df, val_df, test_df
