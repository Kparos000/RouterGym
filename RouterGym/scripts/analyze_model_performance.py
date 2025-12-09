"""Summarize model accuracy overall and by memory mode."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


REQUIRED_COLUMNS = ["model", "memory_mode", "accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-model accuracy and model x memory accuracy."
    )
    parser.add_argument("--results-path", required=True, help="Path to experiment results CSV.")
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {', '.join(missing)}")


def _per_model_accuracy(df: pd.DataFrame) -> Dict[str, float]:
    acc_series = pd.to_numeric(df["accuracy"], errors="coerce")
    grouped = df.assign(accuracy=acc_series).groupby("model")["accuracy"].mean()
    return {str(model): float(value) for model, value in grouped.items()}


def _model_memory_pivot(df: pd.DataFrame) -> pd.DataFrame:
    acc_series = pd.to_numeric(df["accuracy"], errors="coerce")
    pivot = pd.pivot_table(
        df.assign(accuracy=acc_series),
        index="model",
        columns="memory_mode",
        values="accuracy",
        aggfunc="mean",
    )
    return pivot


def analyze_model_performance(results_path: Path) -> None:
    df = pd.read_csv(results_path)
    _validate_columns(df)

    per_model = _per_model_accuracy(df)
    print("=== Per-model accuracy ===")
    for model in sorted(per_model.keys()):
        print(f"{model:<8}: {per_model[model]:.3f}")
    print()

    pivot = _model_memory_pivot(df)
    print("=== Model x Memory accuracy ===")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print()


def main() -> None:
    args = parse_args()
    analyze_model_performance(Path(args.results_path))


if __name__ == "__main__":
    main()
