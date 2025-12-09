"""Summarize router effectiveness: accuracy, confidence, and model usage patterns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


REQUIRED_COLUMNS = [
    "router",
    "memory_mode",
    "model_used",
    "accuracy",
    "router_confidence_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze router accuracy, confidence, and model usage behaviour."
    )
    parser.add_argument("--results-path", required=True, help="Path to results CSV.")
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {', '.join(missing)}")


def _router_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    metrics: Dict[str, Dict[str, object]] = {}
    routers = sorted(df["router"].dropna().unique())
    for router in routers:
        subset = df[df["router"] == router]
        acc = pd.to_numeric(subset["accuracy"], errors="coerce").mean()
        confidence = pd.to_numeric(subset["router_confidence_score"], errors="coerce").mean()
        distribution = subset["model_used"].value_counts(normalize=True)
        metrics[router] = {
            "accuracy": float(acc),
            "confidence": float(confidence),
            "distribution": distribution,
        }
    return metrics


def _print_router_summary(metrics: Dict[str, Dict[str, object]]) -> None:
    print("=== Router summary ===")
    for router, values in metrics.items():
        print(f"{router}:")
        print(f"  accuracy={values['accuracy']:.3f}")
        print(f"  avg_confidence={values['confidence']:.3f}")
        print("  model_used distribution:")
        distribution: pd.Series = values["distribution"]  # type: ignore[assignment]
        for model_used, prob in distribution.items():
            print(f"    {model_used}: {prob:.3f}")
        print()


def _print_router_memory_table(df: pd.DataFrame) -> None:
    print("=== Router x Memory accuracy ===")
    pivot = pd.pivot_table(
        df,
        index="router",
        columns="memory_mode",
        values="accuracy",
        aggfunc="mean",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print()


def analyze_router_effectiveness(results_path: Path) -> None:
    df = pd.read_csv(results_path)
    _validate_columns(df)

    metrics = _router_metrics(df)
    _print_router_summary(metrics)
    _print_router_memory_table(df)


def main() -> None:
    args = parse_args()
    analyze_router_effectiveness(Path(args.results_path))


if __name__ == "__main__":
    main()
