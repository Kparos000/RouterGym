"""Summarize RAG memory quality metrics from experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


REQUIRED_COLUMNS = [
    "memory_mode",
    "memory_relevance_score",
    "retrieval_latency_ms",
    "retrieved_context_length",
]

METRIC_MAP = {
    "memory_relevance_score": "relevance",
    "retrieval_latency_ms": "retrieval_latency",
    "retrieved_context_length": "context_length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze RAG memory quality (relevance, latency, context length)."
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="Path to the experiment results CSV.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional path to write aggregated stats CSV.",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {', '.join(missing)}")


def _compute_stats(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def summarize_memory_quality(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("memory_mode")
    records = []
    for memory_mode, group in grouped:
        row: Dict[str, object] = {"memory_mode": memory_mode}
        for column, prefix in METRIC_MAP.items():
            stats = _compute_stats(group[column])
            row[f"{prefix}_count"] = stats["count"]
            row[f"{prefix}_mean"] = stats["mean"]
            row[f"{prefix}_median"] = stats["median"]
            row[f"{prefix}_min"] = stats["min"]
            row[f"{prefix}_max"] = stats["max"]
        records.append(row)
    return pd.DataFrame.from_records(records).set_index("memory_mode")


def _print_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("No data to summarize.")
        return
    metric_labels = list(METRIC_MAP.keys())
    label_width = max(len(label) for label in metric_labels)

    for memory_mode, row in summary.sort_index().iterrows():
        print(f"Memory mode: {memory_mode}")
        for column, prefix in METRIC_MAP.items():
            count = int(row[f"{prefix}_count"])
            mean = row[f"{prefix}_mean"]
            median = row[f"{prefix}_median"]
            minimum = row[f"{prefix}_min"]
            maximum = row[f"{prefix}_max"]
            label = column.ljust(label_width)
            print(
                f"  {label}: count={count}, mean={mean:.3f}, median={median:.3f}, "
                f"min={minimum:.3f}, max={maximum:.3f}"
            )
        print()


def analyze_memory_quality(results_path: Path, output_path: Path | None = None) -> None:
    df = pd.read_csv(results_path)
    _validate_columns(df)

    summary = summarize_memory_quality(df)
    _print_summary(summary)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index_label="memory_mode")


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_path)
    output_path = Path(args.output_path) if args.output_path else None
    analyze_memory_quality(results_path, output_path)


if __name__ == "__main__":
    main()
