"""Analyze prompt/schema compliance and label usage across routers and models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "router",
    "model",
    "json_valid",
    "schema_valid",
    "predicted_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize JSON/schema compliance and label usage (miscellaneous)."
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="Path to the experiment results CSV.",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {', '.join(missing)}")


def _normalize_bool(series: pd.Series) -> pd.Series:
    """Convert booleans/0-1 strings to numeric 0/1 with NaN for unknowns."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric


def _compute_metrics(df: pd.DataFrame, has_schema_validity: bool) -> Dict[str, float]:
    json_rate = float(_normalize_bool(df["json_valid"]).mean())
    schema_rate = float(_normalize_bool(df["schema_valid"]).mean())
    misc_rate = float((df["predicted_category"] == "miscellaneous").mean())

    metrics: Dict[str, float] = {
        "json_valid_rate": json_rate,
        "schema_valid_rate": schema_rate,
        "misc_rate": misc_rate,
    }

    if has_schema_validity:
        metrics["schema_validity_mean"] = float(
            pd.to_numeric(df["schema_validity"], errors="coerce").mean()
        )
    return metrics


def _print_metrics_block(title: str, items: Iterable[tuple[str, Dict[str, float]]]) -> None:
    print(f"=== {title} ===")
    for name, metrics in items:
        suffix = (
            f", schema_validity_mean={metrics['schema_validity_mean']:.3f}"
            if "schema_validity_mean" in metrics
            else ""
        )
        print(
            f"{name} : json={metrics['json_valid_rate']:.3f}, "
            f"schema={metrics['schema_valid_rate']:.3f}, "
            f"misc_rate={metrics['misc_rate']:.3f}{suffix}"
        )
    print()


def analyze_prompt_compliance(results_path: Path) -> None:
    df = pd.read_csv(results_path)
    _validate_columns(df)

    has_schema_validity = "schema_validity" in df.columns

    # Global metrics.
    global_metrics = _compute_metrics(df, has_schema_validity)
    print("=== Global ===")
    suffix = (
        f", schema_validity_mean={global_metrics['schema_validity_mean']:.3f}"
        if "schema_validity_mean" in global_metrics
        else ""
    )
    print(
        f"json_valid_rate={global_metrics['json_valid_rate']:.3f}, "
        f"schema_valid_rate={global_metrics['schema_valid_rate']:.3f}, "
        f"misc_rate={global_metrics['misc_rate']:.3f}{suffix}\n"
    )

    # By model.
    model_items = []
    for model in sorted(df["model"].dropna().unique()):
        subset = df[df["model"] == model]
        model_items.append((model, _compute_metrics(subset, has_schema_validity)))
    _print_metrics_block("By model", model_items)

    # By router.
    router_items = []
    for router in sorted(df["router"].dropna().unique()):
        subset = df[df["router"] == router]
        router_items.append((router, _compute_metrics(subset, has_schema_validity)))
    _print_metrics_block("By router", router_items)


def main() -> None:
    args = parse_args()
    analyze_prompt_compliance(Path(args.results_path))


if __name__ == "__main__":
    main()
