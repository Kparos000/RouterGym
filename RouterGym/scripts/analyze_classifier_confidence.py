"""Analyze classifier confidence calibration across confidence bins."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


REQUIRED_COLUMNS = [
    "classifier_mode",
    "classifier_confidence",
    "classifier_accuracy",
]

BINS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
LABELS = ["[0.0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bin classifier confidences and compare to empirical accuracy."
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="Path to the experiment results CSV.",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {', '.join(missing)}")


def _calibration_by_bin(df: pd.DataFrame) -> List[Dict[str, object]]:
    # Ensure numeric types for calculations.
    confidences = pd.to_numeric(df["classifier_confidence"], errors="coerce")
    accuracies = pd.to_numeric(df["classifier_accuracy"], errors="coerce")

    binned = pd.cut(confidences, bins=BINS, labels=LABELS, include_lowest=True, right=False)
    temp = pd.DataFrame(
        {
            "confidence_bin": binned,
            "classifier_confidence": confidences,
            "classifier_accuracy": accuracies,
        }
    )

    results: List[Dict[str, object]] = []
    for label in LABELS:
        bin_df = temp[temp["confidence_bin"] == label]
        count = int(len(bin_df))
        mean_conf = float(bin_df["classifier_confidence"].mean()) if count > 0 else float("nan")
        empirical_acc = float(bin_df["classifier_accuracy"].mean()) if count > 0 else float("nan")
        results.append(
            {
                "bin": label,
                "count": count,
                "mean_conf": mean_conf,
                "empirical_acc": empirical_acc,
            }
        )
    return results


def analyze_confidence(results_path: Path) -> None:
    df = pd.read_csv(results_path)
    _validate_columns(df)

    classifier_modes = sorted(df["classifier_mode"].dropna().unique())
    for mode in classifier_modes:
        subset = df[df["classifier_mode"] == mode]
        print(f"=== Classifier: {mode} ===")
        for record in _calibration_by_bin(subset):
            print(
                f"{record['bin']:11} : n={record['count']}, "
                f"mean_conf={record['mean_conf']:.3f}, empirical_acc={record['empirical_acc']:.3f}"
            )
        print()


def main() -> None:
    args = parse_args()
    analyze_confidence(Path(args.results_path))


if __name__ == "__main__":
    main()
