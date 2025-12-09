"""Analyze classifier confusion matrices and per-class accuracy from a results CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


REQUIRED_COLUMNS = ["classifier_mode", "gold_category", "predicted_category"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-class confusion matrices for classifier results."
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="Path to the experiment results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory to write confusion and per-class accuracy CSVs.",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {', '.join(missing)}")


def _sorted_labels(series: pd.Series) -> List[str]:
    return sorted(series.dropna().astype(str).unique())


def build_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    gold_labels = _sorted_labels(df["gold_category"])
    predicted_labels = _sorted_labels(df["predicted_category"])
    matrix = (
        pd.crosstab(df["gold_category"], df["predicted_category"])
        .reindex(index=gold_labels, columns=predicted_labels, fill_value=0)
        .astype(int)
    )
    matrix.index.name = "gold_category"
    matrix.columns.name = "predicted_category"
    return matrix


def compute_per_class_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    labels = _sorted_labels(df["gold_category"])
    records = []
    for label in labels:
        gold_mask = df["gold_category"] == label
        total = int(gold_mask.sum())
        correct = int((gold_mask & (df["predicted_category"] == label)).sum())
        accuracy = correct / total if total > 0 else float("nan")
        records.append(
            {
                "gold_category": label,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
            }
        )
    per_class = pd.DataFrame.from_records(records).set_index("gold_category")
    return per_class


def _print_section_header(mode: str, correct: int, total: int) -> None:
    overall_accuracy = correct / total if total > 0 else float("nan")
    print(f"=== Classifier: {mode} ===")
    print(f"Overall accuracy: {correct}/{total} = {overall_accuracy:.3f}")


def _print_per_class(per_class: pd.DataFrame) -> None:
    if per_class.empty:
        print("Per-class accuracy: (no rows)")
        return
    max_label_length = max(len(label) for label in per_class.index)
    print("Per-class accuracy:")
    for label, row in per_class.iterrows():
        correct = int(row["correct"])
        total = int(row["total"])
        accuracy = row["accuracy"]
        print(f"{label.ljust(max_label_length)} : {correct}/{total} = {accuracy:.3f}")


def _print_confusion(confusion: pd.DataFrame) -> None:
    print("Confusion matrix (rows=gold, cols=predicted):")
    if confusion.empty:
        print("(empty)")
    else:
        print(confusion.to_string())
    print()


def analyze_results(results_path: Path, output_dir: Path | None = None) -> None:
    df = pd.read_csv(results_path)
    _validate_columns(df)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    classifier_modes = _sorted_labels(df["classifier_mode"])

    for mode in classifier_modes:
        subset = df[df["classifier_mode"] == mode]
        total_rows = int(len(subset))
        correct_rows = int((subset["gold_category"] == subset["predicted_category"]).sum())

        confusion = build_confusion_matrix(subset)
        per_class = compute_per_class_accuracy(subset)

        _print_section_header(mode, correct_rows, total_rows)
        _print_per_class(per_class)
        _print_confusion(confusion)

        if output_dir:
            confusion_path = output_dir / f"confusion_{mode}.csv"
            per_class_path = output_dir / f"per_class_{mode}.csv"
            confusion.to_csv(confusion_path, index_label="gold_category")
            per_class.to_csv(per_class_path, index_label="gold_category")


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    analyze_results(results_path, output_dir)


if __name__ == "__main__":
    main()
