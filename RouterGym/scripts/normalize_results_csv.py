"""Utility to normalize historical experiment CSV files to the 30-column schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from RouterGym.experiments.run_grid import RESULT_COLUMNS, write_results_csv


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_RESULTS_DIR = ROOT_DIR / "results" / "experiments"
DEFAULT_FILES = [
    EXPERIMENT_RESULTS_DIR / "results.csv",
    EXPERIMENT_RESULTS_DIR / "results_10tickets_backup.csv",
    EXPERIMENT_RESULTS_DIR / "results_200tickets.csv",
]


def _load_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {column: row.get(column, "") for column in RESULT_COLUMNS}
            records.append(normalized)
    return records


def normalize_file(path: Path) -> None:
    records = _load_records(path)
    write_results_csv(path, records)


def normalize_results(files: Iterable[Path]) -> None:
    for file_path in files:
        if not file_path.exists():
            print(f"[normalize] Skipping missing file: {file_path}")
            continue
        print(f"[normalize] Normalizing {file_path}")
        normalize_file(file_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize experiment CSV files")
    parser.add_argument("paths", nargs="*", help="Optional CSV file paths to normalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.paths:
        files = [Path(p) for p in args.paths]
    else:
        files = DEFAULT_FILES
    normalize_results(files)


if __name__ == "__main__":
    main()
