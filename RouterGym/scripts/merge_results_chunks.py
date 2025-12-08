"""Utility to merge per-chunk results CSVs into a single file.

Example:
    python -m RouterGym.scripts.merge_results_chunks \
        --pattern "results_47000tickets_slm1_part*.csv" \
        --output "results_47000tickets_slm1.csv"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def merge_results(pattern: str, output: Path) -> None:
    pattern_path = Path(pattern)
    if pattern_path.anchor:
        base = Path(pattern_path.anchor) / pattern_path.relative_to(pattern_path.anchor).parent
        files = sorted(base.glob(pattern_path.name))
    else:
        files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    header: List[str] | None = None
    rows_written = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.writer(out_handle)
        for file in files:
            with file.open("r", newline="", encoding="utf-8") as in_handle:
                reader = csv.reader(in_handle)
                try:
                    file_header = next(reader)
                except StopIteration:
                    continue  # empty file

                if header is None:
                    header = file_header
                    writer.writerow(header)
                elif header != file_header:
                    raise ValueError(f"Header mismatch in file {file}: expected {header}, got {file_header}")

                for row in reader:
                    writer.writerow(row)
                    rows_written += 1

    print(f"[Merge] Files: {len(files)} | Rows merged: {rows_written} | Output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-chunk results CSVs into a single file.")
    parser.add_argument("--pattern", required=True, help="Glob pattern for input files (e.g., 'results_*_part*.csv')")
    parser.add_argument("--output", required=True, help="Output CSV path for merged results")
    args = parser.parse_args()

    merge_results(args.pattern, Path(args.output))


if __name__ == "__main__":
    main()
