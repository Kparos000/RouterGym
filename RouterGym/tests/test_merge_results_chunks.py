"""Tests for merge_results_chunks utility."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pytest

from RouterGym.scripts.merge_results_chunks import merge_results


def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_merge_results_happy_path(tmp_path: Path) -> None:
    header = ["router", "memory", "ticket_id"]
    rows1 = [["llm_first", "none", "1"], ["llm_first", "none", "2"]]
    rows2 = [["llm_first", "none", "3"]]

    f1 = tmp_path / "results_part01.csv"
    f2 = tmp_path / "results_part02.csv"
    out = tmp_path / "merged.csv"
    _write_csv(f1, header, rows1)
    _write_csv(f2, header, rows2)

    merge_results(str(tmp_path / "results_part*.csv"), out)

    assert out.exists()
    with out.open("r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0] == header
    assert reader[1:] == rows1 + rows2


def test_merge_results_header_mismatch_raises(tmp_path: Path) -> None:
    header = ["router", "memory", "ticket_id"]
    bad_header = ["router", "ticket_id"]
    f1 = tmp_path / "results_part01.csv"
    f2 = tmp_path / "results_part02.csv"
    out = tmp_path / "merged.csv"
    _write_csv(f1, header, [["llm_first", "none", "1"]])
    _write_csv(f2, bad_header, [["llm_first", "1"]])

    with pytest.raises(ValueError):
        merge_results(str(tmp_path / "results_part*.csv"), out)
