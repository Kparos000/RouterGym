"""Summarize RouterGym benchmark result JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional


def _iter_result_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.jsonl")):
        name = path.name
        if "failures" in name:
            continue
        if "results" not in name:
            continue
        yield path


def _config_id_for_path(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name
    parts = rel.parts
    if "merged" in parts:
        idx = parts.index("merged")
        if idx > 0:
            return parts[idx - 1]
    if "chunks" in parts:
        idx = parts.index("chunks")
        if idx > 0:
            return parts[idx - 1]
    if len(parts) >= 2:
        return parts[-2]
    return path.parent.name


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                yield {
                    "success": False,
                    "parse_error": f"Result JSONL decode failed at {path}:{line_number}",
                    "_source_path": str(path),
                    "_line_number": line_number,
                }
                continue
            if isinstance(parsed, dict):
                parsed.setdefault("_source_path", str(path))
                parsed.setdefault("_line_number", line_number)
                yield parsed


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _average(values: List[float]) -> Optional[float]:
    return sum(values) / float(len(values)) if values else None


def _metric(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None and isinstance(row.get("metrics"), dict):
        value = row["metrics"].get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolution_steps_count(row: Mapping[str, Any]) -> int:
    steps = row.get("resolution_steps")
    return len(steps) if isinstance(steps, list) else 0


def _distribution(rows: Iterable[Mapping[str, Any]], field: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            value = "None"
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    row_count = len(rows)
    success_count = sum(1 for row in rows if row.get("success", True) is not False)
    generation_valid_count = sum(1 for row in rows if bool(row.get("generation_valid")))
    raw_response_saved_count = sum(1 for row in rows if bool(row.get("raw_response_saved")))
    empty_steps_count = sum(
        1
        for row in rows
        if not isinstance(row.get("resolution_steps"), list) or not row.get("resolution_steps")
    )
    unavailable_count = sum(
        1
        for row in rows
        if "LLM unavailable" in str(row.get("final_answer", ""))
        or "LLM unavailable" in str(row.get("raw_model_response_text", ""))
    )
    parse_error_count = sum(1 for row in rows if bool(row.get("parse_error")))
    latencies = [value for row in rows if (value := _metric(row, "latency_ms")) is not None]
    input_tokens = [
        value for row in rows if (value := _metric(row, "total_input_tokens")) is not None
    ]
    output_tokens = [
        value for row in rows if (value := _metric(row, "total_output_tokens")) is not None
    ]
    total_tokens = [value for row in rows if (value := _metric(row, "total_tokens")) is not None]
    total_costs = [value for row in rows if (value := _metric(row, "total_cost_usd")) is not None]
    bad_rows = [
        {
            "ticket_id": row.get("ticket_id"),
            "parse_error": row.get("parse_error"),
            "validation_error": row.get("validation_error"),
            "generation_valid": row.get("generation_valid"),
            "raw_response_saved": row.get("raw_response_saved"),
            "resolution_steps_count": _resolution_steps_count(row),
            "source_path": row.get("_source_path"),
            "line_number": row.get("_line_number"),
        }
        for row in rows
        if row.get("success", True) is False
        or not bool(row.get("generation_valid"))
        or bool(row.get("parse_error"))
        or not bool(row.get("raw_response_saved"))
        or not isinstance(row.get("resolution_steps"), list)
        or not row.get("resolution_steps")
    ][:5]
    return {
        "row_count": row_count,
        "success_rate": _rate(success_count, row_count),
        "generation_valid_rate": _rate(generation_valid_count, row_count),
        "raw_response_saved_rate": _rate(raw_response_saved_count, row_count),
        "empty_resolution_steps_rate": _rate(empty_steps_count, row_count),
        "llm_unavailable_count": unavailable_count,
        "parse_error_rate": _rate(parse_error_count, row_count),
        "parse_error_count": parse_error_count,
        "gold_label_distribution": _distribution(rows, "gold_label"),
        "predicted_category_distribution": _distribution(rows, "predicted_category"),
        "average_latency_ms": _average(latencies),
        "average_total_input_tokens": _average(input_tokens),
        "average_total_output_tokens": _average(output_tokens),
        "average_total_tokens": _average(total_tokens),
        "average_total_cost_usd": _average(total_costs),
        "sample_bad_rows": bad_rows,
    }


def summarize_benchmark_results(root: Path) -> Dict[str, Any]:
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in _iter_result_files(root):
        config_id = _config_id_for_path(root, path)
        grouped[config_id].extend(_load_jsonl(path))
    return {
        "root": str(root),
        "configs": {config_id: summarize_rows(rows) for config_id, rows in sorted(grouped.items())},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize RouterGym benchmark results.")
    parser.add_argument("--root", type=Path, required=True, help="Benchmark results root.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = summarize_benchmark_results(args.root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
