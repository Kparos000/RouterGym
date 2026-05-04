"""Audit the balanced 60k dissertation benchmark dataset."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_INPUT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_input"
OUTPUT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_outputs"
EXPECTED_ROWS = 60_000
EXPECTED_CONFIGS = 6
EXPECTED_ROWS_PER_CONFIG = 10_000
SAMPLE_LIMIT = 25


COLUMN_GROUP_PATTERNS = {
    "gold labels": ["gold", "label", "topic_group"],
    "predicted categories": ["predicted_category", "prediction"],
    "classifier_predicted_category": ["classifier_predicted_category"],
    "generated_predicted_category": ["generated_predicted_category"],
    "prediction_source": ["prediction_source"],
    "success": ["success"],
    "generation_valid": ["generation_valid"],
    "raw_response_saved": ["raw_response_saved"],
    "parse_error": ["parse_error"],
    "validation_error": ["validation_error"],
    "final_answer": ["final_answer"],
    "reasoning": ["reasoning"],
    "resolution_steps": ["resolution_steps"],
    "escalation flags": ["escalat"],
    "router decisions": ["router", "routing"],
    "token fields": ["token"],
    "cost fields": ["cost"],
    "latency fields": ["latency", "duration", "elapsed"],
    "model telemetry fields": ["telemetry", "model_call"],
    "memory/context fields": ["memory", "context", "kb_", "retrieval"],
    "raw model response fields": ["raw_model_response", "raw_response"],
}


def find_dataset() -> Path:
    candidates = sorted(RESULTS_INPUT_DIR.rglob("balanced_10k_all_configs.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find balanced_10k_all_configs.jsonl under {RESULTS_INPUT_DIR}"
        )
    if len(candidates) > 1:
        print("Multiple balanced_10k_all_configs.jsonl files found; using first:")
        for candidate in candidates:
            print(f"  - {candidate}")
    return candidates[0]


def is_available(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def normalize_ticket_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def detect_columns(columns: list[str]) -> dict[str, list[str]]:
    detected: dict[str, list[str]] = {}
    for group, patterns in COLUMN_GROUP_PATTERNS.items():
        matches = [
            column
            for column in columns
            if any(pattern.lower() in column.lower() for pattern in patterns)
        ]
        detected[group] = sorted(matches)
    return detected


def availability(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    present = sum(1 for row in rows if column in row)
    available = sum(1 for row in rows if is_available(row.get(column)))
    return {
        "column_present_rows": present,
        "available_rows": available,
        "missing_or_empty_rows": len(rows) - available,
        "availability_rate": available / len(rows) if rows else 0.0,
    }


def build_integrity_report(rows: list[dict[str, Any]], dataset_path: Path) -> dict[str, Any]:
    config_counts = Counter(str(row.get("analysis_config", "")) for row in rows)
    config_reports: dict[str, dict[str, Any]] = {}
    ticket_ids_by_config: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        config = str(row.get("analysis_config", ""))
        ticket_id = normalize_ticket_id(row.get("ticket_id"))
        if ticket_id is not None:
            ticket_ids_by_config[config].append(ticket_id)

    for config, count in sorted(config_counts.items()):
        ticket_ids = ticket_ids_by_config.get(config, [])
        ticket_counter = Counter(ticket_ids)
        duplicates = sorted(ticket_id for ticket_id, seen in ticket_counter.items() if seen > 1)
        expected_ids = set(range(EXPECTED_ROWS_PER_CONFIG))
        actual_ids = set(ticket_ids)
        missing_ids = sorted(expected_ids - actual_ids)
        config_reports[config] = {
            "rows": count,
            "min_ticket_id": min(ticket_ids) if ticket_ids else None,
            "max_ticket_id": max(ticket_ids) if ticket_ids else None,
            "unique_ticket_ids": len(actual_ids),
            "duplicate_ticket_id_count": len(duplicates),
            "duplicate_ticket_ids_sample": duplicates[:50],
            "missing_ticket_id_count": len(missing_ids),
            "missing_ticket_ids_sample": missing_ids[:50],
            "expected_0_to_9999_coverage": len(missing_ids) == 0 and len(duplicates) == 0,
        }

    required_availability = {
        column: availability(rows, column)
        for column in [
            "gold_label",
            "predicted_category",
            "success",
            "generation_valid",
            "raw_response_saved",
        ]
    }

    checks = {
        "total_rows_is_60000": len(rows) == EXPECTED_ROWS,
        "exactly_6_analysis_configs": len(config_counts) == EXPECTED_CONFIGS,
        "10000_rows_per_config": all(
            count == EXPECTED_ROWS_PER_CONFIG for count in config_counts.values()
        ),
        "all_configs_have_ticket_0_to_9999_once": all(
            report["expected_0_to_9999_coverage"] for report in config_reports.values()
        ),
        "gold_label_available_all_rows": required_availability["gold_label"]["available_rows"]
        == len(rows),
        "predicted_category_available_all_rows": required_availability["predicted_category"][
            "available_rows"
        ]
        == len(rows),
    }

    return {
        "dataset_path": str(dataset_path),
        "dataset_size_bytes": dataset_path.stat().st_size,
        "total_rows": len(rows),
        "analysis_config_count": len(config_counts),
        "analysis_config_counts": dict(sorted(config_counts.items())),
        "checks": checks,
        "required_column_availability": required_availability,
        "config_integrity": config_reports,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = find_dataset()
    rows = load_jsonl(dataset_path)
    sample_rows = rows[:SAMPLE_LIMIT]
    columns = sorted({column for row in rows[: min(len(rows), 1000)] for column in row})
    detected = detect_columns(columns)
    integrity_report = build_integrity_report(rows, dataset_path)

    (OUTPUT_DIR / "available_columns.txt").write_text("\n".join(columns) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "sample_rows.json").write_text(
        json.dumps(sample_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "dataset_integrity_report.json").write_text(
        json.dumps(integrity_report, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "metric_column_detection_report.json").write_text(
        json.dumps(detected, indent=2), encoding="utf-8"
    )

    print("Balanced 60k schema audit")
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {integrity_report['total_rows']:,}")
    print(f"Configs: {integrity_report['analysis_config_count']}")
    for config, count in integrity_report["analysis_config_counts"].items():
        print(f"  {config}: {count:,} rows")
    print("Checks:")
    for check, passed in integrity_report["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'} {check}")
    print(f"Columns detected: {len(columns)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
