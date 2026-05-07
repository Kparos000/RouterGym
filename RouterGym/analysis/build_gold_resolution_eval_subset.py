"""Build the gold-resolution evaluation subset from balanced 60k outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_INPUT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_input"
GOLD_PATH = REPO_ROOT / "RouterGym" / "data" / "gold_eval" / "gold_eval_final.jsonl"
OUTPUT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_outputs" / "gold_resolution_eval"
EXPECTED_CONFIGS = 6
EXPECTED_GOLD_OVERLAP = 76
EXPECTED_MATCHED_ROWS = 456

PRESERVED_FIELDS = [
    "ticket_id",
    "ticket_index",
    "analysis_config",
    "config_identifier",
    "router_mode",
    "router_family",
    "base_model",
    "base_model_name",
    "escalation_model",
    "escalation_model_name",
    "memory_mode",
    "context_mode",
    "ticket_request",
    "ticket_text",
    "original_query",
    "rewritten_query",
    "gold_label",
    "classifier_predicted_category",
    "generated_predicted_category",
    "predicted_category",
    "prediction_source",
    "classifier_backend",
    "final_answer",
    "reasoning",
    "resolution_steps",
    "escalation_flags",
    "escalated",
    "escalation_reasons",
    "kb_policy_ids",
    "kb_categories",
    "raw_model_response_text",
    "parse_error",
    "validation_error",
    "generation_valid",
    "generation_parser_mode",
    "raw_response_saved",
    "success",
    "total_cost_usd",
    "slm_cost_usd",
    "llm_cost_usd",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "max_output_tokens",
    "metrics",
    "model_call_telemetry",
    "latency_ms",
]


def find_balanced_dataset() -> Path:
    candidates = sorted(ANALYSIS_INPUT_DIR.rglob("balanced_10k_all_configs.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find balanced_10k_all_configs.jsonl under {ANALYSIS_INPUT_DIR}"
        )
    if len(candidates) > 1:
        print("Multiple balanced datasets found; using first:")
        for candidate in candidates:
            print(f"  {candidate}")
    return candidates[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def parse_config(config: str) -> dict[str, str]:
    tokens = str(config or "").split("__")
    parsed = {
        "router_family": tokens[0] if tokens else "",
        "base_model": "",
        "escalation_model": "",
        "memory_mode": "",
    }
    for token in tokens:
        if token.startswith("base_"):
            parsed["base_model"] = token.removeprefix("base_")
        elif token.startswith("esc_"):
            parsed["escalation_model"] = token.removeprefix("esc_")
        elif token.startswith("mem_"):
            parsed["memory_mode"] = token.removeprefix("mem_")
    return parsed


def ticket_id_as_int(record: dict[str, Any]) -> int | None:
    value = record.get("ticket_id", record.get("ticket_index"))
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def build_gold_index(gold_records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed = {}
    for record in gold_records:
        try:
            ticket_index = int(str(record.get("ticket_index")))
        except (TypeError, ValueError):
            continue
        indexed[ticket_index] = record
    return indexed


def slim_record(record: dict[str, Any], gold_record: dict[str, Any]) -> dict[str, Any]:
    config = str(record.get("analysis_config") or record.get("config_identifier") or "")
    parsed_config = parse_config(config)
    output = {field: record.get(field) for field in PRESERVED_FIELDS if field in record}
    output.setdefault("analysis_config", config)
    for key, value in parsed_config.items():
        output[key] = output.get(key) or value
    output["ticket_id"] = str(record.get("ticket_id", gold_record.get("ticket_index", "")))
    output["gold_ticket_index"] = int(gold_record["ticket_index"])
    output["gold_topic_group"] = gold_record.get("topic_group", "")
    output["gold_ticket_text"] = gold_record.get("ticket_text", "")
    output["gold_resolution"] = gold_record.get("gold_resolution", {})
    output["gold_review_status"] = gold_record.get("review_status", "")
    output["gold_provenance"] = gold_record.get("gold_provenance", {})
    if "latency_ms" not in output:
        metrics = output.get("metrics")
        if isinstance(metrics, dict) and metrics.get("latency_ms") is not None:
            output["latency_ms"] = metrics.get("latency_ms")
    return output


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Gold Resolution Match Report",
        "",
        f"- Balanced dataset: `{report['balanced_dataset_path']}`",
        f"- Gold eval path: `{report['gold_path']}`",
        f"- Gold records: {report['gold_record_count']}",
        f"- Gold tickets present in all configs: {report['gold_tickets_present_all_configs']}",
        f"- Matched production rows: {report['matched_row_count']}",
        f"- Expected overlap check: {'PASS' if report['checks']['expected_overlap_76'] else 'FAIL'}",
        f"- Expected row check: {'PASS' if report['checks']['expected_rows_456'] else 'FAIL'}",
        "",
        "## Rows by Config",
        "",
    ]
    for config, count in report["matched_rows_by_config"].items():
        lines.append(f"- `{config}`: {count}")
    lines.extend(["", "## Gold Category Counts", ""])
    for category, count in report["matched_gold_category_counts"].items():
        lines.append(f"- {category}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    balanced_path = find_balanced_dataset()
    gold_records = read_jsonl(GOLD_PATH)
    gold_index = build_gold_index(gold_records)
    gold_ids = set(gold_index)

    rows_by_ticket_config: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    config_counts: Counter[str] = Counter()
    all_config_ids: set[str] = set()

    with balanced_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            ticket_id = ticket_id_as_int(record)
            if ticket_id is None or ticket_id not in gold_ids:
                continue
            config = str(record.get("analysis_config") or record.get("config_identifier") or "")
            if not config:
                continue
            all_config_ids.add(config)
            rows_by_ticket_config[ticket_id][config] = record
            config_counts[config] += 1

    complete_gold_ids = sorted(
        ticket_id
        for ticket_id, config_map in rows_by_ticket_config.items()
        if len(config_map) == EXPECTED_CONFIGS
    )
    matched_records: list[dict[str, Any]] = []
    for ticket_id in complete_gold_ids:
        for config in sorted(rows_by_ticket_config[ticket_id]):
            matched_records.append(
                slim_record(rows_by_ticket_config[ticket_id][config], gold_index[ticket_id])
            )

    matched_rows_by_config = Counter(str(row["analysis_config"]) for row in matched_records)
    matched_category_counts = Counter(
        str(gold_index[ticket_id].get("topic_group", "")) for ticket_id in complete_gold_ids
    )
    report = {
        "balanced_dataset_path": str(balanced_path),
        "gold_path": str(GOLD_PATH),
        "gold_record_count": len(gold_records),
        "gold_ticket_id_min": min(gold_ids) if gold_ids else None,
        "gold_ticket_id_max": max(gold_ids) if gold_ids else None,
        "analysis_config_count_seen_for_gold_rows": len(all_config_ids),
        "analysis_configs_seen_for_gold_rows": sorted(all_config_ids),
        "gold_tickets_present_any_config": len(rows_by_ticket_config),
        "gold_tickets_present_all_configs": len(complete_gold_ids),
        "matched_gold_ticket_ids": complete_gold_ids,
        "matched_row_count": len(matched_records),
        "matched_rows_by_config": dict(sorted(matched_rows_by_config.items())),
        "matched_gold_category_counts": dict(sorted(matched_category_counts.items())),
        "checks": {
            "expected_overlap_76": len(complete_gold_ids) == EXPECTED_GOLD_OVERLAP,
            "expected_rows_456": len(matched_records) == EXPECTED_MATCHED_ROWS,
            "all_matched_configs_have_76_rows": all(
                count == EXPECTED_GOLD_OVERLAP for count in matched_rows_by_config.values()
            ),
        },
    }

    write_jsonl(matched_records, OUTPUT_DIR / "gold_matched_production_outputs.jsonl")
    (OUTPUT_DIR / "gold_match_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_report_md(report, OUTPUT_DIR / "gold_match_report.md")

    print("Gold resolution subset build complete")
    print(f"Gold records: {len(gold_records)}")
    print(f"Gold tickets present in all configs: {len(complete_gold_ids)}")
    print(f"Matched production rows: {len(matched_records)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
