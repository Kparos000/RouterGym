"""Score matched production outputs against frozen gold resolutions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from RouterGym.evaluation.gold_scoring import score_record_against_gold


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_outputs" / "gold_resolution_eval"
MATCHED_PATH = OUTPUT_DIR / "gold_matched_production_outputs.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing matched gold subset: {path}")
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


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def scalar_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def category_correct(record: dict[str, Any]) -> bool | None:
    generated = record.get("generated_predicted_category")
    gold = record.get("gold_topic_group") or record.get("gold_label")
    if not scalar_has_value(generated) or not scalar_has_value(gold):
        return None
    return str(generated).strip() == str(gold).strip()


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for record in records:
        gold_record = {
            "ticket_index": record.get("gold_ticket_index", record.get("ticket_id")),
            "topic_group": record.get("gold_topic_group", record.get("gold_label", "")),
            "ticket_text": record.get("gold_ticket_text", ""),
            "gold_resolution": record.get("gold_resolution", {}),
            "review_status": record.get("gold_review_status", ""),
        }
        score = score_record_against_gold(record, gold_record).as_dict()
        merged = dict(record)
        merged.update(score)
        generated_correct = category_correct(record)
        merged["generated_category_correct"] = generated_correct
        merged["cost_usd"] = numeric(record.get("total_cost_usd"))
        merged["latency_ms"] = numeric(record.get("latency_ms"))
        if merged["latency_ms"] is None and isinstance(record.get("metrics"), dict):
            merged["latency_ms"] = numeric(record["metrics"].get("latency_ms"))
        scored.append(merged)
    return scored


def flatten_for_csv(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    keep = [
        "ticket_id",
        "gold_ticket_index",
        "analysis_config",
        "router_family",
        "base_model",
        "escalation_model",
        "memory_mode",
        "gold_topic_group",
        "gold_label",
        "classifier_predicted_category",
        "generated_predicted_category",
        "generated_category_correct",
        "gold_match_found",
        "gold_review_status",
        "step_coverage_score",
        "acceptance_criteria_alignment_score",
        "escalation_correctness_score",
        "policy_grounding_match_score",
        "overall_gold_quality_score",
        "cost_usd",
        "latency_ms",
        "generation_valid",
        "raw_response_saved",
        "parse_error",
        "validation_error",
        "escalated",
    ]
    return df[[column for column in keep if column in df.columns]]


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    scored = df.copy()
    quality = pd.to_numeric(scored["overall_gold_quality_score"], errors="coerce")
    scored["pass_070"] = quality >= 0.70
    scored["pass_080"] = quality >= 0.80
    aggregations: dict[str, Any] = {
        "rows": ("overall_gold_quality_score", "size"),
        "matched_gold_tickets": ("gold_ticket_index", "nunique"),
        "mean_overall_gold_quality_score": ("overall_gold_quality_score", "mean"),
        "median_overall_gold_quality_score": ("overall_gold_quality_score", "median"),
        "pass_rate_070": ("pass_070", "mean"),
        "pass_rate_080": ("pass_080", "mean"),
        "mean_step_coverage_score": ("step_coverage_score", "mean"),
        "mean_acceptance_criteria_alignment_score": (
            "acceptance_criteria_alignment_score",
            "mean",
        ),
        "mean_escalation_correctness_score": ("escalation_correctness_score", "mean"),
        "mean_policy_grounding_match_score": ("policy_grounding_match_score", "mean"),
        "avg_cost_usd": ("cost_usd", "mean"),
        "avg_latency_ms": ("latency_ms", "mean"),
    }
    if "generated_category_correct" in scored.columns:
        aggregations["generated_category_accuracy"] = ("generated_category_correct", "mean")
        aggregations["generated_category_available_rate"] = (
            "generated_category_correct",
            lambda s: s.notna().mean(),
        )
    return scored.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()


def write_summary(df: pd.DataFrame, path: Path, source_rows: int) -> None:
    best_quality = df.sort_values("mean_overall_gold_quality_score", ascending=False).iloc[0]
    best_cost = df.sort_values("avg_cost_usd", ascending=True).iloc[0]
    best_latency = df.sort_values("avg_latency_ms", ascending=True).iloc[0]
    lines = [
        "# Gold Resolution Evaluation Summary",
        "",
        f"- Scored rows: {source_rows}",
        f"- Configs: {df['analysis_config'].nunique() if 'analysis_config' in df.columns else 'n/a'}",
        f"- Best mean quality: `{best_quality['analysis_config']}` = {best_quality['mean_overall_gold_quality_score']:.4f}",
        f"- Lowest average cost: `{best_cost['analysis_config']}` = {best_cost['avg_cost_usd']:.6f}",
        f"- Lowest average latency: `{best_latency['analysis_config']}` = {best_latency['avg_latency_ms']:.2f} ms",
        "",
        "This evaluation scores generated resolution outputs against the frozen gold-resolution set.",
        "It is separate from classifier-derived category accuracy.",
        "",
        "## Quality by Config",
        "",
        "| Config | Rows | Mean Quality | Median | Pass >=0.70 | Pass >=0.80 | Step | Acceptance | Escalation | Policy | Avg Cost | Avg Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            f"`{row['analysis_config']}` | "
            f"{int(row['rows'])} | "
            f"{row['mean_overall_gold_quality_score']:.4f} | "
            f"{row['median_overall_gold_quality_score']:.4f} | "
            f"{row['pass_rate_070']:.4f} | "
            f"{row['pass_rate_080']:.4f} | "
            f"{row['mean_step_coverage_score']:.4f} | "
            f"{row['mean_acceptance_criteria_alignment_score']:.4f} | "
            f"{row['mean_escalation_correctness_score']:.4f} | "
            f"{row['mean_policy_grounding_match_score']:.4f} | "
            f"{row['avg_cost_usd']:.6f} | "
            f"{row['avg_latency_ms']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_not_run_report(message: str) -> None:
    (OUTPUT_DIR / "gold_scoring_not_run_report.md").write_text(
        "# Gold Scoring Not Run\n\n" + message.strip() + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        records = read_jsonl(MATCHED_PATH)
        scored = score_records(records)
        scored_df = flatten_for_csv(scored)
        if scored_df.empty or scored_df["overall_gold_quality_score"].isna().all():
            raise RuntimeError("Scorer produced no non-null overall_gold_quality_score values.")
    except Exception as exc:
        write_not_run_report(f"Gold scorer failed with: `{type(exc).__name__}: {exc}`")
        raise

    write_jsonl(scored, OUTPUT_DIR / "gold_resolution_scores.jsonl")
    scored_df.to_csv(OUTPUT_DIR / "gold_resolution_scores_flat.csv", index=False)

    by_config = aggregate(scored_df, ["analysis_config"])
    by_router = aggregate(scored_df, ["router_family"])
    by_model = aggregate(scored_df, ["base_model"])
    by_config.to_csv(OUTPUT_DIR / "gold_resolution_quality_by_config.csv", index=False)
    by_router.to_csv(OUTPUT_DIR / "gold_resolution_quality_by_router_family.csv", index=False)
    by_model.to_csv(OUTPUT_DIR / "gold_resolution_quality_by_model.csv", index=False)

    tradeoff_cols = [
        "analysis_config",
        "rows",
        "mean_overall_gold_quality_score",
        "pass_rate_070",
        "pass_rate_080",
        "avg_cost_usd",
        "avg_latency_ms",
    ]
    if "generated_category_accuracy" in by_config.columns:
        tradeoff_cols.append("generated_category_accuracy")
    by_config[tradeoff_cols].to_csv(
        OUTPUT_DIR / "gold_resolution_quality_vs_cost_latency.csv", index=False
    )
    write_summary(by_config, OUTPUT_DIR / "gold_resolution_eval_summary.md", len(scored))

    config_counts = Counter(str(record.get("analysis_config")) for record in records)
    print("Gold resolution scoring complete")
    print(f"Scored rows: {len(scored)}")
    for config, count in sorted(config_counts.items()):
        print(f"  {config}: {count}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
