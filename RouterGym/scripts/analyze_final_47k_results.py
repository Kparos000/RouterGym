"""Analyze the completed 47,837-ticket production benchmark results.

This script reads the merged per-config JSONL artifacts under
`RouterGym/results/production_runs/openai_compatible`, computes dissertation-
ready summaries, and writes tables, plots, and findings under
`RouterGym/results/final_47k_analysis`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence

from RouterGym.contracts.schema_contract import AgentOutputSchema

APPROVED_CONFIGS: Sequence[str] = (
    "slm_only__base_slm1__mem_rag_bm25",
    "slm_only__base_slm2__mem_rag_bm25",
    "llm_only__base_llm1__mem_rag_bm25",
    "llm_only__base_llm2__mem_rag_bm25",
    "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25",
    "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze final 47k RouterGym production results.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("RouterGym/results/production_runs/openai_compatible"),
        help="Root directory containing one folder per config.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("RouterGym/results/final_47k_summary"),
        help="Existing summary root, used only for context reporting.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("test_logs_runpod_final_47k"),
        help="RunPod log directory, used only for reference reporting.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("RouterGym/results/final_47k_analysis"),
        help="Directory for generated analysis files.",
    )
    return parser


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    weight = position - lower
    return float(lower_value + (upper_value - lower_value) * weight)


def _friendly_label(config_id: str) -> str:
    mapping = {
        "slm_only__base_slm1__mem_rag_bm25": "SLM-only (slm1)",
        "slm_only__base_slm2__mem_rag_bm25": "SLM-only (slm2)",
        "llm_only__base_llm1__mem_rag_bm25": "LLM-only (llm1)",
        "llm_only__base_llm2__mem_rag_bm25": "LLM-only (llm2)",
        "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25": "SLM-dom (slm1->llm2)",
        "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25": "SLM-dom (slm2->llm2)",
    }
    return mapping.get(config_id, config_id)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _load_manifest(results_root: Path, config_id: str) -> Dict[str, Any]:
    manifest_path = results_root / config_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing for config {config_id}: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_merged_paths(manifest: Mapping[str, Any]) -> Dict[str, Path]:
    output_files = manifest.get("output_files", {})
    merged_results_path = Path(str(output_files.get("merged_results_path", "")))
    merged_failures_path = Path(str(output_files.get("merged_failures_path", "")))
    if not merged_results_path.exists():
        raise FileNotFoundError(f"Merged results file missing: {merged_results_path}")
    if not merged_failures_path.exists():
        raise FileNotFoundError(f"Merged failures file missing: {merged_failures_path}")
    return {
        "merged_results_path": merged_results_path,
        "merged_failures_path": merged_failures_path,
    }


def analyze_config(
    *,
    config_id: str,
    manifest: Mapping[str, Any],
    merged_results_path: Path,
    merged_failures_path: Path,
) -> Dict[str, Any]:
    schema = AgentOutputSchema()

    row_count = 0
    success_count = 0
    failure_rows = 0
    schema_valid_count = 0
    classification_comparable_count = 0
    classification_correct_count = 0
    escalation_true_count = 0
    retrieval_score_count = 0
    kb_policy_nonempty_count = 0
    total_kb_policy_refs = 0

    latency_values: List[float] = []
    cost_values: List[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost_usd = 0.0
    retrieval_scores: List[float] = []

    top_level_presence: Counter[str] = Counter()
    metrics_presence: Counter[str] = Counter()
    top_level_types: Dict[str, str] = {}
    metrics_types: Dict[str, str] = {}
    first_row_keys: List[str] = []
    first_metrics_keys: List[str] = []

    with merged_results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_count += 1

            row_keys = list(row.keys())
            if not first_row_keys:
                first_row_keys = sorted(row_keys)
            for key in row_keys:
                top_level_presence[key] += 1
                top_level_types.setdefault(key, type(row[key]).__name__)

            metrics = row.get("metrics", {})
            if isinstance(metrics, dict):
                if not first_metrics_keys:
                    first_metrics_keys = sorted(metrics.keys())
                for key, value in metrics.items():
                    metrics_presence[key] += 1
                    metrics_types.setdefault(key, type(value).__name__)

            success = bool(row.get("success", False))
            if success:
                success_count += 1
            else:
                failure_rows += 1

            valid, _ = schema.validate(dict(row))
            if valid:
                schema_valid_count += 1

            gold_label = str(row.get("gold_label", "") or "").strip()
            predicted_label = str(row.get("topic_group", "") or "").strip()
            if gold_label and predicted_label:
                classification_comparable_count += 1
                if gold_label == predicted_label:
                    classification_correct_count += 1

            if bool(row.get("escalated", False)):
                escalation_true_count += 1

            latency = _safe_float(row.get("metrics", {}).get("latency_ms"))
            if latency is not None:
                latency_values.append(latency)

            total_cost = _safe_float(row.get("total_cost_usd"))
            if total_cost is not None:
                cost_values.append(total_cost)
                total_cost_usd += total_cost

            input_tokens = _safe_int(row.get("total_input_tokens"))
            output_tokens = _safe_int(row.get("total_output_tokens"))
            total_tokens_row = _safe_int(row.get("total_tokens"))
            if input_tokens is not None:
                total_input_tokens += input_tokens
            if output_tokens is not None:
                total_output_tokens += output_tokens
            if total_tokens_row is not None:
                total_tokens += total_tokens_row

            retrieval_score = _safe_float(row.get("retrieval_score"))
            if retrieval_score is not None:
                retrieval_score_count += 1
                retrieval_scores.append(retrieval_score)

            kb_policy_ids = row.get("kb_policy_ids", [])
            if isinstance(kb_policy_ids, list):
                total_kb_policy_refs += len(kb_policy_ids)
                if kb_policy_ids:
                    kb_policy_nonempty_count += 1

    merged_failure_rows = _jsonl_count(merged_failures_path)
    total_tickets_expected = int(manifest.get("total_tickets_expected", row_count))

    classification_accuracy = (
        classification_correct_count / classification_comparable_count
        if classification_comparable_count > 0
        else None
    )
    schema_valid_rate = schema_valid_count / row_count if row_count > 0 else None
    escalation_rate = escalation_true_count / row_count if row_count > 0 else None
    retrieval_score_mean = mean(retrieval_scores) if retrieval_scores else None
    retrieval_score_p95 = _percentile(retrieval_scores, 0.95)

    metrics_summary = {
        "config_identifier": config_id,
        "config_label": _friendly_label(config_id),
        "manifest_run_status": str(manifest.get("run_status", "")),
        "row_count": row_count,
        "completed_tickets": row_count,
        "expected_tickets": total_tickets_expected,
        "success_rows": success_count,
        "failure_rows": failure_rows,
        "merged_failure_rows": merged_failure_rows,
        "failure_rate": (failure_rows / row_count) if row_count > 0 else None,
        "mean_latency_ms": mean(latency_values) if latency_values else None,
        "median_latency_ms": median(latency_values) if latency_values else None,
        "p95_latency_ms": _percentile(latency_values, 0.95),
        "total_cost_usd": total_cost_usd,
        "mean_cost_per_ticket_usd": (total_cost_usd / row_count) if row_count > 0 else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "mean_total_tokens_per_ticket": (total_tokens / row_count) if row_count > 0 else None,
        "schema_validity_rate": schema_valid_rate,
        "classification_accuracy": classification_accuracy,
        "classification_comparable_count": classification_comparable_count,
        "escalation_rate": escalation_rate,
        "retrieval_score_available_rate": (retrieval_score_count / row_count)
        if row_count > 0
        else None,
        "retrieval_score_mean": retrieval_score_mean,
        "retrieval_score_p95": retrieval_score_p95,
        "kb_policy_ref_nonempty_rate": (kb_policy_nonempty_count / row_count)
        if row_count > 0
        else None,
        "avg_kb_policy_refs_per_ticket": (total_kb_policy_refs / row_count)
        if row_count > 0
        else None,
        "groundedness_metric_available": False,
    }

    schema_report = {
        "config_identifier": config_id,
        "merged_results_path": str(merged_results_path),
        "merged_failures_path": str(merged_failures_path),
        "row_count": row_count,
        "top_level_fields": first_row_keys,
        "metrics_fields": first_metrics_keys,
        "top_level_field_presence": dict(sorted(top_level_presence.items())),
        "metrics_field_presence": dict(sorted(metrics_presence.items())),
        "top_level_field_types": dict(sorted(top_level_types.items())),
        "metrics_field_types": dict(sorted(metrics_types.items())),
    }
    return {
        "summary": metrics_summary,
        "schema_report": schema_report,
    }


def _plot_bar(
    *,
    output_path: Path,
    labels: Sequence[str],
    values: Sequence[Optional[float]],
    title: str,
    ylabel: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    filtered = [(label, value) for label, value in zip(labels, values) if value is not None]
    if not filtered:
        return False
    plot_labels = [item[0] for item in filtered]
    plot_values = [float(item[1]) for item in filtered]

    plt.figure(figsize=(11, 5))
    plt.bar(plot_labels, plot_values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def _plot_scatter(
    *,
    output_path: Path,
    xs: Sequence[Optional[float]],
    ys: Sequence[Optional[float]],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    frontier: bool = False,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    filtered = [
        (label, float(x), float(y))
        for label, x, y in zip(labels, xs, ys)
        if x is not None and y is not None
    ]
    if not filtered:
        return False

    plt.figure(figsize=(8, 6))
    for label, x_value, y_value in filtered:
        plt.scatter(x_value, y_value)
        plt.annotate(
            label, (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=8
        )

    if frontier and len(filtered) >= 2:
        frontier_points: List[tuple[str, float, float]] = []
        for label, x_value, y_value in sorted(filtered, key=lambda item: (item[1], -item[2])):
            if not frontier_points or y_value > frontier_points[-1][2]:
                frontier_points.append((label, x_value, y_value))
        if len(frontier_points) >= 2:
            plt.plot(
                [item[1] for item in frontier_points],
                [item[2] for item in frontier_points],
                linestyle="--",
                marker="o",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def write_findings(
    *,
    output_path: Path,
    config_summaries: Sequence[Mapping[str, Any]],
    metric_availability: Mapping[str, bool],
) -> None:
    by_cost = sorted(
        [row for row in config_summaries if row.get("mean_cost_per_ticket_usd") is not None],
        key=lambda row: float(row["mean_cost_per_ticket_usd"]),
    )
    by_latency = sorted(
        [row for row in config_summaries if row.get("mean_latency_ms") is not None],
        key=lambda row: float(row["mean_latency_ms"]),
    )
    by_accuracy = sorted(
        [row for row in config_summaries if row.get("classification_accuracy") is not None],
        key=lambda row: float(row["classification_accuracy"]),
        reverse=True,
    )

    lines: List[str] = [
        "# Initial Findings",
        "",
        "## Scope",
        "",
        f"- Configs analyzed: {len(config_summaries)}",
        f"- Tickets per config expected from manifest: {config_summaries[0]['expected_tickets'] if config_summaries else 'n/a'}",
        "",
        "## Metric Availability",
        "",
    ]
    for metric_name, available in sorted(metric_availability.items()):
        status = "available" if available else "not available from current output fields"
        lines.append(f"- {metric_name}: {status}")

    lines.extend(
        [
            "",
            "## High-Level Observations",
            "",
        ]
    )
    if by_cost:
        cheapest = by_cost[0]
        most_expensive = by_cost[-1]
        lines.append(
            f"- Lowest mean cost per ticket: `{cheapest['config_identifier']}` "
            f"({cheapest['mean_cost_per_ticket_usd']:.6f} USD)."
        )
        lines.append(
            f"- Highest mean cost per ticket: `{most_expensive['config_identifier']}` "
            f"({most_expensive['mean_cost_per_ticket_usd']:.6f} USD)."
        )
    if by_latency:
        fastest = by_latency[0]
        slowest = by_latency[-1]
        lines.append(
            f"- Lowest mean latency: `{fastest['config_identifier']}` "
            f"({fastest['mean_latency_ms']:.2f} ms)."
        )
        lines.append(
            f"- Highest mean latency: `{slowest['config_identifier']}` "
            f"({slowest['mean_latency_ms']:.2f} ms)."
        )
    if by_accuracy:
        best_accuracy = by_accuracy[0]
        worst_accuracy = by_accuracy[-1]
        lines.append(
            f"- Highest classification accuracy against `gold_label`: `{best_accuracy['config_identifier']}` "
            f"({best_accuracy['classification_accuracy']:.4f})."
        )
        lines.append(
            f"- Lowest classification accuracy against `gold_label`: `{worst_accuracy['config_identifier']}` "
            f"({worst_accuracy['classification_accuracy']:.4f})."
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `classification_accuracy` here is derived from `topic_group` versus `gold_label`, so it is a routing/classifier signal, not a direct answer-quality score.",
            "- `schema_validity_rate` is computed from the stored row payloads against the current `AgentOutputSchema`.",
            "- A direct groundedness/faithfulness metric is not present in the raw production outputs, so no groundedness score is reported here.",
            "- Retrieval-related analysis is limited to `retrieval_score` and KB policy reference counts because no explicit groundedness score was emitted in these rows.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    merged_output_rows: List[Dict[str, Any]] = []
    config_summaries: List[Dict[str, Any]] = []
    schema_reports: Dict[str, Any] = {
        "results_root": str(args.results_root),
        "summary_root": str(args.summary_root),
        "logs_root": str(args.logs_root),
        "configs": {},
    }

    for config_id in APPROVED_CONFIGS:
        manifest = _load_manifest(args.results_root, config_id)
        paths = _resolve_merged_paths(manifest)
        analysis = analyze_config(
            config_id=config_id,
            manifest=manifest,
            merged_results_path=paths["merged_results_path"],
            merged_failures_path=paths["merged_failures_path"],
        )
        merged_output_rows.append(
            {
                "config_identifier": config_id,
                "config_label": _friendly_label(config_id),
                "manifest_path": str(args.results_root / config_id / "manifest.json"),
                "merged_results_path": str(paths["merged_results_path"]),
                "merged_failures_path": str(paths["merged_failures_path"]),
            }
        )
        config_summaries.append(dict(analysis["summary"]))
        schema_reports["configs"][config_id] = analysis["schema_report"]

    metric_availability = {
        "row count": True,
        "completed tickets": True,
        "failure rows": True,
        "mean latency": any(row.get("mean_latency_ms") is not None for row in config_summaries),
        "median latency": any(row.get("median_latency_ms") is not None for row in config_summaries),
        "p95 latency": any(row.get("p95_latency_ms") is not None for row in config_summaries),
        "total cost": any(row.get("total_cost_usd") is not None for row in config_summaries),
        "mean cost per ticket": any(
            row.get("mean_cost_per_ticket_usd") is not None for row in config_summaries
        ),
        "input tokens": any(row.get("total_input_tokens") is not None for row in config_summaries),
        "output tokens": any(
            row.get("total_output_tokens") is not None for row in config_summaries
        ),
        "total tokens": any(row.get("total_tokens") is not None for row in config_summaries),
        "schema validity rate": any(
            row.get("schema_validity_rate") is not None for row in config_summaries
        ),
        "classification accuracy": any(
            row.get("classification_accuracy") is not None for row in config_summaries
        ),
        "escalation rate": any(row.get("escalation_rate") is not None for row in config_summaries),
        "retrieval score": any(
            row.get("retrieval_score_mean") is not None for row in config_summaries
        ),
        "groundedness metric": False,
    }
    schema_reports["metric_availability"] = metric_availability

    overall_summary_fields = [
        "config_identifier",
        "config_label",
        "manifest_run_status",
        "row_count",
        "completed_tickets",
        "expected_tickets",
        "success_rows",
        "failure_rows",
        "merged_failure_rows",
        "failure_rate",
        "schema_validity_rate",
        "classification_accuracy",
        "classification_comparable_count",
        "escalation_rate",
        "retrieval_score_mean",
        "kb_policy_ref_nonempty_rate",
    ]
    cost_latency_fields = [
        "config_identifier",
        "config_label",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "total_cost_usd",
        "mean_cost_per_ticket_usd",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "mean_total_tokens_per_ticket",
    ]
    quality_fields = [
        "config_identifier",
        "config_label",
        "schema_validity_rate",
        "classification_accuracy",
        "classification_comparable_count",
        "groundedness_metric_available",
    ]
    routing_fields = [
        "config_identifier",
        "config_label",
        "success_rows",
        "failure_rows",
        "escalation_rate",
        "retrieval_score_available_rate",
        "retrieval_score_mean",
        "retrieval_score_p95",
        "kb_policy_ref_nonempty_rate",
        "avg_kb_policy_refs_per_ticket",
    ]

    _write_csv(
        output_root / "merged_output_index.csv",
        merged_output_rows,
        list(merged_output_rows[0].keys()),
    )
    _write_json(output_root / "schema_report.json", schema_reports)
    _write_csv(output_root / "overall_config_summary.csv", config_summaries, overall_summary_fields)
    _write_csv(output_root / "cost_latency_summary.csv", config_summaries, cost_latency_fields)
    _write_csv(output_root / "quality_summary.csv", config_summaries, quality_fields)
    _write_csv(output_root / "routing_summary.csv", config_summaries, routing_fields)
    write_findings(
        output_path=output_root / "initial_findings.md",
        config_summaries=config_summaries,
        metric_availability=metric_availability,
    )

    labels = [row["config_label"] for row in config_summaries]
    _plot_bar(
        output_path=output_root / "plots" / "cost_per_ticket_by_config.png",
        labels=labels,
        values=[row.get("mean_cost_per_ticket_usd") for row in config_summaries],
        title="Mean Cost Per Ticket by Config",
        ylabel="USD per ticket",
    )
    _plot_bar(
        output_path=output_root / "plots" / "latency_by_config.png",
        labels=labels,
        values=[row.get("mean_latency_ms") for row in config_summaries],
        title="Mean Latency by Config",
        ylabel="Latency (ms)",
    )
    _plot_bar(
        output_path=output_root / "plots" / "total_cost_by_config.png",
        labels=labels,
        values=[row.get("total_cost_usd") for row in config_summaries],
        title="Total Cost by Config",
        ylabel="Total cost (USD)",
    )
    _plot_bar(
        output_path=output_root / "plots" / "schema_validity_by_config.png",
        labels=labels,
        values=[row.get("schema_validity_rate") for row in config_summaries],
        title="Schema Validity Rate by Config",
        ylabel="Rate",
    )
    _plot_bar(
        output_path=output_root / "plots" / "classification_accuracy_by_config.png",
        labels=labels,
        values=[row.get("classification_accuracy") for row in config_summaries],
        title="Classification Accuracy by Config",
        ylabel="Accuracy",
    )
    _plot_scatter(
        output_path=output_root / "plots" / "cost_vs_quality_scatter.png",
        xs=[row.get("mean_cost_per_ticket_usd") for row in config_summaries],
        ys=[row.get("classification_accuracy") for row in config_summaries],
        labels=labels,
        title="Cost vs Quality Proxy",
        xlabel="Mean cost per ticket (USD)",
        ylabel="Classification accuracy",
    )
    _plot_scatter(
        output_path=output_root / "plots" / "pareto_frontier_cost_vs_quality.png",
        xs=[row.get("mean_cost_per_ticket_usd") for row in config_summaries],
        ys=[row.get("classification_accuracy") for row in config_summaries],
        labels=labels,
        title="Pareto Frontier: Cost vs Quality Proxy",
        xlabel="Mean cost per ticket (USD)",
        ylabel="Classification accuracy",
        frontier=True,
    )

    summary_payload = {
        "output_root": str(output_root),
        "config_count": len(config_summaries),
        "created_files": sorted(
            str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()
        ),
        "metric_availability": metric_availability,
    }
    _write_json(output_root / "analysis_manifest.json", summary_payload)
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
