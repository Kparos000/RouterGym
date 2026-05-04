"""Create dissertation-ready analysis tables for the balanced 60k benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
except ImportError:  # pragma: no cover - environment dependent
    accuracy_score = None
    f1_score = None
    precision_score = None
    recall_score = None


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_input"
OUTPUT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_outputs"
PROJECTED_TICKET_COUNT = 47_000


def find_dataset() -> Path:
    candidates = sorted(INPUT_DIR.rglob("balanced_10k_all_configs.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Could not find balanced_10k_all_configs.jsonl under {INPUT_DIR}")
    return candidates[0]


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def scalar_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def has_error(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return bool(value)


def safe_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    return series.fillna(default).astype(bool)


def parse_config(config: str) -> dict[str, str | None]:
    tokens = config.split("__")
    router_family = tokens[0] if tokens else None
    base_model = None
    escalation_model = None
    memory_mode = None
    for token in tokens:
        if token.startswith("base_"):
            base_model = token.removeprefix("base_")
        elif token.startswith("esc_"):
            escalation_model = token.removeprefix("esc_")
        elif token.startswith("mem_"):
            memory_mode = token.removeprefix("mem_")
    return {
        "router_family": router_family,
        "base_model": base_model,
        "escalation_model": escalation_model,
        "memory_mode": memory_mode,
    }


def resolution_step_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return 1
        return len(parsed) if isinstance(parsed, list) else 1
    return 0


def get_latency_ms(df: pd.DataFrame) -> pd.Series | None:
    for column in ["latency_ms", "duration_ms", "elapsed_ms"]:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    if "metrics" in df.columns:
        return df["metrics"].map(
            lambda value: value.get("latency_ms") if isinstance(value, dict) else None
        )
    return None


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in [
        "parse_error",
        "validation_error",
        "final_answer",
        "predicted_category",
        "gold_label",
    ]:
        if column not in df.columns:
            df[column] = None

    parsed = df["analysis_config"].astype(str).map(parse_config).apply(pd.Series)
    for column in ["router_family", "base_model", "escalation_model", "memory_mode"]:
        df[column] = parsed[column]
    df["has_parse_error"] = df["parse_error"].map(has_error)
    df["has_validation_error"] = df["validation_error"].map(has_error)
    df["resolution_step_count"] = df.get("resolution_steps", pd.Series(index=df.index)).map(
        resolution_step_count
    )
    df["final_answer_length"] = df["final_answer"].fillna("").astype(str).str.len()
    df["missing_predicted_category"] = ~df["predicted_category"].map(scalar_has_value)
    df["llm_unavailable"] = df.astype(str).apply(
        lambda row: row.str.contains(
            "llm_unavailable|LLM unavailable", case=False, regex=True
        ).any(),
        axis=1,
    )
    if "success" in df.columns:
        success = df["success"].fillna(False).astype(bool)
    else:
        success = pd.Series(True, index=df.index)
    if "generation_valid" in df.columns:
        generation_valid = df["generation_valid"].fillna(False).astype(bool)
    else:
        generation_valid = pd.Series(True, index=df.index)
    df["analysis_usable"] = (
        success
        & generation_valid
        & ~df["has_parse_error"]
        & ~df["has_validation_error"]
        & ~df["missing_predicted_category"]
        & (df["final_answer_length"] > 0)
    )
    df["classification_correct"] = None
    comparable = df["gold_label"].map(scalar_has_value) & df["predicted_category"].map(
        scalar_has_value
    )
    df.loc[comparable, "classification_correct"] = df.loc[comparable, "gold_label"].astype(
        str
    ) == df.loc[comparable, "predicted_category"].astype(str)
    latency = get_latency_ms(df)
    if latency is not None:
        df["latency_ms"] = pd.to_numeric(latency, errors="coerce")
    return df


def available_numeric_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [
        column
        for column in candidates
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()
    ]


def summarize_bool(
    group: pd.core.groupby.DataFrameGroupBy, column: str, prefix: str
) -> pd.DataFrame:
    return group[column].agg(**{f"{prefix}_count": "sum", f"{prefix}_rate": "mean"})


def classification_metrics(group_df: pd.DataFrame) -> dict[str, Any]:
    data = group_df.dropna(subset=["gold_label", "predicted_category"])
    if data.empty:
        return {
            "support": 0,
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_f1": None,
        }
    y_true = data["gold_label"].astype(str)
    y_pred = data["predicted_category"].astype(str)
    if accuracy_score and f1_score and precision_score and recall_score:
        return {
            "support": len(data),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    labels = sorted(set(y_true) | set(y_pred))
    supports = Counter(y_true)
    f1_values = []
    weighted_values = []
    precision_values = []
    recall_values = []
    for label in labels:
        true_positive = ((y_true == label) & (y_pred == label)).sum()
        false_positive = ((y_true != label) & (y_pred == label)).sum()
        false_negative = ((y_true == label) & (y_pred != label)).sum()
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        support = supports[label]
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_values.append(f1 * support)
    correct = (y_true == y_pred).mean()
    total_support = sum(supports.values())
    return {
        "support": len(data),
        "accuracy": correct,
        "macro_precision": sum(precision_values) / len(precision_values)
        if precision_values
        else None,
        "macro_recall": sum(recall_values) / len(recall_values) if recall_values else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "weighted_f1": sum(weighted_values) / total_support if total_support else None,
    }


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT_DIR / name, index=False)


def make_outputs_readme(generated_csvs: list[str], plot_note: str) -> None:
    lines = [
        "# RouterGym Analysis Outputs",
        "",
        "These files are generated from the local balanced 60k BM25-RAG inference dataset.",
        "The raw JSONL input is intentionally excluded from git because it is large.",
        "`balanced_60k_all_configs_flat.csv` is also a large local derived extract and is ignored.",
        "",
        "## CSV and JSON Artifacts",
    ]
    for csv_name in generated_csvs:
        lines.append(f"- `{csv_name}`: generated analysis table.")
    lines.extend(
        [
            "- `available_columns.txt`: discovered columns in the analysis dataset.",
            "- `sample_rows.json`: small sample of raw JSONL records for schema inspection.",
            "- `dataset_integrity_report.json`: row-count, config-count, and ticket coverage audit.",
            "- `metric_column_detection_report.json`: detected candidate fields by metric family.",
            "",
            "## Plots",
            plot_note,
        ]
    )
    (OUTPUT_DIR / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = find_dataset()
    df = ensure_derived_columns(load_jsonl(dataset_path))
    generated: list[str] = []

    flat_columns = [
        column
        for column in df.columns
        if not df[column].map(lambda value: isinstance(value, (dict, list))).any()
    ]
    write_csv(df[flat_columns], "balanced_60k_all_configs_flat.csv")
    generated.append("balanced_60k_all_configs_flat.csv")

    by_config = df.groupby("analysis_config", dropna=False)
    integrity = by_config.agg(
        rows=("ticket_id", "size"),
        unique_ticket_ids=("ticket_id", "nunique"),
        min_ticket_id=("ticket_id", lambda s: pd.to_numeric(s, errors="coerce").min()),
        max_ticket_id=("ticket_id", lambda s: pd.to_numeric(s, errors="coerce").max()),
        missing_predicted_category_rate=("missing_predicted_category", "mean"),
        parse_error_rate=("has_parse_error", "mean"),
        validation_error_rate=("has_validation_error", "mean"),
        analysis_usable_rate=("analysis_usable", "mean"),
    ).reset_index()
    write_csv(integrity, "dataset_integrity_by_config.csv")
    generated.append("dataset_integrity_by_config.csv")

    cost_cols = available_numeric_columns(
        df,
        [
            "total_cost_usd",
            "slm_cost_usd",
            "llm_cost_usd",
            "total_input_cost_usd",
            "total_output_cost_usd",
        ],
    )
    token_cols = available_numeric_columns(
        df, ["total_input_tokens", "total_output_tokens", "total_tokens", "max_output_tokens"]
    )
    agg_spec: dict[str, Any] = {
        "rows": ("ticket_id", "size"),
        "classification_accuracy": ("classification_correct", "mean"),
        "success_rate": ("success", "mean")
        if "success" in df.columns
        else ("analysis_usable", "mean"),
        "generation_valid_rate": ("generation_valid", "mean")
        if "generation_valid" in df.columns
        else ("analysis_usable", "mean"),
        "analysis_usable_rate": ("analysis_usable", "mean"),
        "parse_error_rate": ("has_parse_error", "mean"),
        "validation_error_rate": ("has_validation_error", "mean"),
        "empty_resolution_steps_rate": ("resolution_step_count", lambda s: (s == 0).mean()),
        "bad_row_count": ("analysis_usable", lambda s: (~s).sum()),
        "llm_unavailable_rate": ("llm_unavailable", "mean"),
        "avg_resolution_steps": ("resolution_step_count", "mean"),
        "avg_final_answer_length": ("final_answer_length", "mean"),
    }
    if "latency_ms" in df.columns:
        agg_spec.update(
            {
                "mean_latency_ms": ("latency_ms", "mean"),
                "median_latency_ms": ("latency_ms", "median"),
                "p95_latency_ms": ("latency_ms", lambda s: s.quantile(0.95)),
            }
        )
    for column in cost_cols + token_cols:
        agg_spec[f"avg_{column}"] = (column, "mean")
        agg_spec[f"sum_{column}"] = (column, "sum")
    summary_by_config = by_config.agg(**agg_spec).reset_index()
    write_csv(summary_by_config, "summary_by_config.csv")
    generated.append("summary_by_config.csv")

    summary_by_router_family = (
        df.groupby("router_family", dropna=False).agg(**agg_spec).reset_index()
    )
    write_csv(summary_by_router_family, "summary_by_router_family.csv")
    generated.append("summary_by_router_family.csv")

    summary_by_model = df.groupby("base_model", dropna=False).agg(**agg_spec).reset_index()
    write_csv(summary_by_model, "summary_by_model.csv")
    generated.append("summary_by_model.csv")

    metrics = [
        {"analysis_config": config, **classification_metrics(group)} for config, group in by_config
    ]
    metrics_df = pd.DataFrame(metrics)
    write_csv(metrics_df, "classification_metrics_by_config.csv")
    generated.append("classification_metrics_by_config.csv")

    category_accuracy = (
        df.dropna(subset=["gold_label", "classification_correct"])
        .groupby(["analysis_config", "gold_label"], dropna=False)
        .agg(rows=("ticket_id", "size"), accuracy=("classification_correct", "mean"))
        .reset_index()
    )
    write_csv(category_accuracy, "classification_accuracy_by_category.csv")
    generated.append("classification_accuracy_by_category.csv")

    confusion = (
        df.dropna(subset=["gold_label", "predicted_category"])
        .groupby(["analysis_config", "gold_label", "predicted_category"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    write_csv(confusion, "confusion_matrix_by_config.csv")
    generated.append("confusion_matrix_by_config.csv")

    misclassified = confusion[
        confusion["gold_label"].astype(str) != confusion["predicted_category"].astype(str)
    ]
    top_mis = misclassified.sort_values("count", ascending=False).head(50)
    write_csv(top_mis, "top_misclassification_pairs.csv")
    generated.append("top_misclassification_pairs.csv")

    generation_quality = summary_by_config[
        [
            "analysis_config",
            "rows",
            "success_rate",
            "generation_valid_rate",
            "analysis_usable_rate",
            "parse_error_rate",
            "validation_error_rate",
            "empty_resolution_steps_rate",
            "llm_unavailable_rate",
            "avg_resolution_steps",
            "avg_final_answer_length",
        ]
    ]
    write_csv(generation_quality, "generation_quality_by_config.csv")
    generated.append("generation_quality_by_config.csv")

    reliability = integrity[
        [
            "analysis_config",
            "rows",
            "missing_predicted_category_rate",
            "parse_error_rate",
            "validation_error_rate",
            "analysis_usable_rate",
        ]
    ]
    write_csv(reliability, "reliability_by_config.csv")
    generated.append("reliability_by_config.csv")

    bad_rows = df[
        (~df["analysis_usable"])
        | df["has_parse_error"]
        | df["has_validation_error"]
        | df["missing_predicted_category"]
        | df["llm_unavailable"]
    ].head(200)
    write_csv(bad_rows[flat_columns], "bad_rows_sample.csv")
    generated.append("bad_rows_sample.csv")

    write_csv(df[df["has_parse_error"]].head(200)[flat_columns], "parse_error_sample.csv")
    generated.append("parse_error_sample.csv")

    token_cost_cols = ["analysis_config", "rows"] + [
        column
        for column in summary_by_config.columns
        if column.startswith(
            ("avg_total", "sum_total", "avg_slm", "sum_slm", "avg_llm", "sum_llm", "avg_max")
        )
    ]
    token_cost_summary = summary_by_config[token_cost_cols]
    write_csv(token_cost_summary, "token_cost_summary_by_config.csv")
    generated.append("token_cost_summary_by_config.csv")

    projected = summary_by_config[["analysis_config", "rows"]].copy()
    if "avg_total_cost_usd" in summary_by_config.columns:
        projected["avg_cost_per_ticket_usd"] = summary_by_config["avg_total_cost_usd"]
        projected["observed_10k_cost_usd"] = summary_by_config["sum_total_cost_usd"]
        projected["projected_47k_cost_usd"] = (
            projected["avg_cost_per_ticket_usd"] * PROJECTED_TICKET_COUNT
        )
    write_csv(projected, "projected_47k_cost_by_config.csv")
    generated.append("projected_47k_cost_by_config.csv")

    savings_rows = []
    if "avg_total_cost_usd" in summary_by_config.columns:
        baselines = summary_by_config[
            summary_by_config["analysis_config"].str.startswith("llm_only")
        ]
        baseline = (
            baselines.sort_values("avg_total_cost_usd").iloc[0] if not baselines.empty else None
        )
        if baseline is not None:
            for _, row in summary_by_config.iterrows():
                savings_rows.append(
                    {
                        "baseline_config": baseline["analysis_config"],
                        "comparison_config": row["analysis_config"],
                        "baseline_avg_cost_per_ticket_usd": baseline["avg_total_cost_usd"],
                        "comparison_avg_cost_per_ticket_usd": row["avg_total_cost_usd"],
                        "absolute_savings_per_ticket_usd": baseline["avg_total_cost_usd"]
                        - row["avg_total_cost_usd"],
                        "relative_savings_rate": 1
                        - (row["avg_total_cost_usd"] / baseline["avg_total_cost_usd"]),
                        "projected_47k_savings_usd": (
                            baseline["avg_total_cost_usd"] - row["avg_total_cost_usd"]
                        )
                        * PROJECTED_TICKET_COUNT,
                    }
                )
    write_csv(pd.DataFrame(savings_rows), "cost_savings_vs_llm_baseline.csv")
    generated.append("cost_savings_vs_llm_baseline.csv")

    latency_cols = [
        column
        for column in [
            "analysis_config",
            "rows",
            "mean_latency_ms",
            "median_latency_ms",
            "p95_latency_ms",
        ]
        if column in summary_by_config.columns
    ]
    write_csv(summary_by_config[latency_cols], "latency_summary_by_config.csv")
    generated.append("latency_summary_by_config.csv")

    if "mean_latency_ms" in summary_by_config.columns:
        throughput = summary_by_config[["analysis_config", "mean_latency_ms"]].copy()
        throughput["tickets_per_minute_per_worker"] = 60_000 / throughput["mean_latency_ms"]
        write_csv(throughput, "throughput_summary_by_config.csv")
        generated.append("throughput_summary_by_config.csv")

    if "escalated" in df.columns:
        escalation_summary = by_config.agg(
            rows=("ticket_id", "size"),
            escalation_rate=("escalated", "mean"),
            escalated_quality=(
                "classification_correct",
                lambda s: s[df.loc[s.index, "escalated"].fillna(False).astype(bool)].mean(),
            ),
            non_escalated_quality=(
                "classification_correct",
                lambda s: s[~df.loc[s.index, "escalated"].fillna(False).astype(bool)].mean(),
            ),
        ).reset_index()
        write_csv(escalation_summary, "routing_escalation_summary.csv")
        generated.append("routing_escalation_summary.csv")
        if "gold_label" in df.columns:
            escalation_by_category = (
                df.groupby(["analysis_config", "gold_label"], dropna=False)
                .agg(rows=("ticket_id", "size"), escalation_rate=("escalated", "mean"))
                .reset_index()
            )
            write_csv(escalation_by_category, "escalation_by_category.csv")
            generated.append("escalation_by_category.csv")

    memory_modes = sorted(df["memory_mode"].dropna().astype(str).unique())
    if len(memory_modes) > 1:
        memory_summary = df.groupby("memory_mode", dropna=False).agg(**agg_spec).reset_index()
        write_csv(memory_summary, "memory_mode_summary.csv")
        generated.append("memory_mode_summary.csv")
    else:
        print(
            "Memory note: The balanced production-scale result fixes memory/context to BM25 RAG; "
            "memory-mode comparison is not claimed from this dataset."
        )

    findings = []
    row_count_passed = len(df) == 60_000
    findings.append({"finding": "row_count_passed", "value": row_count_passed})
    if "avg_total_cost_usd" in summary_by_config.columns:
        cheapest = summary_by_config.sort_values("avg_total_cost_usd").iloc[0]
        findings.append(
            {"finding": "lowest_average_cost_config", "value": cheapest["analysis_config"]}
        )
    if "classification_accuracy" in summary_by_config.columns:
        best_accuracy = summary_by_config.sort_values(
            "classification_accuracy", ascending=False
        ).iloc[0]
        findings.append(
            {
                "finding": "highest_classification_accuracy_config",
                "value": best_accuracy["analysis_config"],
            }
        )
    findings.append({"finding": "memory_mode_scope", "value": ", ".join(memory_modes) or "unknown"})
    write_csv(pd.DataFrame(findings), "dissertation_key_findings.csv")
    generated.append("dissertation_key_findings.csv")

    make_outputs_readme(
        generated, "`plots/` contains PNG figures generated by `plot_balanced_60k.py`."
    )

    print("Balanced 60k analysis")
    print(f"Dataset: {dataset_path}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Row-count check: {'PASS' if row_count_passed else 'FAIL'}")
    print(f"Configs: {df['analysis_config'].nunique()}")
    print(f"Memory modes: {memory_modes}")
    print(f"Generated CSV files: {len(generated)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
