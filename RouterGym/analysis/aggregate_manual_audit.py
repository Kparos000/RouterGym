"""Aggregate completed blinded manual audit scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_outputs" / "manual_audit"
GOLD_QUALITY_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_outputs" / "gold_resolution_eval"
DPI = 220
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

CONFIG_LABELS = {
    "llm_only__base_llm1__mem_rag_bm25": "LLM1",
    "llm_only__base_llm2__mem_rag_bm25": "LLM2",
    "slm_only__base_slm1__mem_rag_bm25": "SLM1",
    "slm_only__base_slm2__mem_rag_bm25": "SLM2",
    "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25": "SLM-dom SLM1->LLM2",
    "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25": "SLM-dom SLM2->LLM2",
}

SCORE_COLUMNS = [
    "category_understanding_manual",
    "answer_actionable_manual",
    "answer_complete_manual",
    "resolution_steps_correct_manual",
    "escalation_appropriate_manual",
    "policy_grounded_manual",
    "overall_manual_quality",
]
COMPONENT_SCORE_COLUMNS = SCORE_COLUMNS[:-1]


def resolve_path(path_arg: str | None, default_name: str) -> Path:
    if path_arg:
        path = Path(path_arg)
        return path if path.is_absolute() else REPO_ROOT / path
    return AUDIT_DIR / default_name


def read_inputs(audit_file: Path, key_file: Path) -> pd.DataFrame:
    blinded_path = audit_file
    key_path = key_file
    if not blinded_path.exists() or not key_path.exists():
        raise FileNotFoundError(
            f"Manual audit blinded/key CSV files are required: {blinded_path}, {key_path}"
        )
    blinded = pd.read_csv(blinded_path)
    key = pd.read_csv(key_path)
    merged = blinded.merge(key, on=["audit_id", "anonymous_system_id"], how="inner")
    if len(merged) != len(blinded):
        raise ValueError(
            f"Audit/key join produced {len(merged)} rows from {len(blinded)} audit rows. "
            "Check audit_id and anonymous_system_id."
        )
    validate_scores(merged)
    return merged


def validate_scores(merged: pd.DataFrame) -> None:
    missing = [column for column in SCORE_COLUMNS if column not in merged.columns]
    if missing:
        raise ValueError(f"Missing required manual score columns: {missing}")
    for column in SCORE_COLUMNS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    if merged["overall_manual_quality"].isna().all():
        raise ValueError(
            "No completed overall_manual_quality scores found. Fill the blinded manual audit CSV first."
        )
    incomplete = [column for column in SCORE_COLUMNS if merged[column].isna().any()]
    if incomplete:
        raise ValueError(
            "Manual audit scoring is incomplete. Fill every score column before aggregation: "
            f"{incomplete}"
        )
    for column in COMPONENT_SCORE_COLUMNS:
        invalid = merged[column].dropna()
        invalid = invalid[~invalid.isin([0, 1, 2])]
        if not invalid.empty:
            raise ValueError(f"{column} contains values outside the allowed 0, 1, 2 range.")
    overall = merged["overall_manual_quality"].dropna()
    invalid_overall = overall[(overall < 0) | (overall > 10) | (overall % 1 != 0)]
    if not invalid_overall.empty:
        raise ValueError("overall_manual_quality must contain whole numbers from 0 to 10.")


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg_spec: dict[str, Any] = {
        "rows": ("overall_manual_quality", "count"),
        "mean_overall_manual_quality": ("overall_manual_quality", "mean"),
        "median_overall_manual_quality": ("overall_manual_quality", "median"),
        "manual_quality_pass_rate_7": ("overall_manual_quality", lambda s: (s >= 7).mean()),
        "avg_cost_usd": ("total_cost_usd", "mean"),
        "avg_latency_ms": ("latency_ms", "mean"),
    }
    for column in SCORE_COLUMNS[:-1]:
        agg_spec[f"mean_{column}"] = (column, "mean")
    return df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


def save_bar(df: pd.DataFrame, filename: str) -> None:
    plot_df = df.copy()
    plot_df["label"] = (
        plot_df["analysis_config"].map(CONFIG_LABELS).fillna(plot_df["analysis_config"])
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(plot_df["label"], plot_df["mean_overall_manual_quality"], color=PALETTE[: len(plot_df)])
    ax.set_title("Manual Quality by Config")
    ax.set_ylabel("Mean manual quality (0-10)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / filename, dpi=DPI)
    fig.savefig(AUDIT_DIR / filename.replace(".png", ".pdf"))
    plt.close(fig)


def save_component_bar(df: pd.DataFrame, filename: str) -> None:
    plot_df = df.copy()
    plot_df["label"] = (
        plot_df["analysis_config"].map(CONFIG_LABELS).fillna(plot_df["analysis_config"])
    )
    component_cols = [f"mean_{column}" for column in COMPONENT_SCORE_COLUMNS]
    labels = [
        "Category",
        "Actionable",
        "Complete",
        "Steps",
        "Escalation",
        "Policy",
    ]
    x_positions = range(len(plot_df))
    width = 0.12
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, column in enumerate(component_cols):
        offsets = [pos + (idx - 2.5) * width for pos in x_positions]
        ax.bar(
            offsets,
            plot_df[column],
            width=width,
            label=labels[idx],
            color=PALETTE[idx % len(PALETTE)],
        )
    ax.set_title("Manual Quality Components by Config")
    ax.set_ylabel("Mean component score (0-2)")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(plot_df["label"], rotation=20, ha="right")
    ax.set_ylim(0, 2)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / filename, dpi=DPI)
    fig.savefig(AUDIT_DIR / filename.replace(".png", ".pdf"))
    plt.close(fig)


def save_scatter(df: pd.DataFrame, x_col: str, filename: str, xlabel: str) -> None:
    plot_df = df.copy()
    plot_df["label"] = (
        plot_df["analysis_config"].map(CONFIG_LABELS).fillna(plot_df["analysis_config"])
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.scatter(row[x_col], row["mean_overall_manual_quality"], color=PALETTE[idx], s=80)
        ax.annotate(
            str(row["label"]),
            (row[x_col], row["mean_overall_manual_quality"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title(filename.replace("_", " ").replace(".png", "").title())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean manual quality (0-10)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / filename, dpi=DPI)
    fig.savefig(AUDIT_DIR / filename.replace(".png", ".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-file",
        help="Path to completed blinded manual audit CSV. Defaults to manual_audit_blinded.csv.",
    )
    parser.add_argument(
        "--key-file",
        help="Path to matching manual audit key CSV. Defaults to manual_audit_key.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_file = resolve_path(args.audit_file, "manual_audit_blinded.csv")
    key_file = resolve_path(args.key_file, "manual_audit_key.csv")
    df = read_inputs(audit_file, key_file)
    by_config = aggregate(df, ["analysis_config"])
    by_router = aggregate(df, ["router_family"])
    by_model = aggregate(df, ["base_model"])
    by_config.to_csv(AUDIT_DIR / "manual_quality_by_config.csv", index=False)
    by_router.to_csv(AUDIT_DIR / "manual_quality_by_router_family.csv", index=False)
    by_model.to_csv(AUDIT_DIR / "manual_quality_by_model.csv", index=False)

    tradeoff_cols = [
        "analysis_config",
        "rows",
        "mean_overall_manual_quality",
        "manual_quality_pass_rate_7",
        "avg_cost_usd",
        "avg_latency_ms",
    ]
    by_config[tradeoff_cols].to_csv(AUDIT_DIR / "manual_quality_vs_cost_latency.csv", index=False)
    component_cols = [
        "analysis_config",
        "rows",
        *[f"mean_{column}" for column in COMPONENT_SCORE_COLUMNS],
    ]
    by_config[component_cols].to_csv(
        AUDIT_DIR / "manual_quality_component_summary_by_config.csv", index=False
    )
    save_bar(by_config, "manual_quality_by_config.png")
    save_component_bar(by_config, "manual_quality_components_by_config.png")
    save_scatter(by_config, "avg_cost_usd", "manual_quality_vs_cost.png", "Average cost (USD)")
    save_scatter(
        by_config, "avg_latency_ms", "manual_quality_vs_latency.png", "Average latency (ms)"
    )

    print(f"Aggregated {len(df)} completed manual audit rows")
    print(f"Audit file: {audit_file}")
    print(f"Key file: {key_file}")
    print(f"Outputs written to: {AUDIT_DIR}")
    print(f"Gold quality outputs remain in: {GOLD_QUALITY_DIR}")


if __name__ == "__main__":
    main()
