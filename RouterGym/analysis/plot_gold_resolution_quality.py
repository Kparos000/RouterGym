"""Plot gold-resolution quality results for dissertation reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_outputs" / "gold_resolution_eval"
PLOT_DIR = OUTPUT_DIR / "plots"
DPI = 220

CONFIG_LABELS = {
    "llm_only__base_llm1__mem_rag_bm25": "LLM1",
    "llm_only__base_llm2__mem_rag_bm25": "LLM2",
    "slm_only__base_slm1__mem_rag_bm25": "SLM1",
    "slm_only__base_slm2__mem_rag_bm25": "SLM2",
    "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25": "SLM-dom SLM1->LLM2",
    "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25": "SLM-dom SLM2->LLM2",
}

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def read_quality() -> pd.DataFrame:
    path = OUTPUT_DIR / "gold_resolution_quality_by_config.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing quality table: {path}")
    df = pd.read_csv(path)
    df["label"] = df["analysis_config"].map(CONFIG_LABELS).fillna(df["analysis_config"])
    order = list(CONFIG_LABELS.values())
    df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
    return df.sort_values("label")


def save_all(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=DPI)
    fig.savefig(PLOT_DIR / f"{name}.pdf")
    plt.close(fig)


def bar_plot(df: pd.DataFrame, column: str, name: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df["label"].astype(str), df[column], color=PALETTE[: len(df)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_all(fig, name)


def component_plot(df: pd.DataFrame) -> None:
    components = [
        ("mean_step_coverage_score", "Step coverage"),
        ("mean_acceptance_criteria_alignment_score", "Acceptance"),
        ("mean_escalation_correctness_score", "Escalation"),
        ("mean_policy_grounding_match_score", "Policy"),
    ]
    x = range(len(df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (column, label) in enumerate(components):
        offsets = [value + (idx - 1.5) * width for value in x]
        ax.bar(offsets, df[column], width=width, label=label, color=PALETTE[idx])
    ax.set_title("Gold Resolution Component Scores by Config")
    ax.set_ylabel("Mean score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"].astype(str), rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_all(fig, "gold_component_scores_by_config")


def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    name: str,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, (_, row) in enumerate(df.iterrows()):
        ax.scatter(
            row[x_col],
            row["mean_overall_gold_quality_score"],
            color=PALETTE[idx % len(PALETTE)],
            s=80,
        )
        ax.annotate(
            str(row["label"]),
            (row[x_col], row["mean_overall_gold_quality_score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean gold quality score")
    ax.grid(alpha=0.25)
    save_all(fig, name)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_quality()
    generated = []

    bar_plot(
        df,
        "mean_overall_gold_quality_score",
        "gold_overall_quality_by_config",
        "Gold Resolution Quality by Config",
        "Mean gold quality score",
    )
    generated.append("gold_overall_quality_by_config.png")
    component_plot(df)
    generated.append("gold_component_scores_by_config.png")
    scatter_plot(
        df,
        "avg_cost_usd",
        "gold_quality_vs_cost",
        "Gold Resolution Quality vs Cost",
        "Average cost per ticket (USD)",
    )
    generated.append("gold_quality_vs_cost.png")
    scatter_plot(
        df,
        "avg_latency_ms",
        "gold_quality_vs_latency",
        "Gold Resolution Quality vs Latency",
        "Average latency (ms)",
    )
    generated.append("gold_quality_vs_latency.png")
    bar_plot(
        df,
        "pass_rate_070",
        "gold_quality_pass_rate_070_by_config",
        "Gold Quality Pass Rate >= 0.70",
        "Pass rate",
    )
    generated.append("gold_quality_pass_rate_070_by_config.png")
    bar_plot(
        df,
        "pass_rate_080",
        "gold_quality_pass_rate_080_by_config",
        "Gold Quality Pass Rate >= 0.80",
        "Pass rate",
    )
    generated.append("gold_quality_pass_rate_080_by_config.png")
    if (
        "generated_category_accuracy" in df.columns
        and df["generated_category_accuracy"].notna().any()
    ):
        bar_plot(
            df,
            "generated_category_accuracy",
            "generated_category_accuracy_by_config",
            "Generated Category Accuracy by Config",
            "Accuracy",
        )
        generated.append("generated_category_accuracy_by_config.png")

    print(f"Generated {len(generated)} gold-resolution plots in {PLOT_DIR}")
    for item in generated:
        print(f"  {item}")


if __name__ == "__main__":
    main()
