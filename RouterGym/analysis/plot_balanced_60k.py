"""Generate dissertation-ready plots from balanced 60k analysis outputs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "RouterGym" / "results" / "analysis_outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def read_csv(name: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"Skipping {name}: file not found")
        return None
    if path.stat().st_size == 0:
        print(f"Skipping {name}: file is empty")
        return None
    return pd.read_csv(path, low_memory=False)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def save_bar(
    df: pd.DataFrame | None,
    x_col: str,
    y_col: str,
    filename: str,
    title: str,
    ylabel: str,
) -> bool:
    if df is None:
        return False
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Skipping {filename}: missing {x_col} or {y_col}")
        return False
    plot_df = df[[x_col, y_col]].dropna()
    if plot_df.empty:
        print(f"Skipping {filename}: no data")
        return False
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=200)
    plt.close(fig)
    return True


def save_scatter(
    df: pd.DataFrame | None,
    x_col: str,
    y_col: str,
    label_col: str,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> bool:
    if df is None:
        return False
    needed = [x_col, y_col, label_col]
    if any(column not in df.columns for column in needed):
        print(f"Skipping {filename}: missing one of {needed}")
        return False
    plot_df = df[needed].dropna()
    if plot_df.empty:
        print(f"Skipping {filename}: no data")
        return False
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(plot_df[x_col], plot_df[y_col])
    for _, row in plot_df.iterrows():
        ax.annotate(str(row[label_col]), (row[x_col], row[y_col]), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=200)
    plt.close(fig)
    return True


def save_distribution(
    flat: pd.DataFrame | None,
    value_col: str,
    group_col: str,
    filename: str,
    title: str,
    xlabel: str,
) -> bool:
    if flat is None:
        return False
    if value_col not in flat.columns or group_col not in flat.columns:
        print(f"Skipping {filename}: missing {value_col} or {group_col}")
        return False
    plot_df = flat[[group_col, value_col]].dropna()
    if plot_df.empty:
        print(f"Skipping {filename}: no data")
        return False
    fig, ax = plt.subplots(figsize=(11, 6))
    groups = [group[value_col].to_numpy() for _, group in plot_df.groupby(group_col)]
    labels = [str(name) for name, _ in plot_df.groupby(group_col)]
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=200)
    plt.close(fig)
    return True


def save_confusion_matrices(confusion: pd.DataFrame | None) -> list[str]:
    if confusion is None:
        return []
    needed = {"analysis_config", "gold_label", "predicted_category", "count"}
    if not needed.issubset(confusion.columns):
        print("Skipping confusion matrices: required columns unavailable")
        return []
    generated: list[str] = []
    for config, group in confusion.groupby("analysis_config"):
        labels = sorted(
            set(group["gold_label"].astype(str)) | set(group["predicted_category"].astype(str))
        )
        matrix = pd.DataFrame(0, index=labels, columns=labels)
        for _, row in group.iterrows():
            matrix.loc[str(row["gold_label"]), str(row["predicted_category"])] = row["count"]
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(matrix.to_numpy())
        ax.set_title(f"Confusion Matrix: {config}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for row_index, label_row in enumerate(labels):
            for col_index, label_col in enumerate(labels):
                value = matrix.loc[label_row, label_col]
                if value:
                    ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        filename = f"confusion_matrix_{safe_name(str(config))}.png"
        fig.savefig(PLOT_DIR / filename, dpi=200)
        plt.close(fig)
        generated.append(filename)
    return generated


def save_top_misclassification_pairs(top_mis: pd.DataFrame | None) -> bool:
    if top_mis is None:
        return False
    needed = {"gold_label", "predicted_category", "count"}
    if not needed.issubset(top_mis.columns):
        print("Skipping top_misclassification_pairs.png: required columns unavailable")
        return False
    plot_df = top_mis.head(20).copy()
    if plot_df.empty:
        print("Skipping top_misclassification_pairs.png: no misclassification pairs")
        return False
    plot_df["pair"] = (
        plot_df["gold_label"].astype(str) + " -> " + plot_df["predicted_category"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["pair"], plot_df["count"])
    ax.set_title("Top Misclassification Pairs")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "top_misclassification_pairs.png", dpi=200)
    plt.close(fig)
    return True


def save_per_category_accuracy(category: pd.DataFrame | None) -> bool:
    if category is None:
        return False
    needed = {"analysis_config", "gold_label", "accuracy"}
    if not needed.issubset(category.columns):
        print("Skipping per_category_accuracy_by_config.png: required columns unavailable")
        return False
    pivot = category.pivot(index="gold_label", columns="analysis_config", values="accuracy")
    if pivot.empty:
        print("Skipping per_category_accuracy_by_config.png: no data")
        return False
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Per-Category Accuracy by Config")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "per_category_accuracy_by_config.png", dpi=200)
    plt.close(fig)
    return True


def save_quality_escalated(escalation: pd.DataFrame | None) -> bool:
    if escalation is None:
        return False
    needed = {"analysis_config", "escalated_quality", "non_escalated_quality"}
    if not needed.issubset(escalation.columns):
        print("Skipping quality_escalated_vs_not_escalated.png: required columns unavailable")
        return False
    plot_df = escalation[list(needed)].dropna(how="all")
    if plot_df.empty:
        print("Skipping quality_escalated_vs_not_escalated.png: no data")
        return False
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(plot_df))
    ax.bar([i - 0.2 for i in x], plot_df["escalated_quality"], width=0.4, label="Escalated")
    ax.bar([i + 0.2 for i in x], plot_df["non_escalated_quality"], width=0.4, label="Not escalated")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["analysis_config"], rotation=35, ha="right")
    ax.set_title("Quality: Escalated vs Not Escalated")
    ax.set_ylabel("Classification Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "quality_escalated_vs_not_escalated.png", dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    summary = read_csv("summary_by_config.csv")
    integrity = read_csv("dataset_integrity_by_config.csv")
    metrics = read_csv("classification_metrics_by_config.csv")
    category = read_csv("classification_accuracy_by_category.csv")
    confusion = read_csv("confusion_matrix_by_config.csv")
    top_mis = read_csv("top_misclassification_pairs.csv")
    generation = read_csv("generation_quality_by_config.csv")
    cost = read_csv("projected_47k_cost_by_config.csv")
    savings = read_csv("cost_savings_vs_llm_baseline.csv")
    latency = read_csv("latency_summary_by_config.csv")
    throughput = read_csv("throughput_summary_by_config.csv")
    escalation = read_csv("routing_escalation_summary.csv")
    escalation_category = read_csv("escalation_by_category.csv")
    router_family = read_csv("summary_by_router_family.csv")
    model = read_csv("summary_by_model.csv")
    flat = read_csv("balanced_60k_all_configs_flat.csv")
    memory = read_csv("memory_mode_summary.csv")

    plot_jobs = [
        (integrity, "analysis_config", "rows", "row_count_by_config.png", "Rows by Config", "Rows"),
        (
            integrity,
            "analysis_config",
            "unique_ticket_ids",
            "ticket_coverage_by_config.png",
            "Ticket Coverage by Config",
            "Unique Ticket IDs",
        ),
        (
            summary,
            "analysis_config",
            "bad_row_count",
            "bad_row_count_by_config.png",
            "Bad Row Count by Config",
            "Bad Rows",
        ),
        (
            metrics,
            "analysis_config",
            "accuracy",
            "classification_accuracy_by_config.png",
            "Classification Accuracy by Config",
            "Accuracy",
        ),
        (
            metrics,
            "analysis_config",
            "macro_f1",
            "macro_f1_by_config.png",
            "Macro F1 by Config",
            "Macro F1",
        ),
        (
            metrics,
            "analysis_config",
            "weighted_f1",
            "weighted_f1_by_config.png",
            "Weighted F1 by Config",
            "Weighted F1",
        ),
        (
            generation,
            "analysis_config",
            "success_rate",
            "success_rate_by_config.png",
            "Success Rate by Config",
            "Success Rate",
        ),
        (
            generation,
            "analysis_config",
            "generation_valid_rate",
            "generation_validity_by_config.png",
            "Generation Validity by Config",
            "Generation Valid Rate",
        ),
        (
            generation,
            "analysis_config",
            "analysis_usable_rate",
            "usable_output_rate_by_config.png",
            "Usable Output Rate by Config",
            "Usable Rate",
        ),
        (
            generation,
            "analysis_config",
            "parse_error_rate",
            "parse_error_rate_by_config.png",
            "Parse Error Rate by Config",
            "Parse Error Rate",
        ),
        (
            generation,
            "analysis_config",
            "validation_error_rate",
            "validation_error_rate_by_config.png",
            "Validation Error Rate by Config",
            "Validation Error Rate",
        ),
        (
            generation,
            "analysis_config",
            "empty_resolution_steps_rate",
            "empty_resolution_steps_rate_by_config.png",
            "Empty Resolution Steps Rate by Config",
            "Empty Resolution Steps Rate",
        ),
        (
            generation,
            "analysis_config",
            "llm_unavailable_rate",
            "llm_unavailable_rate_by_config.png",
            "LLM Unavailable Rate by Config",
            "LLM Unavailable Rate",
        ),
        (
            generation,
            "analysis_config",
            "avg_resolution_steps",
            "avg_resolution_steps_by_config.png",
            "Average Resolution Steps by Config",
            "Average Steps",
        ),
        (
            generation,
            "analysis_config",
            "avg_final_answer_length",
            "avg_final_answer_length_by_config.png",
            "Average Final Answer Length by Config",
            "Characters",
        ),
        (
            summary,
            "analysis_config",
            "avg_total_input_tokens",
            "average_input_tokens_by_config.png",
            "Average Input Tokens by Config",
            "Tokens",
        ),
        (
            summary,
            "analysis_config",
            "avg_total_output_tokens",
            "average_output_tokens_by_config.png",
            "Average Output Tokens by Config",
            "Tokens",
        ),
        (
            summary,
            "analysis_config",
            "avg_total_tokens",
            "average_total_tokens_by_config.png",
            "Average Total Tokens by Config",
            "Tokens",
        ),
        (
            summary,
            "analysis_config",
            "avg_total_cost_usd",
            "average_cost_per_ticket_by_config.png",
            "Average Cost per Ticket by Config",
            "USD",
        ),
        (
            cost,
            "analysis_config",
            "projected_47k_cost_usd",
            "projected_47k_cost_by_config.png",
            "Projected 47k Cost by Config",
            "USD",
        ),
        (
            savings,
            "comparison_config",
            "projected_47k_savings_usd",
            "cost_savings_vs_llm_baseline.png",
            "Projected 47k Savings vs LLM Baseline",
            "USD",
        ),
        (
            latency,
            "analysis_config",
            "mean_latency_ms",
            "average_latency_by_config.png",
            "Average Latency by Config",
            "Milliseconds",
        ),
        (
            latency,
            "analysis_config",
            "median_latency_ms",
            "median_latency_by_config.png",
            "Median Latency by Config",
            "Milliseconds",
        ),
        (
            latency,
            "analysis_config",
            "p95_latency_ms",
            "p95_latency_by_config.png",
            "P95 Latency by Config",
            "Milliseconds",
        ),
        (
            throughput,
            "analysis_config",
            "tickets_per_minute_per_worker",
            "throughput_by_config.png",
            "Throughput by Config",
            "Tickets per Minute per Worker",
        ),
        (
            escalation,
            "analysis_config",
            "escalation_rate",
            "escalation_rate_by_config.png",
            "Escalation Rate by Config",
            "Escalation Rate",
        ),
        (
            escalation_category,
            "gold_label",
            "escalation_rate",
            "escalation_rate_by_category.png",
            "Escalation Rate by Category",
            "Escalation Rate",
        ),
        (
            router_family,
            "router_family",
            "classification_accuracy",
            "router_family_quality_comparison.png",
            "Router Family Quality Comparison",
            "Classification Accuracy",
        ),
        (
            model,
            "base_model",
            "classification_accuracy",
            "model_quality_comparison.png",
            "Model Quality Comparison",
            "Classification Accuracy",
        ),
        (
            router_family,
            "router_family",
            "avg_total_cost_usd",
            "router_family_cost_comparison.png",
            "Router Family Cost Comparison",
            "USD per Ticket",
        ),
        (
            model,
            "base_model",
            "avg_total_cost_usd",
            "model_cost_comparison.png",
            "Model Cost Comparison",
            "USD per Ticket",
        ),
    ]

    for df, x_col, y_col, filename, title, ylabel in plot_jobs:
        if save_bar(df, x_col, y_col, filename, title, ylabel):
            generated.append(filename)

    if save_per_category_accuracy(category):
        generated.append("per_category_accuracy_by_config.png")
    generated.extend(save_confusion_matrices(confusion))
    if save_top_misclassification_pairs(top_mis):
        generated.append("top_misclassification_pairs.png")
    if save_distribution(
        flat,
        "resolution_step_count",
        "analysis_config",
        "resolution_step_distribution_by_config.png",
        "Resolution Step Distribution by Config",
        "Config",
    ):
        generated.append("resolution_step_distribution_by_config.png")
    if save_distribution(
        flat,
        "final_answer_length",
        "analysis_config",
        "final_answer_length_distribution_by_config.png",
        "Final Answer Length Distribution by Config",
        "Config",
    ):
        generated.append("final_answer_length_distribution_by_config.png")

    scatter_jobs = [
        (
            summary,
            "avg_total_cost_usd",
            "classification_accuracy",
            "analysis_config",
            "accuracy_vs_cost.png",
            "Accuracy vs Cost",
            "Average Cost per Ticket (USD)",
            "Classification Accuracy",
        ),
        (
            summary,
            "avg_total_cost_usd",
            "analysis_usable_rate",
            "analysis_config",
            "usable_rate_vs_cost.png",
            "Usable Rate vs Cost",
            "Average Cost per Ticket (USD)",
            "Usable Output Rate",
        ),
        (
            summary,
            "mean_latency_ms",
            "classification_accuracy",
            "analysis_config",
            "quality_vs_latency.png",
            "Quality vs Latency",
            "Mean Latency (ms)",
            "Classification Accuracy",
        ),
        (
            summary,
            "avg_total_cost_usd",
            "classification_accuracy",
            "analysis_config",
            "slm_dominant_vs_llm_baseline_tradeoff.png",
            "SLM-Dominant vs LLM Baseline Tradeoff",
            "Average Cost per Ticket (USD)",
            "Classification Accuracy",
        ),
    ]
    for df, x_col, y_col, label_col, filename, title, xlabel, ylabel in scatter_jobs:
        if save_scatter(df, x_col, y_col, label_col, filename, title, xlabel, ylabel):
            generated.append(filename)

    if save_quality_escalated(escalation):
        generated.append("quality_escalated_vs_not_escalated.png")

    if savings is not None and "relative_savings_rate" in savings.columns:
        if save_bar(
            savings,
            "comparison_config",
            "relative_savings_rate",
            "cost_savings_from_routing.png",
            "Cost Savings from Routing",
            "Relative Savings Rate",
        ):
            generated.append("cost_savings_from_routing.png")

    if (
        memory is not None
        and "memory_mode" in memory.columns
        and memory["memory_mode"].nunique() > 1
    ):
        if save_bar(
            memory,
            "memory_mode",
            "classification_accuracy",
            "memory_mode_quality_comparison.png",
            "Memory Mode Quality Comparison",
            "Classification Accuracy",
        ):
            generated.append("memory_mode_quality_comparison.png")
    else:
        print(
            "Skipping memory_mode_quality_comparison.png: only BM25 RAG is present in this dataset"
        )

    print(f"Generated {len(generated)} plots in {PLOT_DIR}")
    for filename in generated:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
