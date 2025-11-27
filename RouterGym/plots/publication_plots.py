"""Publication-ready plotting utilities for RouterGym experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


def _add_value_labels(ax: plt.Axes, padding: float = 0.01) -> None:
    """Annotate bars with their heights."""
    for patch_obj in ax.patches:
        patch = cast(Any, patch_obj)
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + padding,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_router_conversion_cost(df: pd.DataFrame, outdir: Path, n_tickets: int) -> None:
    grouped = (
        df.assign(is_slm=df["model_used"].str.lower() == "slm")
        .groupby("router")
        .agg(slm_share=("is_slm", "mean"), mean_cost=("cost_usd", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(grouped["router"], grouped["slm_share"], color="#4c72b0")
    axes[0].set_title(f"SLM share by router (N={n_tickets} tickets)")
    axes[0].set_ylabel("Fraction handled by SLM")
    axes[0].set_ylim(0, 1)
    _add_value_labels(axes[0])

    axes[1].bar(grouped["router"], grouped["mean_cost"], color="#55a868")
    axes[1].set_title(f"Mean cost by router (N={n_tickets} tickets)")
    axes[1].set_ylabel("Mean cost (USD)")
    _add_value_labels(axes[1])

    fig.tight_layout()
    fig.savefig(outdir / "router_conversion_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latency_vs_cost(df: pd.DataFrame, outdir: Path, n_tickets: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    routers = sorted(df["router"].unique())
    memories = sorted(df["memory"].unique())
    router_palette = dict(zip(routers, sns.color_palette(n_colors=len(routers))))
    memory_markers = dict(zip(memories, ["o", "s", "D", "^", "v", "P", "*", "X"]))

    for router in routers:
        for memory in memories:
            subset = df[(df["router"] == router) & (df["memory"] == memory)]
            if subset.empty:
                continue
            ax.scatter(
                subset["latency_ms"],
                subset["cost_usd"],
                color=router_palette[router],
                marker=memory_markers.get(memory, "o"),
                alpha=0.6,
                edgecolors="none",
                label=f"{router} / {memory}",
            )

    ax.set_title(f"Latency vs cost by router × memory (N={n_tickets} tickets)")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cost (USD)")
    if (df["cost_usd"] > 0).all():
        cost_range = df["cost_usd"].max() / max(df["cost_usd"].min(), 1e-9)
        if cost_range > 10:
            ax.set_yscale("log")

    router_handles = [
        Line2D([0], [0], color=router_palette[r], marker="o", linestyle="", label=r) for r in routers
    ]
    memory_handles = [
        Line2D([0], [0], color="black", marker=memory_markers[m], linestyle="", label=m) for m in memories
    ]
    leg1 = ax.legend(handles=router_handles, title="Router", loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=memory_handles, title="Memory", loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "latency_vs_cost_by_router_memory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_groundedness_by_memory(df: pd.DataFrame, outdir: Path, n_tickets: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="memory", y="groundedness", ax=ax, color="#c5d1eb")
    means = df.groupby("memory")["groundedness"].mean().reset_index()
    ax.scatter(means["memory"], means["groundedness"], color="#4c72b0", marker="D", label="Mean")
    ax.set_title(f"Groundedness by memory regime (N={n_tickets} tickets)")
    ax.set_xlabel("Memory")
    ax.set_ylabel("Groundedness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "groundedness_by_memory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(df: pd.DataFrame, outdir: Path, n_tickets: int) -> None:
    gold = df["gold_category"].astype(str).str.strip()
    pred = df["predicted_category"].astype(str).str.strip()
    mask = (gold != "") & (pred != "")
    if not mask.any():
        return
    sub = df[mask]
    pivot = (
        sub.pivot_table(index="gold_category", columns="predicted_category", aggfunc="size", fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax)
    ax.set_title(f"Category confusion matrix (N={n_tickets} tickets)")
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Gold category")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix_categories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize(df: pd.DataFrame, outdir: Path) -> None:
    df = df.copy()
    df["is_slm"] = df["model_used"].str.lower() == "slm"
    df["kb_attached_flag"] = df["kb_attached"].astype(bool)

    by_router = (
        df.groupby("router")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_groundedness=("groundedness", "mean"),
            mean_schema_validity=("schema_validity", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            mean_cost_usd=("cost_usd", "mean"),
            slm_share=("is_slm", "mean"),
        )
        .sort_index()
    )

    by_memory = (
        df.groupby("memory")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_groundedness=("groundedness", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            mean_cost_usd=("cost_usd", "mean"),
            kb_attach_rate=("kb_attached_flag", "mean"),
        )
        .sort_index()
    )

    by_model_used = (
        df.groupby("model_used")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_groundedness=("groundedness", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            mean_cost_usd=("cost_usd", "mean"),
        )
        .sort_index()
    )

    by_router.to_csv(outdir / "summary_by_router.csv")
    by_memory.to_csv(outdir / "summary_by_memory.csv")
    by_model_used.to_csv(outdir / "summary_by_model_used.csv")

    print("\nSummary by router:\n", by_router)
    print("\nSummary by memory:\n", by_memory)
    print("\nSummary by model_used:\n", by_model_used)


def main(input_path: Path) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        fallback = Path("RouterGym") / "results" / "experiments" / "results.csv"
        alt = input_path.with_name("results.csv")
        available = sorted(fallback.parent.glob("*.csv")) if fallback.parent.exists() else []
        if fallback.exists():
            print(f"[warn] {input_path} not found; falling back to {fallback}")
            input_path = fallback
        elif alt.exists():
            print(f"[warn] {input_path} not found; falling back to {alt}")
            input_path = alt
        else:
            available_list = "\n".join(str(p) for p in available) if available else "  (none)"
            raise FileNotFoundError(
                f"Input CSV not found: {input_path}\n"
                f"Available CSV files in {fallback.parent}:\n{available_list}"
            )

    df = pd.read_csv(input_path)
    n_tickets = df["ticket_id"].nunique()
    outdir = Path("RouterGym") / "results" / "plots" / f"run_{n_tickets}tickets"
    outdir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plot_router_conversion_cost(df, outdir, n_tickets)
    plot_latency_vs_cost(df, outdir, n_tickets)
    plot_groundedness_by_memory(df, outdir, n_tickets)
    plot_confusion_matrix(df, outdir, n_tickets)
    summarize(df, outdir)

    print(f"\nSaved figures and tables to: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-ready plots for RouterGym experiments.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to experiment CSV (e.g., RouterGym/results/experiments/results_200tickets.csv)",
    )
    args = parser.parse_args()
    main(args.input)
