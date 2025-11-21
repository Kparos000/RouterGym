"""Benchmark analyzer stubs with plotting helpers."""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .metrics import MetricResult, evaluate

PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def summarize(run_artifacts: Dict[str, Any]) -> MetricResult:
    """Summarize run artifacts into aggregate metrics."""
    return evaluate(run_artifacts)


def _save_fig(name: str) -> None:
    """Save current matplotlib figure to the plots directory."""
    path = PLOTS_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_comparison(df: pd.DataFrame) -> None:
    """Bar plot comparing models by accuracy."""
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="model", y="accuracy", hue="router", errorbar=None)
    plt.title("Model comparison by accuracy")
    _save_fig("model_comparison")


def plot_router_performance(df: pd.DataFrame) -> None:
    """Bar plot comparing routers."""
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="router", y="accuracy", hue="memory", errorbar=None)
    plt.title("Router performance by memory setting")
    _save_fig("router_performance")


def plot_memory_effects(df: pd.DataFrame) -> None:
    """Bar plot for memory effects on groundedness."""
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="memory", y="groundedness", hue="router", errorbar=None)
    plt.title("Memory effects on groundedness")
    _save_fig("memory_effects")


def plot_grid_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of accuracy by router and memory."""
    if df.empty:
        return
    pivot = df.pivot_table(values="accuracy", index="router", columns="memory", aggfunc="mean")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Accuracy heatmap by router x memory")
    _save_fig("grid_heatmap")


def plot_cost_quality_frontier(df: pd.DataFrame) -> None:
    """Scatter of cost vs accuracy."""
    if df.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="cost_usd", y="accuracy", hue="model", style="router")
    plt.title("Cost vs quality frontier")
    _save_fig("cost_quality_frontier")
