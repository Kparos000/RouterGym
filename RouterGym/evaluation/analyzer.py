"""Benchmark analyzer with plotting helpers."""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .metrics import MetricResult, evaluate

PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "results.csv"


def load_results(csv_path: Path = RESULTS_CSV) -> pd.DataFrame:
    """Load results.csv into a DataFrame."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


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
    if df.empty or "model" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="model", y="accuracy", hue="router", errorbar=None)
    plt.title("Model comparison by accuracy")
    _save_fig("model_comparison")


def plot_memory_comparison(df: pd.DataFrame) -> None:
    """Bar plot comparing memories by accuracy."""
    if df.empty or "memory" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="memory", y="accuracy", hue="router", errorbar=None)
    plt.title("Memory comparison by accuracy")
    _save_fig("memory_comparison")


def plot_router_comparison(df: pd.DataFrame) -> None:
    """Bar plot comparing routers."""
    if df.empty or "router" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="router", y="accuracy", hue="memory", errorbar=None)
    plt.title("Router comparison")
    _save_fig("router_comparison")


def plot_latency_vs_cost(df: pd.DataFrame) -> None:
    """Scatter plot of latency vs cost."""
    if df.empty or "latency_ms" not in df.columns or "cost_usd" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="latency_ms", y="cost_usd", hue="router", style="memory")
    plt.title("Latency vs Cost")
    _save_fig("latency_vs_cost")


def plot_accuracy_vs_cost(df: pd.DataFrame) -> None:
    """Scatter plot of accuracy vs cost."""
    if df.empty or "accuracy" not in df.columns or "cost_usd" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="cost_usd", y="accuracy", hue="router", style="memory")
    plt.title("Accuracy vs Cost")
    _save_fig("accuracy_vs_cost")


def plot_grid_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of accuracy by router and memory."""
    if df.empty or "accuracy" not in df.columns:
        return
    pivot = df.pivot_table(values="accuracy", index="router", columns="memory", aggfunc="mean")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Accuracy heatmap by router x memory")
    _save_fig("grid_heatmap")
