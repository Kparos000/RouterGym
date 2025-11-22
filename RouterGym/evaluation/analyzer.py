"""Benchmark analyzer with plotting helpers and aggregation."""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .metrics import MetricResult, evaluate

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "results.csv"


def load_results(csv_path: Path = RESULTS_CSV) -> pd.DataFrame:
    """Load results.csv into a DataFrame."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def summarize(run_artifacts: Dict[str, Any]) -> MetricResult:
    """Summarize run artifacts into aggregate metrics."""
    return evaluate(run_artifacts)


def aggregate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregate metrics across the dataframe."""
    if df.empty:
        return {}
    metrics = {
        "accuracy": df.get("accuracy", pd.Series(dtype=float)).mean(),
        "groundedness": df.get("groundedness", pd.Series(dtype=float)).mean(),
        "schema_validity": df.get("schema_validity", pd.Series(dtype=float)).mean(),
        "latency": df.get("latency_ms", pd.Series(dtype=float)).mean(),
        "cost": df.get("cost_usd", pd.Series(dtype=float)).mean(),
    }
    return metrics


def compute_global_statistics(results: pd.DataFrame) -> Dict[str, float]:
    """Compute global statistics from results."""
    if results.empty:
        return {}
    return {
        "conversion_rate": (results["model_used"] != "llm").mean() if "model_used" in results else 0.0,
        "mean_latency_ms": results.get("latency_ms", pd.Series(dtype=float)).mean(),
        "mean_cost_usd": results.get("cost_usd", pd.Series(dtype=float)).mean(),
    }


def aggregate_grid_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean metrics by router and memory."""
    if df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["router", "memory", "model"] if c in df.columns]
    agg_cols = [c for c in ["accuracy", "groundedness", "schema_validity", "latency_ms", "cost_usd"] if c in df.columns]
    return df.groupby(group_cols)[agg_cols].mean().reset_index()


def _save_fig(name: str, output_dir: Path = FIGURES_DIR) -> None:
    """Save current matplotlib figure to the figures directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_comparison(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Bar plot comparing models by accuracy."""
    if df.empty or "model" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="model", y="accuracy", hue="router", errorbar=None)
    plt.title("Model comparison by accuracy")
    _save_fig("model_comparison", output_dir or FIGURES_DIR)


def plot_memory_comparison(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Bar plot comparing memories by accuracy."""
    if df.empty or "memory" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="memory", y="accuracy", hue="router", errorbar=None)
    plt.title("Memory comparison by accuracy")
    _save_fig("memory_comparison", output_dir or FIGURES_DIR)


def plot_router_comparison(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Bar plot comparing routers."""
    if df.empty or "router" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="router", y="accuracy", hue="memory", errorbar=None)
    plt.title("Router comparison")
    _save_fig("router_comparison", output_dir or FIGURES_DIR)


def plot_latency_vs_cost(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Scatter plot of latency vs cost."""
    if df.empty or "latency_ms" not in df.columns or "cost_usd" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="latency_ms", y="cost_usd", hue="router", style="memory")
    plt.title("Latency vs Cost")
    _save_fig("latency_vs_cost", output_dir or FIGURES_DIR)


def plot_accuracy_vs_cost(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Scatter plot of accuracy vs cost."""
    if df.empty or "accuracy" not in df.columns or "cost_usd" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="cost_usd", y="accuracy", hue="router", style="memory")
    plt.title("Accuracy vs Cost")
    _save_fig("accuracy_vs_cost", output_dir or FIGURES_DIR)


def plot_groundedness_distribution(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot groundedness distribution."""
    if df.empty or "groundedness" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.histplot(df["groundedness"], bins=10, kde=True)
    plt.title("Groundedness Distribution")
    _save_fig("groundedness_distribution", output_dir or FIGURES_DIR)


def plot_schema_validity(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot schema validity rates."""
    if df.empty or "schema_validity" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["schema_validity"], y=[df["schema_validity"].mean()])
    plt.title("Schema Validity")
    _save_fig("schema_validity", output_dir or FIGURES_DIR)


def plot_latency_histogram(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot latency histogram."""
    if df.empty or "latency_ms" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.histplot(df["latency_ms"], bins=20)
    plt.title("Latency Histogram")
    _save_fig("latency_histogram", output_dir or FIGURES_DIR)


def plot_router_conversion(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot router conversion rate per strategy."""
    if df.empty or "model_used" not in df.columns:
        return
    conv = (
        df.groupby("router", group_keys=False)["model_used"]
        .apply(lambda s: (s != "llm").mean())
        .reset_index(name="conversion")
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(data=conv, x="router", y="conversion")
    plt.title("Router Conversion Rate (no LLM fallback)")
    _save_fig("router_conversion", output_dir or FIGURES_DIR)


def plot_grid_heatmap(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Heatmap of accuracy by router and memory."""
    if df.empty or "accuracy" not in df.columns:
        return
    pivot = df.pivot_table(values="accuracy", index="router", columns="memory", aggfunc="mean")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Accuracy heatmap by router x memory")
    _save_fig("grid_heatmap", output_dir or FIGURES_DIR)


def export_all_figures(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> None:
    """Generate and save all standard figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(df, out)
    plot_memory_comparison(df, out)
    plot_router_comparison(df, out)
    plot_latency_vs_cost(df, out)
    plot_accuracy_vs_cost(df, out)
    plot_groundedness_distribution(df, out)
    plot_schema_validity(df, out)
    plot_latency_histogram(df, out)
    plot_router_conversion(df, out)
    plot_grid_heatmap(df, out)
