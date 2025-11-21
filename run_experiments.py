"""Entry point to run RouterGym experiments and generate plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from RouterGym.experiments.run_grid import run_full_grid
from RouterGym.evaluation import analyzer
from RouterGym.evaluation import stats as eval_stats
from RouterGym.data import dataset_loader


def run_pipeline(base_dir: Path | None = None, grid_runner=run_full_grid) -> None:
    """Run the full grid, then generate plots and optional ANOVA."""
    results_dir = base_dir or Path("RouterGym/results")
    df = grid_runner()
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    # Generate plots
    analyzer.export_all_figures(df, output_dir=results_dir / "plots")

    # Optional ANOVA on accuracy
    try:
        eval_stats.export_anova_results(df, filename="anova_accuracy.csv")
    except Exception:
        pass


def run_sanity(base_dir: Path | None = None) -> None:
    """Run a shortened sanity sweep with minimal settings."""
    results_dir = base_dir or Path("RouterGym/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Sample tickets
    tickets_path = Path("RouterGym/data/tickets/tickets.csv")
    try:
        tickets = dataset_loader.load_and_preprocess(tickets_path)
        tickets = tickets[:5]
    except Exception:
        tickets = []

    # Build a minimal DataFrame manually
    rows = []
    for t in tickets:
        rows.append(
            {
                "router": "llm_first",
                "memory": "none",
                "model": "llm1",
                "ticket_id": t.get("id"),
                "accuracy": 0.0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
            }
        )
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = results_dir / "sanity_results.csv"
    df.to_csv(csv_path, index=False)

    # Generate plots (sanity)
    analyzer.export_all_figures(df, output_dir=results_dir / "plots")


def main() -> None:
    """Run with default configuration."""
    parser = argparse.ArgumentParser(description="Run RouterGym experiments.")
    parser.add_argument("--sanity", action="store_true", help="Run quick sanity sweep.")
    args = parser.parse_args()

    if args.sanity:
        run_sanity()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
