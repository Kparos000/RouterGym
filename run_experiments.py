"""Entry point to run RouterGym experiments and generate plots."""

from __future__ import annotations

from pathlib import Path

from RouterGym.experiments.run_grid import run_full_grid
from RouterGym.evaluation import analyzer
from RouterGym.evaluation import stats as eval_stats


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


def main() -> None:
    """Run with default configuration."""
    run_pipeline()


if __name__ == "__main__":
    main()
