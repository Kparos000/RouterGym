"""Entry point to run RouterGym experiments and generate plots."""

from __future__ import annotations

from pathlib import Path

from RouterGym.experiments.run_grid import run_full_grid
from RouterGym.evaluation import analyzer
from RouterGym.evaluation import stats as eval_stats


def main() -> None:
    """Run the full grid, then generate plots and optional ANOVA."""
    df = run_full_grid()
    results_dir = Path("RouterGym/results")
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


if __name__ == "__main__":
    main()
