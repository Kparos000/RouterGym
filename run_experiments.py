"""Entry point to run RouterGym experiments and generate plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from RouterGym.experiments.run_grid import run_full_grid
from RouterGym.evaluation import analyzer
from RouterGym.evaluation import stats as eval_stats
from RouterGym.data.tickets.dataset_loader import load_dataset
from RouterGym.data.policy_kb.kb_loader import load_kb


def run_pipeline(
    base_dir: Path | None = None,
    limit: int | None = None,
    routers: list[str] | None = None,
    memories: list[str] | None = None,
    models: list[str] | None = None,
    run_anova: bool = True,
    result_filename: str = "results.csv",
    grid_runner=run_full_grid,
) -> None:
    """Run the full grid, then generate plots and optional ANOVA."""
    results_dir = base_dir or Path("RouterGym/results")
    df = grid_runner(limit=limit, routers=routers, memories=memories, models=models)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / result_filename
    df.to_csv(csv_path, index=False)

    # Generate plots
    analyzer.export_all_figures(df, output_dir=results_dir / "figures")

    # Optional ANOVA on accuracy
    if run_anova:
        try:
            eval_stats.export_anova_results(df, filename="anova_accuracy.csv")
        except Exception:
            pass


def run_sanity(base_dir: Path | None = None, limit: int = 5) -> None:
    """Run a shortened sanity sweep with minimal settings."""
    results_dir = base_dir or Path("RouterGym/results")
    print("[Sanity] Loading dataset...")
    df = load_dataset(limit)
    print(f"[Sanity] Loaded {len(df)} tickets")
    print("[Sanity] Loading KB...")
    kb = load_kb()
    print(f"[Sanity] Loaded {len(kb)} KB docs")
    print("[Sanity] Running with LLMFirstRouter + no memory")
    run_pipeline(
        base_dir=results_dir,
        limit=limit,
        routers=["llm_first"],
        memories=["none"],
        models=["llm1"],
        run_anova=False,
        result_filename="sanity_results.csv",
    )


def _load_config(config_path: Path | None) -> dict:
    """Load optional config from YAML or JSON."""
    if config_path is None or not config_path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        try:
            import json
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def main() -> None:
    """Run with default configuration."""
    parser = argparse.ArgumentParser(description="Run RouterGym experiments.")
    parser.add_argument("--sanity", action="store_true", help="Run quick sanity sweep.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickets.")
    parser.add_argument("--config", type=str, default=None, help="Path to grid config (yaml/json).")
    args = parser.parse_args()

    if args.sanity:
        run_sanity(limit=args.limit or 5)
    else:
        cfg = _load_config(Path(args.config)) if args.config else {}
        routers = cfg.get("routers") if isinstance(cfg.get("routers"), list) else None
        memories = cfg.get("memories") if isinstance(cfg.get("memories"), list) else None
        models = cfg.get("models") if isinstance(cfg.get("models"), list) else None
        run_pipeline(limit=args.limit, routers=routers, memories=memories, models=models)


if __name__ == "__main__":
    main()
