"""Experiment grid runner stub with plotting hooks and data/KB loaders."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from RouterGym.evaluation import analyzer as eval_analyzer
from RouterGym.data import dataset_loader
from RouterGym.data import kb_loader


def _default_grid() -> Dict[str, List[str]]:
    """Provide a minimal default grid when none is supplied."""
    return {
        "routers": ["slm_dominant", "llm_first", "hybrid_specialist"],
        "memories": ["none", "static", "dynamic", "salience"],
        "slms": ["phi3mini", "qwen1.5b"],
        "llms": ["llama70b", "mixtral46b"],
        "contracts": ["on", "off"],
        "seeds": ["1"],
    }


def run_grid(config: Dict[str, Any]) -> pd.DataFrame:
    """Run an experiment grid (placeholder) and generate plots."""
    grid = config.get("grid") if config else None
    grid = grid or _default_grid()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    tickets_path = data_dir / "tickets"
    kb_path = data_dir / "policy_kb"

    # Load dataset if present.
    try:
        df_raw = dataset_loader.load_kaggle_dataset(tickets_path)
        ticket_records = dataset_loader.preprocess_tickets(df_raw)
    except Exception:
        df_raw = pd.DataFrame()
        ticket_records: List[Dict[str, Any]] = []

    # Load KB index if dependencies are available.
    kb_loaded = False
    try:
        if kb_path.exists():
            kb_loader.load_kb(kb_path)
            kb_loaded = True
    except Exception:
        kb_loaded = False

    results: List[Dict[str, Any]] = []
    routers = grid.get("routers", [])
    memories = grid.get("memories", [])
    models = list(grid.get("slms", [])) + list(grid.get("llms", []))
    seeds = grid.get("seeds", [])

    for router in routers:
        for memory in memories:
            for model in models:
                for seed in seeds:
                    is_slm = model in grid.get("slms", [])
                    results.append(
                        {
                            "router": router,
                            "memory": memory,
                            "model": model,
                            "seed": seed,
                            "groundedness": 0.0,
                            "schema_validity": 0.0,
                            "latency_ms": 0.0,
                            "cost_usd": 0.001 if is_slm else 0.01,
                            "fallback_rate": 0.0,
                            "accuracy": 0.0,
                            "kb_attached": kb_loaded,
                            "tickets_loaded": bool(ticket_records),
                        }
                    )

    df = pd.DataFrame(results)
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    if not df.empty:
        eval_analyzer.plot_model_comparison(df)
        eval_analyzer.plot_router_performance(df)
        eval_analyzer.plot_memory_effects(df)
        eval_analyzer.plot_grid_heatmap(df)
        eval_analyzer.plot_cost_quality_frontier(df)

    return df


if __name__ == "__main__":
    run_grid({})
