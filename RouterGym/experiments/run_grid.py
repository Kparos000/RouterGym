"""Experiment grid runner stub with plotting hooks and data/KB loaders."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from RouterGym.evaluation import analyzer as eval_analyzer
from RouterGym.data import dataset_loader
from RouterGym.data import kb_loader
from RouterGym.routing.llm_first import LLMFirstRouter
from RouterGym.routing.slm_dominant import SLMDominantRouter
from RouterGym.routing.hybrid_specialist import HybridSpecialistRouter


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


def run_full_grid(
    grid: Dict[str, Any],
    tickets: List[Dict[str, Any]],
    kb_retriever: Optional[Any],
) -> pd.DataFrame:
    """Run the inner grid loop (placeholder)."""
    results: List[Dict[str, Any]] = []
    routers = grid.get("routers", [])
    memories = grid.get("memories", [])
    models = list(grid.get("slms", [])) + list(grid.get("llms", []))
    seeds = grid.get("seeds", [])

    router_map = {
        "slm_dominant": SLMDominantRouter(),
        "llm_first": LLMFirstRouter(),
        "hybrid_specialist": HybridSpecialistRouter(),
    }

    sample_text = tickets[0]["text"] if tickets else "Sample ticket text."

    for router_name in routers:
        router = router_map.get(router_name)
        for memory in memories:
            for model in models:
                for seed in seeds:
                    is_slm = model in grid.get("slms", [])
                    routing_meta = router.route(sample_text, kb_retriever=kb_retriever) if router else {}
                    results.append(
                        {
                            "router": router_name,
                            "memory": memory,
                            "model": model,
                            "seed": seed,
                            "groundedness": 0.0,
                            "schema_validity": 0.0,
                            "latency_ms": 0.0,
                            "cost_usd": 0.001 if is_slm else 0.01,
                            "fallback_rate": 0.0,
                            "accuracy": 0.0,
                            "kb_attached": bool(kb_retriever),
                            "tickets_loaded": bool(tickets),
                            "routing_meta": routing_meta,
                        }
                    )
    return pd.DataFrame(results)


def run_grid(config: Dict[str, Any]) -> pd.DataFrame:
    """Run an experiment grid (placeholder) and generate plots."""
    grid = config.get("grid") if config else None
    grid = grid or _default_grid()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    tickets_path = data_dir / "tickets" / "tickets.csv"
    kb_path = data_dir / "policy_kb"

    # Load dataset if present.
    try:
        ticket_records = dataset_loader.load_and_preprocess(tickets_path)
    except Exception:
        ticket_records = []

    # Load KB index if dependencies are available.
    kb_retriever: Optional[Any] = None
    try:
        if kb_path.exists():
            kb_loader.load_kb(kb_path)
            kb_retriever = kb_loader
    except Exception:
        kb_retriever = None

    df = run_full_grid(grid=grid, tickets=ticket_records, kb_retriever=kb_retriever)

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
