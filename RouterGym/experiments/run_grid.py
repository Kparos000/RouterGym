"""Experimental grid runner with routers, memories, and KB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from RouterGym.evaluation import analyzer as eval_analyzer
from RouterGym.data import dataset_loader
from RouterGym.data import kb_loader
from RouterGym.routing.llm_first import LLMFirstRouter
from RouterGym.routing.slm_dominant import SLMDominantRouter
from RouterGym.routing.hybrid_specialist import HybridSpecialistRouter
from RouterGym.memory.none import NoneMemory
from RouterGym.memory.transcript import TranscriptMemory
from RouterGym.memory.rag import RAGMemory
from RouterGym.memory.salience import SalienceGatedMemory


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
DEFAULT_TICKETS_PATH = Path("RouterGym/data/tickets/tickets.csv")
DEFAULT_KB_PATH = Path("RouterGym/data/policy_kb")

ROUTER_NAMES = ["llm_first", "slm_dominant", "hybrid_specialist"]
MEMORY_MODES = ["none", "transcript", "rag", "salience"]
MODEL_NAMES = ["slm1", "slm2", "llm1", "llm2"]


def load_tickets(path: Path = DEFAULT_TICKETS_PATH) -> List[Dict[str, Any]]:
    """Load and preprocess tickets."""
    try:
        return dataset_loader.load_and_preprocess(path)
    except Exception:
        return []


def load_kb(path: Path = DEFAULT_KB_PATH) -> Optional[Any]:
    """Load KB index and return retriever module."""
    try:
        if path.exists():
            kb_loader.refresh_index(path)
            return kb_loader
    except Exception:
        return None
    return None


def init_router(name: str):
    router_map = {
        "llm_first": LLMFirstRouter(),
        "slm_dominant": SLMDominantRouter(),
        "hybrid_specialist": HybridSpecialistRouter(),
    }
    return router_map.get(name)


def init_memory(name: str):
    if name == "none":
        return NoneMemory()
    if name == "transcript":
        return TranscriptMemory()
    if name == "rag":
        return RAGMemory()
    if name == "salience":
        return SalienceGatedMemory()
    return NoneMemory()


def run_single(ticket: Dict[str, Any], router: Any, memory_mode: str, kb_retriever: Optional[Any]) -> Dict[str, Any]:
    """Run a single ticket through a router + memory."""
    memory = init_memory(memory_mode)
    memory.add(ticket.get("text", ""))
    memory_context = memory.get_context()
    routing_meta = router.route(ticket, kb_retriever=kb_retriever) if router else {}
    return {
        "ticket_id": ticket.get("id"),
        "router": routing_meta.get("strategy"),
        "memory": memory_mode,
        "target_model": routing_meta.get("target_model", "slm"),
        "kb_attached": routing_meta.get("kb_attached", False),
        "routing_meta": routing_meta,
        "memory_context": memory_context,
        "result": "success",
    }


def run_config(router_name: str, memory_mode: str, tickets: List[Dict[str, Any]], kb_retriever: Optional[Any]) -> List[Dict[str, Any]]:
    """Run all tickets for a router/memory combo."""
    router = init_router(router_name)
    outputs: List[Dict[str, Any]] = []
    for ticket in tickets:
        outputs.append(run_single(ticket, router, memory_mode, kb_retriever))
    return outputs


def run_full_grid() -> pd.DataFrame:
    """Run the full grid over routers, memories, and models (models are placeholders)."""
    tickets = load_tickets()
    kb_retriever = load_kb()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    aggregate: List[Dict[str, Any]] = []

    for router_name in ROUTER_NAMES:
        for memory_mode in MEMORY_MODES:
            # models loop kept for parity but not used in stub outputs
            for model_name in MODEL_NAMES:
                records = run_config(router_name, memory_mode, tickets, kb_retriever)
                raw_path = RAW_DIR / f"{router_name}__{memory_mode}__{model_name}.jsonl"
                with raw_path.open("w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec) + "\n")
                for rec in records:
                    aggregate.append(
                        {
                            "router": router_name,
                            "memory": memory_mode,
                            "model": model_name,
                            "ticket_id": rec.get("ticket_id"),
                            "kb_attached": rec.get("kb_attached", False),
                            "result": rec.get("result"),
                        }
                    )

    df = pd.DataFrame(aggregate)
    csv_path = RESULTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)

    if not df.empty:
        eval_analyzer.plot_model_comparison(df)
        eval_analyzer.plot_router_performance(df)
        eval_analyzer.plot_memory_effects(df)
        eval_analyzer.plot_grid_heatmap(df)
        eval_analyzer.plot_cost_quality_frontier(df)

    return df


if __name__ == "__main__":
    run_full_grid()
