"""Experimental grid runner with routers, memories, and KB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from RouterGym.evaluation import analyzer as eval_analyzer
from RouterGym.data.tickets import dataset_loader
from RouterGym.data.policy_kb import kb_loader
from RouterGym.engines.model_registry import load_models
from RouterGym.routing.llm_first import LLMFirstRouter
from RouterGym.routing.slm_dominant import SLMDominantRouter
from RouterGym.routing.hybrid_specialist import HybridSpecialistRouter
from RouterGym.memory.base import MemoryBase
from RouterGym.memory.none import NoneMemory
from RouterGym.memory.transcript import TranscriptMemory
from RouterGym.memory.rag import RAGMemory
from RouterGym.memory.salience import SalienceGatedMemory
from RouterGym.routing.base import BaseRouter


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
DEFAULT_TICKETS_PATH = Path("RouterGym/data/tickets/tickets.csv")
DEFAULT_KB_PATH = Path("RouterGym/data/policy_kb")

ROUTER_NAMES = ["llm_first", "slm_dominant", "hybrid_specialist"]
MEMORY_MODES = ["none", "transcript", "rag", "salience"]
MODEL_NAMES = ["slm1", "slm2", "llm1", "llm2"]


def load_tickets(path: Path = DEFAULT_TICKETS_PATH, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load and preprocess tickets."""
    try:
        df = dataset_loader.load_dataset(limit if limit is not None else None)
        records: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            records.append({"id": idx, "text": row.get("text", ""), "category": row.get("label")})
        return records
    except Exception:
        return []


def load_kb(path: Path = DEFAULT_KB_PATH) -> Optional[Any]:
    """Load KB mapping and return loader module."""
    try:
        kb_loader.load_kb(path)
        return kb_loader
    except Exception:
        return None


def init_router(name: str) -> Optional[BaseRouter]:
    router_map = {
        "llm_first": LLMFirstRouter(),
        "slm_dominant": SLMDominantRouter(),
        "hybrid_specialist": HybridSpecialistRouter(),
    }
    return router_map.get(name)


def init_memory(name: str) -> MemoryBase:
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
    routing_meta = router.route(ticket, kb=kb_retriever, models=None, memory=memory) if router else {}
    return {
        "ticket_id": ticket.get("id"),
        "router": routing_meta.get("strategy"),
        "memory": memory_mode,
        "target_model": routing_meta.get("target_model", "slm"),
        "kb_attached": routing_meta.get("kb_attached", False),
        "routing_meta": routing_meta,
        "memory_context": memory_context,
        "result": "success",
        "accuracy": 0.0,
        "groundedness": 0.0,
        "schema_validity": 0.0,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
    }


def run_config(router_name: str, memory_mode: str, tickets: List[Dict[str, Any]], kb_retriever: Optional[Any]) -> List[Dict[str, Any]]:
    """Run all tickets for a router/memory combo."""
    router = init_router(router_name)
    outputs: List[Dict[str, Any]] = []
    for ticket in tickets:
        outputs.append(run_single(ticket, router, memory_mode, kb_retriever))
    return outputs


def run_full_grid(
    tickets: Optional[List[Dict[str, Any]]] = None,
    limit: Optional[int] = None,
    routers: Optional[List[str]] = None,
    memories: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    kb_retriever: Optional[Any] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the full grid over routers, memories, and models (models are placeholders)."""
    tickets = tickets if tickets is not None else load_tickets(limit=limit)
    kb_retriever = kb_retriever if kb_retriever is not None else load_kb()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    aggregate: List[Dict[str, Any]] = []

    router_list = routers or ROUTER_NAMES
    memory_list = memories or MEMORY_MODES
    model_list = models or MODEL_NAMES

    for router_name in router_list:
        for memory_mode in memory_list:
            for model_name in model_list:
                if verbose:
                    print(f"[Grid] Router={router_name} Memory={memory_mode} Model={model_name} Tickets={len(tickets)}")
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
                            "accuracy": rec.get("accuracy", 0.0),
                            "groundedness": rec.get("groundedness", 0.0),
                            "schema_validity": rec.get("schema_validity", 0.0),
                            "latency_ms": rec.get("latency_ms", 0.0),
                            "cost_usd": rec.get("cost_usd", 0.0),
                        }
                    )

    df = pd.DataFrame(aggregate)
    csv_path = RESULTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)

    if not df.empty:
        eval_analyzer.export_all_figures(df)

    return df


def main() -> None:
    """CLI entrypoint for running the grid."""
    parser = argparse.ArgumentParser(description="RouterGym grid runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickets")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (not used yet)")
    args = parser.parse_args()

    try:
        if args.verbose:
            print("[Grid] Loading dataset...")
        tickets = dataset_loader.load_and_preprocess(DEFAULT_TICKETS_PATH, limit=args.limit) if hasattr(dataset_loader, "load_and_preprocess") else dataset_loader.load_dataset(args.limit)
        if args.verbose:
            print(f"[Grid] Loaded tickets: {len(tickets)}")

        if args.verbose:
            print("[Grid] Loading KB...")
        kb_docs = kb_loader.load_kb()
        if args.verbose:
            print(f"[Grid] Loaded KB docs: {len(kb_docs)}")

        if args.verbose:
            print("[Grid] Loading models...")
        _ = load_models(sanity=False)

        df = run_full_grid(tickets=tickets, kb_retriever=kb_loader, limit=args.limit, verbose=args.verbose)
        out_dir = RESULTS_DIR / "experiments"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.csv"
        df.to_csv(out_path, index=False)
        if args.verbose:
            print(f"[Grid] Results saved to {out_path}")
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"[Grid] Error: {exc}")


if __name__ == "__main__":
    main()
