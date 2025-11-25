"""Experimental grid runner with routers, memories, and KB."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from RouterGym.evaluation import analyzer as eval_analyzer
from RouterGym.evaluation import metrics as eval_metrics
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
from RouterGym.agents.generator import normalize_output
from RouterGym.utils.kb_utils import coerce_kb_hits
from RouterGym.contracts.schema_contract import SchemaContract


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
DEFAULT_TICKETS_PATH = Path("RouterGym/data/tickets/tickets.csv")
DEFAULT_KB_PATH = Path("RouterGym/data/policy_kb")

ROUTER_NAMES = ["llm_first", "slm_dominant", "hybrid_specialist"]
MEMORY_MODES = ["none", "transcript", "rag", "salience"]
MODEL_NAMES = ["slm1", "slm2", "llm1", "llm2"]


def _as_dict(obj: Any, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return obj if dict else a safe default dict."""
    if isinstance(obj, dict):
        return obj
    return default or {}


def _coerce_tickets(tickets: Any) -> List[Dict[str, Any]]:
    """Ensure tickets is a list of dicts."""
    records: List[Dict[str, Any]] = []
    if isinstance(tickets, pd.DataFrame):
        for idx, row in tickets.iterrows():
            rec = {
                "id": row.get("id", idx),
                "text": row.get("text", row.get("document", "")),
                "category": row.get("category", row.get("label", "")),
            }
            records.append(rec)
        return records
    if isinstance(tickets, list):
        for idx, t in enumerate(tickets):
            if isinstance(t, dict):
                records.append(
                    {
                        "id": t.get("id", idx),
                        "text": t.get("text", ""),
                        "category": t.get("category", t.get("label", "")),
                    }
                )
            else:
                records.append({"id": idx, "text": str(t), "category": ""})
    return records


def release_local_models(models: Optional[Dict[str, Any]]) -> None:
    """Placeholder for backward compatibility; remote models do not need unloading."""
    if not models:
        return
    gc.collect()


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


def _coerce_record(rec: Any) -> Dict[str, Any]:
    """Ensure records are dictionaries to avoid attribute errors downstream."""
    if isinstance(rec, dict):
        return rec
    return {"result": "error", "raw_record": str(rec)}


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


def run_single(
    ticket: Dict[str, Any],
    router: Any,
    memory_mode: str,
    kb_retriever: Optional[Any],
    models: Optional[Dict[str, Any]],
    force_llm: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single ticket through a router + memory."""
    start_total = time.perf_counter()
    if not isinstance(ticket, dict):
        return {
            "ticket_id": None,
            "router": None,
            "memory": memory_mode,
            "target_model": "unknown",
            "kb_attached": False,
            "routing_meta": {},
            "memory_context": "",
            "result": "error",
            "accuracy": 0.0,
            "groundedness": 0.0,
            "schema_validity": 0.0,
            "latency_ms": 0.0,
            "cost_usd": 0.0,
            "output": normalize_output(""),
            "model_used": "unknown",
            "json_valid": False,
            "schema_valid": False,
            "predicted_category": "",
            "routing_time": 0.0,
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "repair_time": 0.0,
            "metrics_time": 0.0,
            "latency_escalated": False,
        }
    try:
        t_route_start = time.perf_counter()
        memory = init_memory(memory_mode)
        memory.add(ticket.get("text", ""))
        memory_context = memory.get_context()
        routing_meta = router.route(ticket, kb=kb_retriever, models=models, memory=memory, force_llm=force_llm) if router else {}
        routing_meta = _as_dict(routing_meta, {"model_used": "unknown", "json_valid": False, "schema_valid": False})
        routing_time = (time.perf_counter() - t_route_start) * 1000
        t_gen_start = time.perf_counter()
        final_output = normalize_output(routing_meta.get("final_output", ""))
        generation_time = (time.perf_counter() - t_gen_start) * 1000
        schema = SchemaContract()
        schema_valid, _ = schema.validate(final_output)
        record = {
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
            "output": final_output,
            "model_used": routing_meta.get("model_used", "slm"),
            "json_valid": True,
            "schema_valid": schema_valid,
            "predicted_category": final_output.get("predicted_category", "") or routing_meta.get("predicted_category", ""),
            "routing_time": routing_time,
            "retrieval_time": 0.0,
            "generation_time": generation_time,
            "repair_time": 0.0,
            "metrics_time": 0.0,
            "latency_escalated": routing_meta.get("latency_escalated", False),
        }
        # compute metrics
        kb_texts = []
        if kb_retriever:
            try:
                t_retr_start = time.perf_counter()
                raw_hits = kb_retriever.retrieve(ticket.get("text", ""), top_k=3)
                hits = coerce_kb_hits(raw_hits)
                kb_texts = [h["text"] for h in hits if h["text"]]
                record["retrieval_time"] = (time.perf_counter() - t_retr_start) * 1000
            except Exception:
                kb_texts = []
        record["kb_snippets"] = kb_texts
        t_metrics_start = time.perf_counter()
        metric_values = eval_metrics.compute_all_metrics(
            {
                "output": record["output"],
                "label": ticket.get("category", ""),
                "predicted": record.get("predicted_category", ""),
                "kb_snippets": kb_texts,
                "model_used": record["model_used"],
                "latency_ms": record["latency_ms"],
                "reasoning": record["output"].get("reasoning", "") if isinstance(record["output"], dict) else "",
            }
        )
        record.update(
            {
                "accuracy": metric_values["accuracy"],
                "groundedness": metric_values["groundedness"],
                "schema_validity": metric_values["schema_validity"],
                "cost_usd": metric_values["cost"],
            }
        )
        record["metrics_time"] = (time.perf_counter() - t_metrics_start) * 1000
        record["latency_ms"] = (time.perf_counter() - start_total) * 1000
        if verbose:
            print(
                f"[Single] model_used={record['model_used']} json_valid={record['json_valid']} "
                f"schema_valid={record['schema_valid']} cost={record['cost_usd']:.4f}"
            )
        return record
    except Exception:
        print("[Single] Exception encountered")
        print(traceback.format_exc())
        print(
            f"[Single][Types] routing_meta={type(locals().get('routing_meta')).__name__} "
            f"kb_retriever={type(kb_retriever).__name__ if kb_retriever else 'None'} "
            f"ticket={type(ticket).__name__} models={type(models).__name__}"
        )
        raise


def run_config(
    router_name: str,
    memory_mode: str,
    tickets: List[Dict[str, Any]],
    kb_retriever: Optional[Any],
    models: Optional[Dict[str, Any]],
    force_llm: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run all tickets for a router/memory combo."""
    router = init_router(router_name)
    outputs: List[Dict[str, Any]] = []
    for ticket in tickets:
        try:
            outputs.append(run_single(ticket, router, memory_mode, kb_retriever, models, force_llm=force_llm, verbose=verbose))
        except Exception:
            if verbose:
                print("[Config] Exception in run_single")
                print(traceback.format_exc())
                print(
                    f"[Config][Types] ticket={type(ticket).__name__} router={type(router).__name__ if router else 'None'} "
                    f"kb={type(kb_retriever).__name__ if kb_retriever else 'None'} models={type(models).__name__}"
                )
                raise
            raise
    return outputs


def run_full_grid(
    tickets: Optional[List[Dict[str, Any]]] = None,
    limit: Optional[int] = None,
    routers: Optional[List[str]] = None,
    memories: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    kb_retriever: Optional[Any] = None,
    verbose: bool = False,
    force_llm: bool = False,
    slm_subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run the full grid over routers, memories, and models (models are placeholders)."""
    tickets = _coerce_tickets(tickets if tickets is not None else load_tickets(limit=limit))
    kb_retriever = kb_retriever if kb_retriever is not None else load_kb()
    models_loaded: Optional[Dict[str, Any]] = None
    try:
        models_loaded = load_models(sanity=False, slm_subset=slm_subset, force_llm=force_llm)
    except Exception:
        models_loaded = None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    aggregate: List[Dict[str, Any]] = []

    router_list = routers or ROUTER_NAMES
    memory_list = memories or MEMORY_MODES
    if models:
        model_list = models
    elif slm_subset:
        model_list = slm_subset
    else:
        model_list = ["llm1", "llm2"] if force_llm else MODEL_NAMES

    for router_name in router_list:
        for memory_mode in memory_list:
            for model_name in model_list:
                if verbose:
                    print(f"[Grid] Router={router_name} Memory={memory_mode} Model={model_name} Tickets={len(tickets)}")
                try:
                    records = run_config(router_name, memory_mode, tickets, kb_retriever, models_loaded, force_llm=force_llm, verbose=verbose)
                    raw_path = RAW_DIR / f"{router_name}__{memory_mode}__{model_name}.jsonl"
                    with raw_path.open("w", encoding="utf-8") as f:
                        for rec in records:
                            safe_rec = _coerce_record(rec)
                            f.write(json.dumps(safe_rec) + "\n")
                    for rec in records:
                        safe_rec = _coerce_record(rec)
                        aggregate.append(
                            {
                                "router": router_name,
                                "memory": memory_mode,
                                "model": model_name,
                                "ticket_id": safe_rec.get("ticket_id"),
                                "kb_attached": safe_rec.get("kb_attached", False),
                                "result": safe_rec.get("result"),
                                "accuracy": safe_rec.get("accuracy", 0.0),
                                "groundedness": safe_rec.get("groundedness", 0.0),
                                "schema_validity": safe_rec.get("schema_validity", 0.0),
                                "latency_ms": safe_rec.get("latency_ms", 0.0),
                                "cost_usd": safe_rec.get("cost_usd", 0.0),
                                "model_used": safe_rec.get("model_used", ""),
                                "json_valid": bool(safe_rec.get("json_valid", False)),
                                "schema_valid": bool(safe_rec.get("schema_valid", False)),
                            }
                        )
                except Exception:
                    if verbose:
                        print("[Grid] Exception in run_config loop:")
                        print(traceback.format_exc())
                        print(
            f"[Grid][Types] routing_meta_type={type({}).__name__} "
            f"kb_retriever_type={type(kb_retriever).__name__} "
            f"ticket_type={type(tickets[0]).__name__ if isinstance(tickets, list) and tickets else 'n/a'} "
            f"models_type={type(models_loaded).__name__}"
        )
                        raise
                    raise
            release_local_models(models_loaded)

    df = pd.DataFrame(aggregate)
    exp_dir = RESULTS_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    if not df.empty:
        eval_analyzer.export_all_figures(df)

    release_local_models(models_loaded)

    return df


def main() -> None:
    """CLI entrypoint for running the grid."""
    parser = argparse.ArgumentParser(description="RouterGym grid runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickets")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (not used yet)")
    parser.add_argument("--force-llm", action="store_true", dest="force_llm", help="Force LLM for all routing/generation")
    parser.add_argument("--slm_subset", type=str, default=None, help="Comma-separated SLM keys to enable")
    args = parser.parse_args()

    slm_subset = [s.strip() for s in args.slm_subset.split(",")] if args.slm_subset else None

    try:
        if args.verbose:
            print("[Grid] Loading dataset...")
        tickets_loaded = dataset_loader.load_and_preprocess(DEFAULT_TICKETS_PATH, limit=args.limit) if hasattr(dataset_loader, "load_and_preprocess") else dataset_loader.load_dataset(args.limit)
        tickets = _coerce_tickets(tickets_loaded)
        if args.verbose:
            print(f"[Grid] Loaded tickets: {len(tickets)}")

        if args.verbose:
            print("[Grid] Loading KB...")
        kb_docs = kb_loader.load_kb()
        if args.verbose:
            print(f"[Grid] Loaded KB docs: {len(kb_docs)}")

        if args.verbose:
            print("[Grid] Loading models...")
        _ = load_models(sanity=False, slm_subset=slm_subset, force_llm=args.force_llm)

        df = run_full_grid(tickets=tickets, kb_retriever=kb_loader, limit=args.limit, verbose=args.verbose, force_llm=args.force_llm, slm_subset=slm_subset)
        out_dir = RESULTS_DIR / "experiments"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.csv"
        df.to_csv(out_path, index=False)
        if args.verbose:
            print(f"[Grid] Results saved to {out_path}")
    except Exception:  # pragma: no cover - CLI error handling
        print("[Grid] Fatal error during grid run:")
        print(traceback.format_exc())
        print(
            f"[Grid][Types] tickets_type={type(locals().get('tickets'))} "
            f"first_ticket={type(tickets[0]).__name__ if isinstance(tickets, list) and tickets else 'n/a'} "
            f"kb_type={type(kb_loader)}"
        )
        if args.verbose:
            raise


if __name__ == "__main__":
    main()
