"""Experimental grid runner with routers, memories, and KB."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import traceback
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from RouterGym.evaluation import analyzer as eval_analyzer
from RouterGym.evaluation import metrics as eval_metrics
from RouterGym.data.tickets import dataset_loader
from RouterGym.data.policy_kb import kb_loader
from RouterGym.engines.model_registry import load_models
from RouterGym.routing.llm_first import LLMFirstRouter
from RouterGym.routing.slm_dominant import SLMDominantRouter
from RouterGym.routing.hybrid_specialist import HybridSpecialistRouter
from RouterGym.memory.base import MemoryBase, MemoryRetrieval
from RouterGym.memory import MEMORY_MODES, get_memory_class, resolve_memory_mode
from RouterGym.routing.base import BaseRouter
from RouterGym.routing.router_engine import CLASSIFIER_MODES, RouterEngine
from RouterGym.agents.generator import CLASS_LABELS, infer_category_from_text, normalize_output
from RouterGym.contracts.schema_contract import SchemaContract


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
DEFAULT_TICKETS_PATH = Path("RouterGym/data/tickets/tickets.csv")
DEFAULT_KB_PATH = Path("RouterGym/data/policy_kb")

ROUTER_NAMES = ["llm_first", "slm_dominant", "hybrid_specialist"]
MODEL_NAMES = ["slm1", "slm2", "llm1", "llm2"]

RESULT_COLUMNS: Sequence[str] = (
    "router",
    "memory",
    "memory_mode",
    "model",
    "classifier_mode",
    "classifier_label",
    "classifier_confidence",
    "classifier_latency_ms",
    "classifier_token_cost",
    "classifier_accuracy",
    "classifier_efficiency_score",
    "ticket_id",
    "kb_attached",
    "result",
    "accuracy",
    "groundedness",
    "schema_validity",
    "latency_ms",
    "cost_usd",
    "retrieved_context_length",
    "retrieval_latency_ms",
    "memory_cost_tokens",
    "memory_relevance_score",
    "memory_context_used",
    "model_used",
    "json_valid",
    "schema_valid",
    "gold_category",
    "predicted_category",
    "classifier_metadata",
)


def _format_metadata(value: Any) -> str:
    """Serialize classifier metadata without commas for CSV highlighting."""
    if value in (None, "", {}, []):
        return "{}"
    if isinstance(value, str):
        text = value.strip().replace("\n", " ").replace("\r", " ")
        return text or "{}"
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            key_text = str(key).strip().replace("\n", " ").replace("\r", " ")
            if isinstance(item, (dict, list)):
                item_text = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
            else:
                item_text = str(item)
            item_text = item_text.strip().replace(",", ";").replace("\n", " ").replace("\r", " ")
            parts.append(f"{key_text}:{item_text}")
        return ";".join(parts) if parts else "{}"
    text = str(value).strip().replace(",", ";").replace("\n", " ").replace("\r", " ")
    return text or "{}"


def _clean_csv_value(column: str, value: Any) -> Any:
    """Sanitize individual values for flat CSV export."""
    if column == "classifier_metadata":
        return _format_metadata(value)
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace("\n", " ").replace("\r", " ")
    return text


def _records_to_rows(records: Iterable[Dict[str, Any]]) -> Iterable[List[Any]]:
    """Align records to the canonical column order for CSV output."""
    for record in records:
        yield [_clean_csv_value(column, record.get(column, "")) for column in RESULT_COLUMNS]


def write_results_csv(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write sanitized records to CSV with strict formatting rules."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_COLUMNS)
        for idx, row in enumerate(_records_to_rows(records), start=1):
            if len(row) != len(RESULT_COLUMNS):
                raise ValueError(
                    f"CSV row length mismatch at row {idx}: expected {len(RESULT_COLUMNS)} columns"
                )
            writer.writerow(row)


def _results_output_path(ticket_count: int) -> Path:
    """Return the appropriate CSV path based on ticket volume."""
    if ticket_count <= 50:
        return RESULTS_DIR / "results.csv"
    return RESULTS_DIR / "experiments" / f"results_{ticket_count}tickets.csv"


def _build_result_row(
    router_name: str,
    memory_mode: str,
    model_name: str,
    classifier_mode: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Project raw run outputs into the canonical 30-column schema."""
    row = {
        "router": router_name,
        "memory": memory_mode,
        "memory_mode": record.get("memory_mode", memory_mode),
        "model": model_name,
        "classifier_mode": record.get("classifier_mode", classifier_mode),
        "classifier_label": record.get("classifier_label", ""),
        "classifier_confidence": record.get("classifier_confidence", 0.0),
        "classifier_latency_ms": record.get("classifier_latency_ms", 0.0),
        "classifier_token_cost": record.get("classifier_token_cost", 0.0),
        "classifier_accuracy": record.get("classifier_accuracy", 0.0),
        "classifier_efficiency_score": record.get("classifier_efficiency_score", 0.0),
        "ticket_id": record.get("ticket_id"),
        "kb_attached": bool(record.get("kb_attached", False)),
        "result": record.get("result", "unknown"),
        "accuracy": record.get("accuracy", 0.0),
        "groundedness": record.get("groundedness", 0.0),
        "schema_validity": record.get("schema_validity", 0.0),
        "latency_ms": record.get("latency_ms", 0.0),
        "cost_usd": record.get("cost_usd", 0.0),
        "retrieved_context_length": record.get("retrieved_context_length", 0),
        "retrieval_latency_ms": record.get("retrieval_latency_ms", record.get("retrieval_time", 0.0)),
        "memory_cost_tokens": record.get("memory_cost_tokens", 0),
        "memory_relevance_score": record.get("memory_relevance_score", 0.0),
        "memory_context_used": record.get("memory_context", record.get("memory_context_used", "")),
        "model_used": record.get("model_used", ""),
        "json_valid": bool(record.get("json_valid", False)),
        "schema_valid": bool(record.get("schema_valid", False)),
        "gold_category": record.get("gold_category", ""),
        "predicted_category": record.get("predicted_category", ""),
        "classifier_metadata": record.get("classifier_metadata", {}),
    }
    # Guarantee the downstream writer sees every expected column.
    for column in RESULT_COLUMNS:
        row.setdefault(column, "")
    return row


def _norm_label(label: Any) -> str:
    """Normalize category labels."""
    text = str(label or "").strip().lower()
    return text or "unknown"


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
            label = _norm_label(row.get("category", row.get("label", "")))
            rec = {
                "id": row.get("id", idx),
                "text": row.get("text", row.get("document", "")),
                "category": label,
                "gold_category": label,
            }
            records.append(rec)
        return records
    if isinstance(tickets, list):
        for idx, t in enumerate(tickets):
            if isinstance(t, dict):
                label = _norm_label(t.get("category", t.get("label", "")))
                records.append(
                    {
                        "id": t.get("id", idx),
                        "text": t.get("text", ""),
                        "category": label,
                        "gold_category": label,
                    }
                )
            else:
                records.append({"id": idx, "text": str(t), "category": "unknown", "gold_category": "unknown"})
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
            label = _norm_label(row.get("label"))
            records.append({"id": idx, "text": row.get("text", ""), "category": label, "gold_category": label})
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
    canonical = resolve_memory_mode(name)
    memory_cls = get_memory_class(canonical)
    if memory_cls is None:
        return get_memory_class("none")()  # type: ignore[operator]
    return memory_cls()


def run_single(
    ticket: Dict[str, Any],
    router: Any,
    memory_mode: str,
    kb_retriever: Optional[Any],
    models: Optional[Dict[str, Any]],
    force_llm: bool = False,
    verbose: bool = False,
    router_engine: Optional[RouterEngine] = None,
    classifier_mode: str = "tfidf",
) -> Dict[str, Any]:
    """Run a single ticket through a router + memory + optional classifier."""
    memory_mode = resolve_memory_mode(memory_mode)
    start_total = time.perf_counter()
    active_mode = router_engine.classifier_mode if router_engine else classifier_mode
    if not isinstance(ticket, dict):
        base_record = {
            "ticket_id": None,
            "router": None,
            "memory": memory_mode,
            "memory_mode": memory_mode,
            "target_model": "unknown",
            "kb_attached": False,
            "routing_meta": {},
            "memory_context": "",
            "memory_relevance_score": 0.0,
            "memory_cost_tokens": 0,
            "memory_retrieval_metadata": {},
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
            "retrieval_latency_ms": 0.0,
            "retrieved_context_length": 0,
            "generation_time": 0.0,
            "repair_time": 0.0,
            "metrics_time": 0.0,
            "latency_escalated": False,
        }
        if router_engine:
            summary = router_engine.classify_ticket(ticket)
            base_record.update(summary.as_dict(active_mode))
        else:
            base_record["classifier_mode"] = active_mode
        return base_record
    memory = init_memory(memory_mode)
    memory.load(ticket)
    retrieval_result: MemoryRetrieval = memory.retrieve(ticket.get("text", ""))
    memory_context = retrieval_result.retrieved_context
    memory_relevance = retrieval_result.memory_relevance_score
    memory_cost_tokens = retrieval_result.memory_cost_tokens
    memory_metadata = retrieval_result.retrieval_metadata
    retrieval_latency_ms = retrieval_result.retrieval_latency_ms
    retrieved_context_length = retrieval_result.retrieved_context_length
    classifier_summary = (
        router_engine.classify_ticket(ticket, retrieval_result, memory_mode) if router_engine else None
    )
    try:
        t_route_start = time.perf_counter()
        router_kb = kb_retriever if memory_mode in {"rag_dense", "rag_bm25", "rag_hybrid"} else None
        routing_meta = router.route(ticket, kb=router_kb, models=models, memory=memory, force_llm=force_llm) if router else {}
        routing_meta = _as_dict(routing_meta, {"model_used": "unknown", "json_valid": False, "schema_valid": False})
        routing_time = (time.perf_counter() - t_route_start) * 1000
        t_gen_start = time.perf_counter()
        final_output = normalize_output(routing_meta.get("final_output", ""))
        generation_time = (time.perf_counter() - t_gen_start) * 1000
        schema = SchemaContract()
        schema_valid, _ = schema.validate(final_output)
        kb_texts = list(routing_meta.get("kb_snippets", [])) if isinstance(routing_meta, dict) else []
        kb_used_in_prompt = bool(kb_texts) and memory_mode in {"rag_dense", "rag_bm25", "rag_hybrid"}
        gold_category = _norm_label(ticket.get("gold_category", ticket.get("category", "")))
        predicted_category = _norm_label(final_output.get("predicted_category", routing_meta.get("predicted_category", "")))
        if predicted_category not in CLASS_LABELS or predicted_category == "unknown":
            predicted_category = infer_category_from_text(ticket.get("text", ""))
        elif predicted_category == "miscellaneous":
            inferred = infer_category_from_text(ticket.get("text", ""))
            if inferred in CLASS_LABELS and inferred != "miscellaneous":
                predicted_category = inferred
        record = {
            "ticket_id": ticket.get("id"),
            "router": routing_meta.get("strategy"),
            "memory": memory_mode,
            "target_model": routing_meta.get("target_model", "slm"),
            "kb_attached": kb_used_in_prompt,
            "routing_meta": routing_meta,
            "memory_context": memory_context,
            "memory_mode": memory_mode,
            "memory_relevance_score": memory_relevance,
            "memory_cost_tokens": memory_cost_tokens,
            "memory_retrieval_metadata": memory_metadata,
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
            "predicted_category": predicted_category,
            "gold_category": gold_category,
            "routing_time": routing_time,
            "retrieval_time": retrieval_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "retrieved_context_length": retrieved_context_length or len(memory_context),
            "generation_time": generation_time,
            "repair_time": 0.0,
            "metrics_time": 0.0,
            "latency_escalated": routing_meta.get("latency_escalated", False),
            "prompt": routing_meta.get("prompt", ""),
        }
        if classifier_summary:
            record.update(classifier_summary.as_dict(active_mode))
        else:
            record["classifier_mode"] = active_mode
        record["kb_snippets"] = kb_texts if kb_used_in_prompt else []
        memory.update(final_output.get("final_answer", ""), metadata=routing_meta)
        t_metrics_start = time.perf_counter()
        metric_values = eval_metrics.compute_all_metrics(
            {
                "output": record["output"],
                "gold_category": record.get("gold_category", ""),
                "predicted_category": record.get("predicted_category", ""),
                "kb_snippets": record["kb_snippets"],
                "model_used": record["model_used"],
                "latency_ms": record["latency_ms"],
                "kb_attached": record["kb_attached"],
                "prompt_text": record.get("prompt", ""),
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
                f"schema_valid={record['schema_valid']} cost={record['cost_usd']:.4f} "
                f"classifier={record.get('classifier_mode', active_mode)}"
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
    router_engine: Optional[RouterEngine] = None,
    classifier_mode: str = "tfidf",
) -> List[Dict[str, Any]]:
    """Run all tickets for a router/memory combo."""
    router = init_router(router_name)
    outputs: List[Dict[str, Any]] = []
    for ticket in tickets:
        try:
            outputs.append(
                run_single(
                    ticket,
                    router,
                    memory_mode,
                    kb_retriever,
                    models,
                    force_llm=force_llm,
                    verbose=verbose,
                    router_engine=router_engine,
                    classifier_mode=classifier_mode,
                )
            )
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
    classifier_modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run the full grid over routers, memories, models, and classifier modes."""
    tickets = _coerce_tickets(tickets if tickets is not None else load_tickets(limit=limit))
    ticket_count = len(tickets)
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
    raw_memory_list = memories or MEMORY_MODES
    memory_list = []
    for mem in raw_memory_list:
        canonical = resolve_memory_mode(mem)
        if canonical not in memory_list:
            memory_list.append(canonical)
    if models:
        model_list = models
    elif slm_subset:
        model_list = slm_subset
    else:
        model_list = ["llm1", "llm2"] if force_llm else MODEL_NAMES
    classifier_list = classifier_modes or CLASSIFIER_MODES

    for classifier_mode in classifier_list:
        engine = RouterEngine(classifier_mode)
        for router_name in router_list:
            for memory_mode in memory_list:
                for model_name in model_list:
                    if verbose:
                        print(
                            f"[Grid] Classifier={classifier_mode} Router={router_name} "
                            f"Memory={memory_mode} Model={model_name} Tickets={len(tickets)}"
                        )
                    try:
                        records = run_config(
                            router_name,
                            memory_mode,
                            tickets,
                            kb_retriever,
                            models_loaded,
                            force_llm=force_llm,
                            verbose=verbose,
                            router_engine=engine,
                            classifier_mode=classifier_mode,
                        )
                        raw_path = RAW_DIR / f"{router_name}__{memory_mode}__{model_name}__{classifier_mode}.jsonl"
                        with raw_path.open("w", encoding="utf-8") as f:
                            for rec in records:
                                safe_rec = _coerce_record(rec)
                                f.write(json.dumps(safe_rec) + "\n")
                        for rec in records:
                            safe_rec = _coerce_record(rec)
                            aggregate.append(
                                _build_result_row(
                                    router_name,
                                    memory_mode,
                                    model_name,
                                    classifier_mode,
                                    safe_rec,
                                )
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

    release_local_models(models_loaded)

    df = pd.DataFrame(aggregate)
    output_path = _results_output_path(ticket_count)
    write_results_csv(output_path, aggregate)

    if not df.empty:
        try:
            eval_analyzer.export_all_figures(df)
        except Exception:
            pass

    return df


def main() -> None:
    """CLI entrypoint for running the grid."""
    parser = argparse.ArgumentParser(description="RouterGym grid runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickets")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (not used yet)")
    parser.add_argument("--force-llm", action="store_true", dest="force_llm", help="Force LLM for all routing/generation")
    parser.add_argument("--slm_subset", type=str, default=None, help="Comma-separated SLM keys to enable")
    parser.add_argument(
        "--classifier-modes",
        type=str,
        default=None,
        help="Comma-separated classifier modes to enable (tfidf, encoder, slm_finetuned)",
    )
    args = parser.parse_args()

    slm_subset = [s.strip() for s in args.slm_subset.split(",")] if args.slm_subset else None
    classifier_modes = [s.strip() for s in args.classifier_modes.split(",")] if args.classifier_modes else None

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

        df = run_full_grid(
            tickets=tickets,
            kb_retriever=kb_loader,
            limit=args.limit,
            verbose=args.verbose,
            force_llm=args.force_llm,
            slm_subset=slm_subset,
            classifier_modes=classifier_modes,
        )
        if args.verbose:
            out_path = _results_output_path(len(tickets))
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
