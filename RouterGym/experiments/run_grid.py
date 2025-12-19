"""Experimental grid runner with routers, memories, and KB."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import traceback
import time
from datetime import datetime, timezone
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
from RouterGym.memory import get_memory_class, resolve_memory_mode
from RouterGym.routing.base import BaseRouter
from RouterGym.routing.router_engine import RouterEngine
from RouterGym.agents.generator import CLASS_LABELS, infer_category_from_text, normalize_output
from RouterGym.contracts.schema_contract import SchemaContract
from RouterGym.engines.model_registry import get_model_backend
from RouterGym.label_space import canonical_label


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
DEFAULT_TICKETS_PATH = Path("RouterGym/data/tickets/tickets.csv")
DEFAULT_KB_PATH = Path("RouterGym/data/policy_kb")

ROUTER_CONFIGS = [
    # SLM-only
    {"router_mode": "slm_only", "base_model": "slm1"},
    {"router_mode": "slm_only", "base_model": "slm2"},
    # LLM-only
    {"router_mode": "llm_only", "base_model": "llm1"},
    {"router_mode": "llm_only", "base_model": "llm2"},
    # SLM-dominant (all SLM→LLM pairs)
    {"router_mode": "slm_dominant", "base_model": "slm1", "escalation_model": "llm1"},
    {"router_mode": "slm_dominant", "base_model": "slm1", "escalation_model": "llm2"},
    {"router_mode": "slm_dominant", "base_model": "slm2", "escalation_model": "llm1"},
    {"router_mode": "slm_dominant", "base_model": "slm2", "escalation_model": "llm2"},
    # Hybrid specialist placeholder
    {"router_mode": "hybrid_specialist", "base_model": "slm1"},
]

MEMORY_MODES_CANONICAL = ["none", "rag_bm25", "rag_dense", "rag_hybrid"]
MODEL_NAMES = ["slm1", "slm2", "llm1", "llm2"]
CLASSIFIER_MODES = ["tfidf", "encoder", "slm_finetuned"]
ROUTER_MODE_NAMES = sorted({cfg["router_mode"] for cfg in ROUTER_CONFIGS})

RESULT_COLUMNS: Sequence[str] = (
    "router",
    "memory",
    "memory_mode",
    "model",
    "router_confidence_score",
    "router_decision_reason",
    "classifier_mode",
    "classifier_backend",
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
    "llm_category",
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


def _results_output_path(ticket_count: int, override: Optional[Path] = None) -> Path:
    """Return the appropriate CSV path based on ticket volume or explicit override."""
    if override:
        return override
    if ticket_count <= 50:
        return RESULTS_DIR / "results.csv"
    return RESULTS_DIR / "experiments" / f"results_{ticket_count}tickets.csv"


def log_run_metadata(metadata: Dict[str, Any], log_path: Optional[Path] = None) -> None:
    """Append run metadata to CSV; swallow IO errors."""
    log_file = log_path or (RESULTS_DIR / "run_metadata.csv")
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_file.exists()
        with log_file.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            header = [
                "timestamp",
                "backend",
                "ticket_start",
                "ticket_limit",
                "num_tickets",
                "routers",
                "memories",
                "classifiers",
                "models",
                "output_path",
                "wall_clock_seconds",
            ]
            if write_header:
                writer.writerow(header)
            writer.writerow([metadata.get(col, "") for col in header])
    except Exception:
        print(f"[Metadata] Warning: failed to write run metadata to {log_file}")


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
        "router_confidence_score": record.get("router_confidence_score", 0.0),
        "router_decision_reason": record.get("router_decision_reason", ""),
        "classifier_mode": record.get("classifier_mode", classifier_mode),
        "classifier_backend": record.get("classifier_backend", record.get("classifier_mode", classifier_mode)),
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
        "llm_category": record.get("llm_category", ""),
        "predicted_category": record.get("predicted_category", ""),
        "classifier_metadata": record.get("classifier_metadata", {}),
    }
    # Guarantee the downstream writer sees every expected column.
    for column in RESULT_COLUMNS:
        row.setdefault(column, "")
    return row


def _norm_label(label: Any) -> str:
    """Normalize category labels into the canonical label space."""
    try:
        return canonical_label(str(label or ""))
    except RuntimeError:
        return "Miscellaneous"


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
        "llm_only": LLMFirstRouter(),
        "slm_only": HybridSpecialistRouter(),
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
            "router_confidence_score": 0.0,
            "router_decision_reason": "",
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
            "llm_category": "",
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
        llm_category = predicted_category
        record = {
            "ticket_id": ticket.get("id"),
            "router": routing_meta.get("strategy"),
            "memory": memory_mode,
            "target_model": routing_meta.get("target_model", "slm"),
            "router_confidence_score": routing_meta.get("router_confidence_score", 0.0),
            "router_decision_reason": routing_meta.get("router_decision_reason", ""),
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
            "llm_category": llm_category,
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
            record["classifier_backend"] = active_mode
        classifier_label = record.get("classifier_label", "")
        record["llm_category"] = llm_category
        record["predicted_category"] = classifier_label or llm_category
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
    ticket_start: int = 0,
    ticket_limit: int = -1,
    routers: Optional[List[str]] = None,
    memories: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    kb_retriever: Optional[Any] = None,
    verbose: bool = False,
    force_llm: bool = False,
    slm_subset: Optional[List[str]] = None,
    classifier_modes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full grid over routers, memories, models, and classifier modes."""
    t_start = time.time()
    all_tickets = _coerce_tickets(tickets if tickets is not None else load_tickets(limit=limit))
    start = max(ticket_start, 0)
    end = start + ticket_limit if ticket_limit is not None and ticket_limit >= 0 else len(all_tickets)
    tickets = all_tickets[start:end]
    ticket_count = len(tickets)
    kb_retriever = kb_retriever if kb_retriever is not None else load_kb()
    models_loaded: Optional[Dict[str, Any]] = None
    try:
        models_loaded = load_models(sanity=False, slm_subset=slm_subset, force_llm=force_llm)
    except Exception:
        models_loaded = None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    streaming_output = ticket_count > 50
    output_path = _results_output_path(ticket_count, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate: List[Dict[str, Any]] = []

    router_list = ROUTER_CONFIGS if routers is None else [cfg for cfg in ROUTER_CONFIGS if cfg["router_mode"] in routers]
    if models:
        filtered = [
            cfg
            for cfg in router_list
            if (cfg.get("base_model") in models) or (cfg.get("escalation_model") in models)
        ]
        router_list = filtered or router_list
    raw_memory_list = memories or MEMORY_MODES_CANONICAL
    memory_list = []
    for mem in raw_memory_list:
        canonical = resolve_memory_mode(mem)
        if canonical not in memory_list:
            memory_list.append(canonical)
    classifier_list = classifier_modes or CLASSIFIER_MODES
    if models:
        model_list = models
    else:
        model_set = set()
        for cfg in router_list:
            base_model = cfg.get("base_model")
            escalation_model = cfg.get("escalation_model")
            if base_model:
                model_set.add(base_model)
            if escalation_model:
                model_set.add(escalation_model)
        model_list = list(model_set)

    with output_path.open("w", encoding="utf-8", newline="") as csv_handle:
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(RESULT_COLUMNS)

        for classifier_mode in classifier_list:
            encoder_use_lexical_prior = None
            engine = RouterEngine(classifier_mode, encoder_use_lexical_prior=encoder_use_lexical_prior)
            for router_cfg in router_list:
                router_name = router_cfg["router_mode"]
                base_model = router_cfg.get("base_model", "")
                escalation_model = router_cfg.get("escalation_model")
                for memory_mode in memory_list:
                    if verbose:
                        print(
                            f"[Grid] Classifier={classifier_mode} Router={router_name} "
                            f"Memory={memory_mode} BaseModel={base_model} Tickets={len(tickets)}"
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
                        esc_suffix = f"__esc_{escalation_model}" if escalation_model else ""
                        raw_path = RAW_DIR / f"{router_name}__base_{base_model}{esc_suffix}__{memory_mode}__{classifier_mode}.jsonl"
                        with raw_path.open("w", encoding="utf-8") as f:
                            for rec in records:
                                safe_rec = _coerce_record(rec)
                                f.write(json.dumps(safe_rec) + "\n")
                        for rec in records:
                            safe_rec = _coerce_record(rec)
                            row_dict = _build_result_row(
                                router_name,
                                memory_mode,
                                base_model,
                                classifier_mode,
                                safe_rec,
                            )
                            row_dict.setdefault("base_model_name", base_model)
                            row_dict.setdefault("escalation_model_name", escalation_model or "")
                            csv_writer.writerow([_clean_csv_value(col, row_dict.get(col, "")) for col in RESULT_COLUMNS])
                            if not streaming_output:
                                aggregate.append(row_dict)
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

    if streaming_output:
        df = pd.read_csv(output_path)
    else:
        df = pd.DataFrame(aggregate)

    try:
        log_run_metadata(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "backend": get_model_backend(),
                "ticket_start": ticket_start,
                "ticket_limit": ticket_limit,
                "num_tickets": ticket_count,
                "routers": "|".join(cfg["router_mode"] for cfg in router_list),
                "memories": "|".join(memory_list),
                "classifiers": "|".join(classifier_list),
                "models": "|".join(model_list),
                "output_path": str(output_path),
                "wall_clock_seconds": time.time() - t_start,
            }
        )
    except Exception:
        print("[Metadata] Warning: failed to log run metadata.")

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
    parser.add_argument("--ticket-start", type=int, default=0, help="0-based start index into tickets.csv")
    parser.add_argument("--ticket-limit", type=int, default=-1, help="Max tickets from start; -1 means all remaining")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (not used yet)")
    parser.add_argument("--force-llm", action="store_true", dest="force_llm", help="Force LLM for all routing/generation")
    parser.add_argument("--slm_subset", type=str, default=None, help=f"Comma-separated SLM keys to enable (default all: {', '.join(MODEL_NAMES)})")
    parser.add_argument(
        "--classifier-modes",
        type=str,
        default=None,
        help="Comma-separated classifier modes to enable (tfidf, encoder, slm_finetuned)",
    )
    parser.add_argument(
        "--routers",
        type=str,
        default=None,
        help=f"Comma-separated routers (default: {', '.join(ROUTER_MODE_NAMES)})",
    )
    parser.add_argument(
        "--memories",
        type=str,
        default=None,
        help=f"Comma-separated memory modes (default: {', '.join(MEMORY_MODES_CANONICAL)})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated models (default: {', '.join(MODEL_NAMES)})",
    )
    parser.add_argument(
        "--smoke-10",
        action="store_true",
        dest="smoke10",
        help="Run a 10-ticket smoke grid (overwrites results.csv; defaults to models=slm1).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit CSV output path (overrides size-based default).",
    )
    args = parser.parse_args()

    slm_subset = [s.strip() for s in args.slm_subset.split(",")] if args.slm_subset else None
    classifier_modes = [s.strip() for s in args.classifier_modes.split(",")] if args.classifier_modes else None
    routers = [s.strip() for s in args.routers.split(",")] if args.routers else None
    memories = [s.strip() for s in args.memories.split(",")] if args.memories else None
    models = [s.strip() for s in args.models.split(",")] if args.models else None
    output_path = Path(args.output_path) if args.output_path else None

    try:
        if args.verbose:
            print("[Grid] Loading dataset...")
        tickets_loaded = dataset_loader.load_and_preprocess(DEFAULT_TICKETS_PATH, limit=args.limit) if hasattr(dataset_loader, "load_and_preprocess") else dataset_loader.load_dataset(args.limit)
        tickets = _coerce_tickets(tickets_loaded)
        if args.smoke10:
            tickets = tickets[:10]
            if args.verbose:
                print(
                    f"[Smoke] Running 10-ticket grid with routers={routers or ROUTER_MODE_NAMES}, "
                    f"memories={memories or MEMORY_MODES_CANONICAL}, "
                    f"classifiers={classifier_modes or CLASSIFIER_MODES}, "
                    f"models={models or ['slm1']}"
                )
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
            ticket_start=args.ticket_start,
            ticket_limit=args.ticket_limit,
            verbose=args.verbose,
            force_llm=args.force_llm,
            slm_subset=slm_subset,
            classifier_modes=classifier_modes,
            routers=routers,
            memories=memories,
            models=models if not args.smoke10 else (models or ["slm1"]),
            output_path=output_path,
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
