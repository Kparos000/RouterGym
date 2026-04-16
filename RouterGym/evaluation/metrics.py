"""Evaluation metrics implementations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from RouterGym.engines.pricing import calculate_call_costs
from RouterGym.engines.telemetry import estimate_text_tokens
from RouterGym.label_space import canonical_label

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

EMBED_MODEL = "all-MiniLM-L6-v2"
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None and SentenceTransformer is not None:
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") not in {"0", "false", "False"}:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        try:
            _embedder = SentenceTransformer(EMBED_MODEL)
        except Exception:
            _embedder = None
    return _embedder


@dataclass
class MetricResult:
    """Container for metric scores."""

    groundedness: float | None = None
    schema_validity: float | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    fallback_rate: float | None = None
    accuracy: float | None = None
    faithfulness: float | None = None
    conversion: float | None = None
    step_coverage_score: float | None = None
    acceptance_criteria_alignment_score: float | None = None
    escalation_correctness_score: float | None = None
    policy_grounding_match_score: float | None = None
    overall_gold_quality_score: float | None = None


def _embed(texts: List[str]) -> np.ndarray:
    embedder = _get_embedder()
    if embedder is None or not texts:
        return np.zeros((0, 0), dtype="float32")
    return np.array(embedder.encode(texts), dtype="float32")


def schema_validity(pred: Any) -> int:
    """Return 1 if valid schema, else 0."""
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except Exception:
            return 0
    if not isinstance(pred, dict):
        return 0
    required = {"final_answer", "reasoning"}
    for field in required:
        if field not in pred:
            return 0
    return 1


def groundedness_score(answer: str, kb_snippets: List[str]) -> float:
    """Similarity between answer and KB snippets, normalized to [0, 1]."""
    if not kb_snippets or not answer:
        return 0.0
    ans_emb = _embed([answer])
    kb_emb = _embed(kb_snippets)
    if ans_emb.size == 0 or kb_emb.size == 0:
        # Fallback keyword overlap
        tokens = set(answer.lower().split())
        overlaps = []
        for snip in kb_snippets:
            snip_tokens = set(snip.lower().split())
            overlaps.append(len(tokens & snip_tokens) / max(len(snip_tokens), 1))
        return max(overlaps) if overlaps else 0.0
    norm_ans = ans_emb / (np.linalg.norm(ans_emb, axis=1, keepdims=True) + 1e-9)
    norm_kb = kb_emb / (np.linalg.norm(kb_emb, axis=1, keepdims=True) + 1e-9)
    sims = norm_kb @ norm_ans.T
    sims = np.clip(sims, -1.0, 1.0)
    normalized = (sims + 1.0) / 2.0
    flat = normalized.flatten()
    if flat.size == 0:
        return 0.0
    topk = np.sort(flat)[-min(3, flat.size) :]
    return float(topk.mean())


def faithfulness_score(reasoning: str, kb_snippets: List[str]) -> float:
    """Faithfulness based on reasoning similarity to KB."""
    return groundedness_score(reasoning, kb_snippets)


def semantic_correctness(answer: str, reference: Optional[str] = None) -> float:
    """Similarity to reference or default 0 if unavailable."""
    if reference is None:
        return 0.0
    return groundedness_score(answer, [reference])


def latency(start_time: float, end_time: float) -> float:
    """Latency in ms."""
    return max((end_time - start_time) * 1000.0, 0.0)


def _normalize_label(label: str) -> str:
    try:
        return canonical_label(label)
    except RuntimeError:
        return str(label).strip()


def estimate_tokens(*texts: str) -> int:
    """Rough token estimate based on character length."""
    return sum(estimate_text_tokens(text) for text in texts)


def estimate_cost_usd(model_used: str, prompt_text: str, answer_text: str, reasoning: str = "") -> float:
    """Estimate cost per call using normalized benchmark pricing."""
    input_tokens = estimate_tokens(prompt_text)
    output_tokens = estimate_tokens(answer_text, reasoning)
    return float(calculate_call_costs(model_used, input_tokens, output_tokens)["total_cost_usd"])


def router_conversion_rate(router_stats: List[Dict[str, Any]]) -> float:
    """Compute % handled without LLM fallback."""
    if not router_stats:
        return 0.0
    handled = sum(1 for r in router_stats if r.get("model_used") != "llm")
    return handled / len(router_stats)


def compute_all_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    """Compute all metrics from a record.

    gold_category comes from the dataset; predicted_category comes from the model output.
    accuracy is an exact match between normalized predicted_category and gold_category
    (unknown/other counts as incorrect). Groundedness compares the answer to KB snippets
    that were actually injected into the prompt for this ticket.
    """
    output_raw = record.get("output", "")
    output = output_raw.get("final_answer") if isinstance(output_raw, dict) else output_raw
    label = _normalize_label(str(record.get("gold_category", record.get("label", ""))))
    predicted_label = _normalize_label(str(record.get("predicted_category", record.get("predicted", ""))))
    kb_snippets = record.get("kb_snippets", [])
    model_used = str(record.get("model_used", "slm")).lower()
    reasoning = ""
    if isinstance(output_raw, dict):
        reasoning = output_raw.get("reasoning", "")
    prompt_text = record.get("prompt_text", record.get("prompt", ""))
    kb_attached = bool(record.get("kb_attached", False))

    acc = 1.0 if predicted_label and predicted_label == label and predicted_label not in {"unknown", "other"} else 0.0
    grounded = 0.0 if not kb_attached else groundedness_score(str(output), kb_snippets)
    faithful = groundedness_score(reasoning or str(output), kb_snippets) if kb_attached else 0.0
    schema_val = schema_validity(record.get("parsed_output", record.get("output", {})))
    metrics_blob = record.get("metrics", {})
    if isinstance(metrics_blob, dict) and isinstance(metrics_blob.get("total_cost_usd"), (int, float)):
        cost = float(metrics_blob["total_cost_usd"])
    elif isinstance(record.get("total_cost_usd"), (int, float)):
        cost = float(record["total_cost_usd"])
    else:
        model_key = str(record.get("model_name") or record.get("model_used") or model_used)
        cost = estimate_cost_usd(model_key, str(prompt_text), str(output), reasoning)

    metrics = {
        "accuracy": acc,
        "groundedness": grounded,
        "schema_validity": schema_val,
        "latency": record.get("latency_ms", 0.0),
        "cost": cost,
        "faithfulness": faithful,
    }
    gold_record = record.get("gold_record") or record.get("gold_reference")
    if isinstance(gold_record, dict):
        from RouterGym.evaluation.gold_scoring import score_record_against_gold

        gold_scores = score_record_against_gold(record, gold_record).as_dict()
        for field in (
            "step_coverage_score",
            "acceptance_criteria_alignment_score",
            "escalation_correctness_score",
            "policy_grounding_match_score",
            "overall_gold_quality_score",
        ):
            value = gold_scores.get(field)
            if isinstance(value, (int, float)):
                metrics[field] = float(value)
    return metrics


def evaluate(outputs: Dict[str, Any]) -> MetricResult:
    """Wrapper to compute MetricResult from outputs dict."""
    metrics = compute_all_metrics(outputs)
    return MetricResult(
        accuracy=metrics["accuracy"],
        groundedness=metrics["groundedness"],
        schema_validity=metrics["schema_validity"],
        latency_ms=outputs.get("latency_ms"),
        cost_usd=metrics["cost"],
        fallback_rate=outputs.get("fallback_rate"),
        faithfulness=metrics["faithfulness"],
        step_coverage_score=metrics.get("step_coverage_score"),
        acceptance_criteria_alignment_score=metrics.get("acceptance_criteria_alignment_score"),
        escalation_correctness_score=metrics.get("escalation_correctness_score"),
        policy_grounding_match_score=metrics.get("policy_grounding_match_score"),
        overall_gold_quality_score=metrics.get("overall_gold_quality_score"),
    )
