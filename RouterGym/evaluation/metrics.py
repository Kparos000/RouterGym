"""Evaluation metrics implementations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

EMBED_MODEL = "all-MiniLM-L6-v2"
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None and SentenceTransformer is not None:
        _embedder = SentenceTransformer(EMBED_MODEL)
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
    """Max similarity between answer and KB snippets."""
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
    return float(sims.max())


def faithfulness_score(reasoning: str, kb_snippets: List[str]) -> float:
    """Faithfulness based on reasoning similarity to KB."""
    return groundedness_score(reasoning, kb_snippets)


def classification_f1(true_label: str, predicted_label: str) -> float:
    """Binary F1 for single label."""
    if not true_label and not predicted_label:
        return 1.0
    if true_label == predicted_label:
        return 1.0
    return 0.0


def semantic_correctness(answer: str, reference: Optional[str] = None) -> float:
    """Similarity to reference or default 0 if unavailable."""
    if reference is None:
        return 0.0
    return groundedness_score(answer, [reference])


COST_FACTOR = {"slm": 0.0001, "llm": 0.001}


def token_cost(model_name: str, output_text: str) -> float:
    """Estimate token cost by model family."""
    tokens = len(output_text.split())
    family = "slm" if model_name.startswith("slm") else "llm"
    rate = COST_FACTOR.get(family, 0.001)
    return rate * tokens


def latency(start_time: float, end_time: float) -> float:
    """Latency in ms."""
    return max((end_time - start_time) * 1000.0, 0.0)


def router_conversion_rate(router_stats: List[Dict[str, Any]]) -> float:
    """Compute % handled without LLM fallback."""
    if not router_stats:
        return 0.0
    handled = sum(1 for r in router_stats if r.get("model_used") != "llm")
    return handled / len(router_stats)


def compute_all_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    """Compute all metrics from a record."""
    output_raw = record.get("output", "")
    output = output_raw.get("final_answer") if isinstance(output_raw, dict) else output_raw
    label = record.get("label", "")
    kb_snippets = record.get("kb_snippets", [])
    model_name = record.get("model_used", "slm")
    reasoning = ""
    if isinstance(output_raw, dict):
        reasoning = output_raw.get("reasoning", "")

    acc = classification_f1(label, record.get("predicted", label))
    grounded = groundedness_score(str(output), kb_snippets)
    faithful = faithfulness_score(reasoning or str(output), kb_snippets)
    schema_val = schema_validity(record.get("parsed_output", record.get("output", {})))
    cost = token_cost(model_name, str(output))

    return {
        "accuracy": acc,
        "groundedness": grounded,
        "schema_validity": schema_val,
        "latency": record.get("latency_ms", 0.0),
        "cost": cost,
        "faithfulness": faithful,
    }


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
    )
