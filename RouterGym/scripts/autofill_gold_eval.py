"""Automate gold eval filling using KB-grounded LLM drafting and review."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from RouterGym.data.policy_kb import kb_loader
from RouterGym.engines.model_registry import load_models
from RouterGym.label_space import canonicalize_label
from RouterGym.memory import MEMORY_MODES, get_memory_class

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "gold_eval" / "gold_eval.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "gold_eval" / "gold_eval_auto.jsonl"
DEFAULT_REVIEW = Path(__file__).resolve().parents[1] / "data" / "gold_eval" / "gold_eval_review_queue.jsonl"
DEFAULT_MEMORY_MODE = "rag_hybrid"
DEFAULT_TOP_K = 4
DEFAULT_ANNOTATOR = "llm2"
DEFAULT_REVIEWER = "llm1"
DEFAULT_SEED = 42

ALLOWED_MEMORY_MODES = set(MEMORY_MODES + ["rag", "salience"])
FLUFF_PHRASES = [
    "i can't access your system",
    "i cannot access your system",
    "i can't log in to your system",
    "as an ai",
    "as a language model",
    "cannot perform actions",
]
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found at {path}. Run build_gold_eval_template first to create it."
        )
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


@lru_cache(maxsize=1)
def _kb_index_by_id() -> Dict[str, Dict[str, Any]]:
    try:
        articles = kb_loader.load_kb_index()
    except Exception:
        return {}
    return {str(article.get("id", "")): dict(article) for article in articles if article.get("id")}


def _valid_policy_ids() -> Set[str]:
    return set(_kb_index_by_id().keys())


def _tokenize(text: Any) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text or ""))]


def _token_overlap_score(query_tokens: Sequence[str], text: Any) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = set(_tokenize(text))
    if not doc_tokens:
        return 0.0
    return float(len(set(query_tokens) & doc_tokens))


def _rank_kb_entries(
    query_tokens: Sequence[str],
    entries: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Tuple[float, Dict[str, Any]]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in entries:
        text = f"{entry.get('title', '')} {entry.get('content', '')}"
        score = _token_overlap_score(query_tokens, text)
        scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))
    return scored[: max(top_k, 0)]


def _entry_to_snippet(entry: Dict[str, Any], score: float) -> Dict[str, Any]:
    return {
        "policy_id": str(entry.get("id", "")),
        "category": str(entry.get("category", "")),
        "title": str(entry.get("title", "")),
        "text": str(entry.get("content", "")),
        "score": float(score),
        "escalation_notes": str(entry.get("escalation_notes", "")),
        "path": str(entry.get("path", "")),
    }


def _fallback_policy_selection(
    ticket_text: str,
    topic_group: str,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[str], str | None]:
    kb_map = _kb_index_by_id()
    if not kb_map:
        return [], [], "kb_index_empty_or_unreadable"
    entries = list(kb_map.values())
    query_tokens = _tokenize(ticket_text)
    category_entries = [entry for entry in entries if entry.get("category") == topic_group]
    scored = _rank_kb_entries(query_tokens, category_entries, top_k) if category_entries else []
    if not scored:
        scored = _rank_kb_entries(query_tokens, entries, top_k)
    if not scored:
        return [], [], "kb_index_empty_or_unreadable"
    snippets = [_entry_to_snippet(entry, score) for score, entry in scored]
    policy_ids = [snippet["policy_id"] for snippet in snippets if snippet.get("policy_id")]
    return snippets, policy_ids, None


def _extract_snippets(metadata: Any) -> List[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []
    if "fused_snippets" in metadata and isinstance(metadata.get("fused_snippets"), list):
        return [s for s in metadata.get("fused_snippets", []) if isinstance(s, dict)]
    if "snippets" in metadata and isinstance(metadata.get("snippets"), list):
        return [s for s in metadata.get("snippets", []) if isinstance(s, dict)]
    return []


def _extract_policy_id(snippet: Dict[str, Any], allowed_ids: Set[str]) -> str:
    candidates = [
        snippet.get("policy_id"),
        snippet.get("id"),
        snippet.get("source"),
        snippet.get("path"),
    ]
    for cand in candidates:
        if cand is None:
            continue
        value = str(cand).strip()
        if not value:
            continue
        if value.startswith("dynamic:"):
            continue
        if value in allowed_ids:
            return value
    return ""


def _normalize_snippet(snippet: Dict[str, Any], allowed_ids: Set[str]) -> Dict[str, Any] | None:
    policy_id = _extract_policy_id(snippet, allowed_ids)
    if not policy_id:
        return None
    kb_map = _kb_index_by_id()
    article = kb_map.get(policy_id, {})
    return {
        "policy_id": policy_id,
        "category": str(article.get("category", "") or snippet.get("category", "")).strip(),
        "title": str(article.get("title", "") or snippet.get("title", "")).strip(),
        "text": str(snippet.get("text", "") or article.get("content", "")).strip(),
        "score": float(snippet.get("score", 0.0) or 0.0),
        "escalation_notes": str(article.get("escalation_notes", "")).strip(),
        "path": str(article.get("path", "")).strip(),
    }


def retrieve_kb(ticket_text: str, memory_mode: str, top_k: int) -> Dict[str, Any]:
    """Retrieve KB snippets using configured memory backend."""
    mode = memory_mode
    if mode not in ALLOWED_MEMORY_MODES:
        raise ValueError(f"Unsupported memory_mode '{mode}'. Allowed: {sorted(ALLOWED_MEMORY_MODES)}")
    memory_cls = get_memory_class(mode)
    if memory_cls is None:
        raise ValueError(f"No memory class found for mode '{mode}'")
    try:
        memory = memory_cls(top_k=top_k)  # type: ignore[arg-type]
    except TypeError:
        memory = memory_cls()  # type: ignore[call-arg]
    # Do not update memory with ticket text; use KB-only index.
    retrieval = memory.retrieve(ticket_text)
    raw_snippets = _extract_snippets(retrieval.retrieval_metadata)
    allowed_ids = _valid_policy_ids()
    snippets = []
    for raw in raw_snippets:
        normalized = _normalize_snippet(raw, allowed_ids)
        if normalized:
            snippets.append(normalized)
    policy_ids = [s["policy_id"] for s in snippets if s.get("policy_id")]
    return {
        "context": retrieval.retrieved_context or "",
        "snippets": snippets,
        "policy_ids": policy_ids,
    }


def _format_snippets_for_prompt(snippets: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for idx, snip in enumerate(snippets, start=1):
        policy_id = snip.get("policy_id", "")
        title = snip.get("title", "")
        category = snip.get("category", "")
        esc = snip.get("escalation_notes", "")
        text = snip.get("text", "")
        score = snip.get("score", 0.0)
        lines = [
            f"KB #{idx} [{policy_id}] (score={score:.2f})",
            f"Category: {category}",
            f"Title: {title}",
        ]
        if esc:
            lines.append(f"Escalation notes: {esc}")
        lines.append(f"Content: {text}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def build_annotator_prompt(
    ticket_text: str,
    topic_group: str,
    snippets: Sequence[Dict[str, Any]],
    allowed_policy_ids: Sequence[str],
) -> str:
    snippets_text = _format_snippets_for_prompt(snippets) or "No KB snippets found."
    allowed = ", ".join(allowed_policy_ids) if allowed_policy_ids else "none"
    return "\n".join(
        [
            "You are generating gold resolutions for IT tickets using only the provided KB snippets.",
            f"Ticket (Topic={topic_group}):",
            ticket_text.strip(),
            "",
            "KB snippets (from policy_kb_index.json):",
            snippets_text,
            "",
            "Rules:",
            "- Use ONLY the KB snippets above; do not invent procedures.",
            "- If KB is insufficient, set escalation_required=true and explain why in escalation_reason.",
            "- steps must be concrete, numbered actions (3 to 8 items).",
            "- kb_policies must be a subset of the retrieved policy_ids: " + allowed,
            "- acceptance_criteria must contain at least 2 items describing verification.",
            "",
            "Output STRICT JSON with fields only:",
            '{"summary": "...", "steps": ["..."], "escalation_required": true/false, '
            '"escalation_reason": "...", "kb_policies": ["policy_id"], "acceptance_criteria": ["..."]}',
        ]
    )


def build_reviewer_prompt(
    ticket_text: str,
    topic_group: str,
    snippets: Sequence[Dict[str, Any]],
    allowed_policy_ids: Sequence[str],
    annotator_output: Dict[str, Any],
) -> str:
    snippets_text = _format_snippets_for_prompt(snippets) or "No KB snippets found."
    allowed = ", ".join(allowed_policy_ids) if allowed_policy_ids else "none"
    annotator_json = json.dumps(annotator_output, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "You are reviewing and correcting a draft gold resolution.",
            f"Ticket (Topic={topic_group}):",
            ticket_text.strip(),
            "",
            "KB snippets (from policy_kb_index.json):",
            snippets_text,
            "",
            "Draft gold_resolution JSON from annotator:",
            annotator_json,
            "",
            "Tasks:",
            "- Fix gaps, ensure clarity, and keep steps concrete (3-8 items).",
            "- Ensure kb_policies ONLY reference retrieved policy_ids: " + allowed,
            "- If KB is insufficient, set escalation_required=true and explain why.",
            "- Provide acceptance_criteria with at least 2 items.",
            "",
            "Output corrected JSON with the same fields only:",
            '{"summary": "...", "steps": ["..."], "escalation_required": true/false, '
            '"escalation_reason": "...", "kb_policies": ["policy_id"], "acceptance_criteria": ["..."]}',
        ]
    )


def _format_allowed_policies(snippets: Sequence[Dict[str, Any]], allowed_policy_ids: Sequence[str]) -> str:
    allowed_set = set(allowed_policy_ids)
    parts = []
    for snip in snippets:
        pid = str(snip.get("policy_id", "")).strip()
        if not pid or pid not in allowed_set:
            continue
        title = str(snip.get("title", "")).strip()
        text = str(snip.get("text", "")).strip()
        excerpt = text[:300] + ("..." if len(text) > 300 else "")
        parts.append(f"- {pid}: {title}\n  Excerpt: {excerpt}")
    return "\n".join(parts) if parts else "none"


def build_repair_prompt(
    ticket_text: str,
    topic_group: str,
    snippets: Sequence[Dict[str, Any]],
    allowed_policy_ids: Sequence[str],
    prior_output: Dict[str, Any],
    validation_errors: Sequence[str],
) -> str:
    allowed_block = _format_allowed_policies(snippets, allowed_policy_ids)
    errors = ", ".join(validation_errors) if validation_errors else "unknown"
    prior_json = json.dumps(prior_output, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "You must repair the gold_resolution JSON using only the allowed KB policies.",
            f"Ticket (Topic={topic_group}):",
            ticket_text.strip(),
            "",
            "Allowed KB policies (id, title, excerpt):",
            allowed_block,
            "",
            f"Validation errors: {errors}",
            "",
            "Prior output:",
            prior_json,
            "",
            "Instructions:",
            "- Fix all validation errors and return corrected JSON only.",
            "- steps must be 3-8 actionable items.",
            "- acceptance_criteria must contain at least 2 items.",
            "- kb_policies must be a subset of allowed policy_ids above.",
            "",
            "Return corrected JSON only:",
            '{"summary": "...", "steps": ["..."], "escalation_required": true/false, '
            '"escalation_reason": "...", "kb_policies": ["policy_id"], "acceptance_criteria": ["..."]}',
        ]
    )


def call_model(model: Any, prompt: str) -> str:
    if model is None:
        return ""
    if hasattr(model, "generate"):
        try:
            return model.generate(prompt, max_new_tokens=384, temperature=0.2)
        except TypeError:
            return model.generate(prompt)  # type: ignore[call-arg]
    if callable(model):
        try:
            return model(prompt, max_new_tokens=384, temperature=0.2)
        except TypeError:
            return model(prompt)
    return ""


def parse_json_safely(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _coerce_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    return cleaned


def normalize_gold_resolution(resolution: Dict[str, Any] | None) -> Dict[str, Any]:
    res = resolution or {}
    summary = str(res.get("summary", "")).strip()
    steps = _coerce_list(res.get("steps", []))
    escalation_required = bool(res.get("escalation_required", False))
    escalation_reason = str(res.get("escalation_reason", "")).strip()
    kb_policies = _coerce_list(res.get("kb_policies", []))
    acceptance = _coerce_list(res.get("acceptance_criteria", []))
    return {
        "summary": summary,
        "steps": steps,
        "escalation_required": escalation_required,
        "escalation_reason": escalation_reason,
        "kb_policies": kb_policies,
        "acceptance_criteria": acceptance,
    }


def _is_actionable_step(step: Any) -> bool:
    text = str(step or "").strip().lower()
    if not text:
        return False
    if any(phrase in text for phrase in FLUFF_PHRASES):
        return False
    tokens = _tokenize(text)
    return len(tokens) >= 2


def validate_gold_resolution(resolution: Dict[str, Any], allowed_policy_ids: Iterable[str]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    allowed = set(allowed_policy_ids)
    if not resolution:
        return False, ["empty_resolution"]

    summary = resolution.get("summary", "")
    if not summary or str(summary).strip().lower() == "todo":
        reasons.append("summary_missing")

    steps = resolution.get("steps", [])
    if not isinstance(steps, list) or len(steps) < 3 or len(steps) > 8:
        reasons.append("steps_count_out_of_range")
    elif any(not _is_actionable_step(step) for step in steps):
        reasons.append("steps_not_actionable")

    acceptance = resolution.get("acceptance_criteria", [])
    if not isinstance(acceptance, list) or len(acceptance) < 2:
        reasons.append("empty_acceptance_criteria")

    kb_policies = resolution.get("kb_policies", [])
    if not isinstance(kb_policies, list) or len(kb_policies) == 0:
        reasons.append("kb_policies_missing")
    else:
        invalid_ids = [pid for pid in kb_policies if pid not in allowed]
        if invalid_ids:
            reasons.append(f"kb_policy_not_retrieved:{','.join(sorted(set(invalid_ids)))}")

    escalation_required = bool(resolution.get("escalation_required", False))
    escalation_reason = str(resolution.get("escalation_reason", "")).strip()
    if escalation_required and not escalation_reason:
        reasons.append("escalation_reason_missing")

    text_blob = f"{summary} {' '.join(steps if isinstance(steps, list) else [])} {' '.join(acceptance if isinstance(acceptance, list) else [])}".lower()
    if any(phrase in text_blob for phrase in FLUFF_PHRASES):
        reasons.append("non_actionable_fluff")

    return len(reasons) == 0, reasons


def _extract_resolution(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "gold_resolution" in obj and isinstance(obj["gold_resolution"], dict):
        return obj["gold_resolution"]
    return obj


def process_record(
    record: Dict[str, Any],
    models: Dict[str, Any],
    memory_mode: str,
    top_k: int,
    annotator_model: str,
    reviewer_model: str,
) -> Dict[str, Any]:
    ticket_text = str(record.get("ticket_text", "")).strip()
    topic_group = str(record.get("topic_group", "")).strip() or "Miscellaneous"
    try:
        topic_group = canonicalize_label(topic_group)
    except Exception:
        topic_group = topic_group or "Miscellaneous"

    if not _valid_policy_ids():
        return {
            "ticket_index": record.get("ticket_index"),
            "topic_group": topic_group,
            "ticket_text": ticket_text,
            "gold_resolution": normalize_gold_resolution({}),
            "needs_human_review": True,
            "review_reasons": ["kb_index_empty_or_unreadable"],
        }

    retrieval = retrieve_kb(ticket_text, memory_mode, top_k)
    snippets = retrieval["snippets"]
    allowed_policy_ids = [pid for pid in retrieval["policy_ids"] if pid in _valid_policy_ids()]
    fallback_used = False
    if not allowed_policy_ids:
        fallback_snippets, fallback_ids, fallback_error = _fallback_policy_selection(
            ticket_text, topic_group, top_k
        )
        if fallback_error:
            return {
                "ticket_index": record.get("ticket_index"),
                "topic_group": topic_group,
                "ticket_text": ticket_text,
                "gold_resolution": normalize_gold_resolution({}),
                "needs_human_review": True,
                "review_reasons": [fallback_error],
            }
        snippets = fallback_snippets
        allowed_policy_ids = fallback_ids
        fallback_used = True

    allowed_set = set(allowed_policy_ids)
    if not allowed_set:
        return {
            "ticket_index": record.get("ticket_index"),
            "topic_group": topic_group,
            "ticket_text": ticket_text,
            "gold_resolution": normalize_gold_resolution({}),
            "needs_human_review": True,
            "review_reasons": ["kb_index_empty_or_unreadable"],
        }

    annotator_prompt = build_annotator_prompt(ticket_text, topic_group, snippets, allowed_policy_ids)
    annotator_output_raw = call_model(models.get(annotator_model), annotator_prompt)
    annotator_obj = parse_json_safely(annotator_output_raw)
    annotator_resolution = normalize_gold_resolution(_extract_resolution(annotator_obj))
    annotator_resolution["kb_policies"] = [
        pid for pid in annotator_resolution.get("kb_policies", []) if pid in allowed_set
    ]
    annotator_ok, annotator_reasons = validate_gold_resolution(annotator_resolution, allowed_policy_ids)

    for _attempt in range(2):
        if annotator_ok:
            break
        repair_prompt = build_repair_prompt(
            ticket_text=ticket_text,
            topic_group=topic_group,
            snippets=snippets,
            allowed_policy_ids=allowed_policy_ids,
            prior_output=annotator_resolution,
            validation_errors=annotator_reasons,
        )
        repair_output_raw = call_model(models.get(annotator_model), repair_prompt)
        repair_obj = parse_json_safely(repair_output_raw)
        annotator_resolution = normalize_gold_resolution(_extract_resolution(repair_obj))
        annotator_resolution["kb_policies"] = [
            pid for pid in annotator_resolution.get("kb_policies", []) if pid in allowed_set
        ]
        annotator_ok, annotator_reasons = validate_gold_resolution(annotator_resolution, allowed_policy_ids)

    reviewer_prompt = build_reviewer_prompt(
        ticket_text, topic_group, snippets, allowed_policy_ids, annotator_resolution
    )
    reviewer_output_raw = call_model(models.get(reviewer_model), reviewer_prompt)
    reviewer_obj = parse_json_safely(reviewer_output_raw)
    reviewer_resolution = normalize_gold_resolution(_extract_resolution(reviewer_obj))
    reviewer_resolution["kb_policies"] = [
        pid for pid in reviewer_resolution.get("kb_policies", []) if pid in allowed_set
    ]
    reviewer_ok, reviewer_reasons = validate_gold_resolution(reviewer_resolution, allowed_policy_ids)

    if reviewer_ok:
        final_resolution = reviewer_resolution
        final_ok = True
        final_reasons: List[str] = []
    elif annotator_ok:
        final_resolution = annotator_resolution
        final_ok = True
        final_reasons = []
    else:
        final_resolution = reviewer_resolution or annotator_resolution
        final_ok = False
        final_reasons = reviewer_reasons or annotator_reasons or ["model_invalid_output"]

    needs_review = not final_ok
    review_reasons = list(final_reasons)
    if fallback_used:
        review_reasons.append("kb_fallback_used")

    output = {
        "ticket_index": record.get("ticket_index"),
        "topic_group": topic_group,
        "ticket_text": ticket_text,
        "gold_resolution": final_resolution,
        "needs_human_review": needs_review,
        "review_reasons": review_reasons if (needs_review or review_reasons) else [],
    }
    return output


def autofill_records(
    records: Sequence[Dict[str, Any]],
    models: Dict[str, Any],
    memory_mode: str = DEFAULT_MEMORY_MODE,
    top_k: int = DEFAULT_TOP_K,
    annotator_model: str = DEFAULT_ANNOTATOR,
    reviewer_model: str = DEFAULT_REVIEWER,
    start: int = 0,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)
    slice_records = list(records)[start : start + limit if limit is not None else None]
    outputs: List[Dict[str, Any]] = []
    review_queue: List[Dict[str, Any]] = []
    for record in slice_records:
        processed = process_record(record, models, memory_mode, top_k, annotator_model, reviewer_model)
        outputs.append(processed)
        if processed.get("needs_human_review"):
            review_queue.append(processed)
    return outputs, review_queue


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autofill gold eval set with KB-grounded LLMs.")
    parser.add_argument("--input-path", type=str, default=str(DEFAULT_INPUT), help="Path to gold_eval.jsonl template.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT), help="Path to write filled gold eval JSONL.")
    parser.add_argument("--review-queue-path", type=str, default=str(DEFAULT_REVIEW), help="Path to write human review queue JSONL.")
    parser.add_argument("--memory-mode", type=str, default=DEFAULT_MEMORY_MODE, help="Memory mode for KB retrieval.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k snippets to retrieve from KB.")
    parser.add_argument("--annotator-model", type=str, default=DEFAULT_ANNOTATOR, help="Model name for annotator pass.")
    parser.add_argument("--reviewer-model", type=str, default=DEFAULT_REVIEWER, help="Model name for reviewer pass.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of records to process.")
    parser.add_argument("--start", type=int, default=0, help="Optional offset into records.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic ordering.")
    parser.add_argument("--offline", action="store_true", help="Skip LLM calls; fill placeholders and mark for review.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    review_path = Path(args.review_queue_path)

    records = _read_jsonl(input_path)
    if not records:
        raise RuntimeError(f"No records found at {input_path}")

    env_offline = not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))
    offline = args.offline or env_offline
    if offline:
        def _offline_model(_prompt: str, **_kwargs: Any) -> str:
            return "{}"
        models = {args.annotator_model: _offline_model, args.reviewer_model: _offline_model}
        print("Running in offline mode; model outputs will be empty and marked for review.")
    else:
        subset_models = list({args.annotator_model, args.reviewer_model})
        models = load_models(sanity=False, slm_subset=subset_models, force_llm=True)
        if args.annotator_model not in models or args.reviewer_model not in models:
            missing = [m for m in [args.annotator_model, args.reviewer_model] if m not in models]
            raise RuntimeError(f"Models not available: {missing}")

    filled, review_queue = autofill_records(
        records=records,
        models=models,
        memory_mode=args.memory_mode,
        top_k=args.top_k,
        annotator_model=args.annotator_model,
        reviewer_model=args.reviewer_model,
        start=args.start,
        limit=args.limit,
        seed=args.seed,
    )

    _write_jsonl(filled, output_path)
    _write_jsonl(review_queue, review_path)

    print(f"Wrote {len(filled)} gold eval tickets to {output_path}")
    print(f"Queued {len(review_queue)} tickets for human review at {review_path}")


if __name__ == "__main__":
    main()
