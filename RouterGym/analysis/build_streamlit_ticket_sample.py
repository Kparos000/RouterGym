"""Build a small deployable ticket sample for the Streamlit results explorer.

The app must not load the large 60k analysis table. This script creates a compact
CSV from the richest local source available, then enriches it with deterministic
gold scores and blinded manual-audit scores when those files exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "RouterGym" / "results" / "analysis_outputs"
GOLD_DIR = ANALYSIS_DIR / "gold_resolution_eval"
MANUAL_DIR = ANALYSIS_DIR / "manual_audit"

GOLD_MATCHED_JSONL = GOLD_DIR / "gold_matched_production_outputs.jsonl"
BALANCED_FLAT_CSV = ANALYSIS_DIR / "balanced_60k_all_configs_flat.csv"
SAMPLE_ROWS_JSON = ANALYSIS_DIR / "sample_rows.json"
GOLD_SCORES_CSV = GOLD_DIR / "gold_resolution_scores_flat.csv"
MANUAL_COMPLETED_CSV = MANUAL_DIR / "manual_audit_full_blinded_completed.csv"
MANUAL_KEY_CSV = MANUAL_DIR / "manual_audit_full_key.csv"
OUTPUT_CSV = ANALYSIS_DIR / "streamlit_ticket_sample.csv"

TARGET_MIN_ROWS = 200
TARGET_MAX_ROWS = 500
CHUNK_SIZE = 50_000

CONFIG_ORDER = [
    "llm_only__base_llm1__mem_rag_bm25",
    "llm_only__base_llm2__mem_rag_bm25",
    "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25",
    "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25",
    "slm_only__base_slm1__mem_rag_bm25",
    "slm_only__base_slm2__mem_rag_bm25",
]

SOURCE_COLUMNS = [
    "ticket_id",
    "ticket_index",
    "analysis_config",
    "config_identifier",
    "router_family",
    "router_mode",
    "base_model",
    "base_model_name",
    "escalation_model",
    "escalation_model_name",
    "memory_mode",
    "context_mode",
    "gold_label",
    "topic_group",
    "gold_topic_group",
    "original_query",
    "ticket_request",
    "rewritten_query",
    "gold_ticket_text",
    "predicted_category",
    "classifier_predicted_category",
    "generated_predicted_category",
    "final_answer",
    "generated_final_answer",
    "resolution_steps",
    "generated_resolution_steps",
    "reasoning",
    "generated_reasoning",
    "escalation_flags",
    "escalated",
    "parse_error",
    "validation_error",
    "generation_valid",
    "analysis_usable",
    "raw_response_saved",
    "total_tokens",
    "total_cost_usd",
    "cost_usd",
    "latency_ms",
    "metrics",
    "gold_resolution",
    "gold_review_status",
]


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not file_exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def safe_read_jsonl(path: Path) -> pd.DataFrame:
    if not file_exists(path):
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(rows)


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def parse_maybe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def pick_series(df: pd.DataFrame, candidates: Iterable[str], default: Any = "") -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series([default] * len(df), index=df.index)


def infer_router_family(config: Any) -> str:
    text = str(config or "")
    if text.startswith("llm_only"):
        return "llm_only"
    if text.startswith("slm_only"):
        return "slm_only"
    if text.startswith("slm_dominant"):
        return "slm_dominant"
    return ""


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["ticket_id"] = pick_series(out, ["ticket_id", "ticket_index"]).astype(str)
    out["analysis_config"] = pick_series(out, ["analysis_config", "config_identifier"]).astype(str)
    return out


def gold_ticket_ids() -> set[str]:
    scores = safe_read_csv(GOLD_SCORES_CSV, dtype={"ticket_id": str})
    if not scores.empty and "ticket_id" in scores.columns:
        return set(scores["ticket_id"].dropna().astype(str).unique().tolist())
    return set()


def load_from_gold_matched() -> tuple[pd.DataFrame, str]:
    df = safe_read_jsonl(GOLD_MATCHED_JSONL)
    if df.empty:
        return df, ""
    return normalize_keys(df), str(GOLD_MATCHED_JSONL.relative_to(ROOT))


def load_from_balanced_flat() -> tuple[pd.DataFrame, str]:
    if not file_exists(BALANCED_FLAT_CSV):
        return pd.DataFrame(), ""

    header = pd.read_csv(BALANCED_FLAT_CSV, nrows=0)
    usecols = [column for column in SOURCE_COLUMNS if column in header.columns]
    if "ticket_id" not in usecols and "ticket_index" not in usecols:
        return pd.DataFrame(), ""

    preferred_ids = gold_ticket_ids()
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        BALANCED_FLAT_CSV,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        dtype={"ticket_id": str},
    ):
        chunk = normalize_keys(chunk)
        if preferred_ids:
            selected = chunk[chunk["ticket_id"].isin(preferred_ids)]
        else:
            selected = chunk
        if not selected.empty:
            chunks.append(selected)
        if not preferred_ids and sum(len(part) for part in chunks) >= TARGET_MAX_ROWS:
            break

    if not chunks:
        return pd.DataFrame(), ""
    df = pd.concat(chunks, ignore_index=True)
    return df, str(BALANCED_FLAT_CSV.relative_to(ROOT))


def load_from_sample_rows() -> tuple[pd.DataFrame, str]:
    if not file_exists(SAMPLE_ROWS_JSON):
        return pd.DataFrame(), ""
    try:
        df = pd.read_json(SAMPLE_ROWS_JSON)
    except ValueError:
        return pd.DataFrame(), ""
    if df.empty:
        return df, ""
    return normalize_keys(df), str(SAMPLE_ROWS_JSON.relative_to(ROOT))


def load_source() -> tuple[pd.DataFrame, str]:
    # This small derived file is the richest deployable source when present: it
    # already contains the gold-overlap examples, generated answers, and steps.
    for loader in (load_from_gold_matched, load_from_balanced_flat, load_from_sample_rows):
        df, source = loader()
        if not df.empty:
            return df, source
    return pd.DataFrame(), ""


def flatten_gold_resolution(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "gold_resolution" not in out.columns:
        out["gold_resolution_summary"] = ""
        out["gold_resolution_steps"] = ""
        out["gold_acceptance_criteria"] = ""
        return out

    parsed = out["gold_resolution"].map(parse_maybe_json)
    out["gold_resolution_summary"] = parsed.map(
        lambda value: value.get("summary", "") if isinstance(value, dict) else ""
    )
    out["gold_resolution_steps"] = parsed.map(
        lambda value: to_text(value.get("steps", [])) if isinstance(value, dict) else ""
    )
    out["gold_acceptance_criteria"] = parsed.map(
        lambda value: (
            to_text(value.get("acceptance_criteria", [])) if isinstance(value, dict) else ""
        )
    )
    return out


def merge_gold_scores(df: pd.DataFrame) -> pd.DataFrame:
    scores = safe_read_csv(GOLD_SCORES_CSV, dtype={"ticket_id": str})
    if scores.empty:
        return df
    scores = normalize_keys(scores)
    keep = [
        "ticket_id",
        "analysis_config",
        "overall_gold_quality_score",
        "step_coverage_score",
        "acceptance_criteria_alignment_score",
        "escalation_correctness_score",
        "policy_grounding_match_score",
        "generated_category_correct",
    ]
    keep = [column for column in keep if column in scores.columns]
    return df.merge(scores[keep], on=["ticket_id", "analysis_config"], how="left")


def merge_manual_scores(df: pd.DataFrame) -> pd.DataFrame:
    completed = safe_read_csv(MANUAL_COMPLETED_CSV, dtype={"ticket_id": str})
    key = safe_read_csv(MANUAL_KEY_CSV, dtype={"ticket_id": str})
    if (
        completed.empty
        or key.empty
        or "audit_id" not in completed.columns
        or "audit_id" not in key.columns
    ):
        return df

    key = normalize_keys(key)
    key_cols = [
        "audit_id",
        "analysis_config",
        "anonymous_system_id",
        "parse_error_present",
        "validation_error_present",
        "total_cost_usd",
        "latency_ms",
    ]
    key_cols = [column for column in key_cols if column in key.columns]
    manual = completed.merge(key[key_cols], on="audit_id", how="left", suffixes=("", "_key"))
    manual = normalize_keys(manual)

    manual_cols = [
        "ticket_id",
        "analysis_config",
        "anonymous_system_id",
        "category_understanding_manual",
        "answer_actionable_manual",
        "answer_complete_manual",
        "resolution_steps_correct_manual",
        "escalation_appropriate_manual",
        "policy_grounded_manual",
        "overall_manual_quality",
        "reviewer_notes",
    ]
    manual_cols = [column for column in manual_cols if column in manual.columns]
    manual = manual[manual_cols].drop_duplicates(subset=["ticket_id", "analysis_config"])
    return df.merge(manual, on=["ticket_id", "analysis_config"], how="left")


def compute_usable_output(df: pd.DataFrame) -> pd.Series:
    if "analysis_usable" in df.columns:
        return df["analysis_usable"]
    generation_valid = pick_series(df, ["generation_valid"], default=True).astype(str).str.lower()
    final_answer = pick_series(df, ["final_answer", "generated_final_answer"], default="")
    parse_error = pick_series(df, ["parse_error"], default="")
    validation_error = pick_series(df, ["validation_error"], default="")
    return (
        generation_valid.isin(["true", "1", "yes"])
        & final_answer.map(lambda value: bool(str(value).strip()))
        & parse_error.map(lambda value: not bool(str(value).strip()) or str(value).lower() == "nan")
        & validation_error.map(
            lambda value: not bool(str(value).strip()) or str(value).lower() == "nan"
        )
    )


def extract_latency(df: pd.DataFrame) -> pd.Series:
    if "latency_ms" in df.columns:
        return df["latency_ms"]
    if "metrics" not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df["metrics"].map(
        lambda value: (
            parse_maybe_json(value).get("latency_ms", "")
            if isinstance(parse_maybe_json(value), dict)
            else ""
        )
    )


def make_output(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = flatten_gold_resolution(normalize_keys(df))
    df = merge_gold_scores(df)
    df = merge_manual_scores(df)

    config = pick_series(df, ["analysis_config", "config_identifier"]).astype(str)
    router = pick_series(df, ["router_family", "router_mode"], default="")
    router = router.where(
        router.astype(str).str.strip().astype(bool), config.map(infer_router_family)
    )

    out = pd.DataFrame(index=df.index)
    out["ticket_id"] = pick_series(df, ["ticket_id", "ticket_index"]).astype(str)
    out["config_identifier"] = config
    out["router_family"] = router
    out["base_model_key"] = pick_series(df, ["base_model", "base_model_name"])
    out["escalation_model_key"] = pick_series(df, ["escalation_model", "escalation_model_name"])
    out["memory_mode"] = pick_series(df, ["memory_mode", "context_mode"])
    out["ticket_text"] = pick_series(
        df,
        ["gold_ticket_text", "original_query", "ticket_request", "rewritten_query"],
    )
    out["gold_label"] = pick_series(df, ["gold_label", "topic_group", "gold_topic_group"])
    out["predicted_category"] = pick_series(
        df,
        ["predicted_category", "classifier_predicted_category"],
    )
    out["generated_predicted_category"] = pick_series(df, ["generated_predicted_category"])
    out["final_answer"] = pick_series(df, ["final_answer", "generated_final_answer"])
    out["resolution_steps"] = pick_series(
        df, ["resolution_steps", "generated_resolution_steps"]
    ).map(to_text)
    out["reasoning"] = pick_series(df, ["reasoning", "generated_reasoning"])
    out["escalated"] = pick_series(df, ["escalated"], default=False)
    out["escalation_flags"] = pick_series(df, ["escalation_flags"], default="").map(to_text)
    out["parse_error"] = pick_series(df, ["parse_error"])
    out["validation_error"] = pick_series(df, ["validation_error"])
    out["generation_valid"] = pick_series(df, ["generation_valid"])
    out["usable_output"] = compute_usable_output(df)
    out["raw_response_saved"] = pick_series(df, ["raw_response_saved"])
    out["total_tokens"] = pick_series(df, ["total_tokens"])
    out["cost_usd"] = pick_series(df, ["cost_usd", "total_cost_usd"])
    out["latency_ms"] = extract_latency(df)
    out["deterministic_gold_score"] = pick_series(df, ["overall_gold_quality_score"])
    out["manual_quality_score"] = pick_series(df, ["overall_manual_quality"])
    out["manual_category_understanding"] = pick_series(df, ["category_understanding_manual"])
    out["manual_answer_actionable"] = pick_series(df, ["answer_actionable_manual"])
    out["manual_answer_complete"] = pick_series(df, ["answer_complete_manual"])
    out["manual_resolution_steps_correct"] = pick_series(df, ["resolution_steps_correct_manual"])
    out["manual_escalation_appropriate"] = pick_series(df, ["escalation_appropriate_manual"])
    out["manual_policy_grounded"] = pick_series(df, ["policy_grounded_manual"])
    out["manual_reviewer_notes"] = pick_series(df, ["reviewer_notes"])
    out["anonymous_system_id"] = pick_series(df, ["anonymous_system_id"])
    out["gold_resolution_summary"] = pick_series(df, ["gold_resolution_summary"])
    out["gold_resolution_steps"] = pick_series(df, ["gold_resolution_steps"]).map(to_text)
    out["gold_acceptance_criteria"] = pick_series(df, ["gold_acceptance_criteria"]).map(to_text)
    out["gold_review_status"] = pick_series(df, ["gold_review_status"])
    out["sample_source"] = source

    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(to_text)

    out["_config_rank"] = (
        out["config_identifier"]
        .map({config_name: index for index, config_name in enumerate(CONFIG_ORDER)})
        .fillna(999)
    )
    out["_ticket_rank"] = pd.to_numeric(out["ticket_id"], errors="coerce")
    out = out.sort_values(["_ticket_rank", "_config_rank", "config_identifier"]).drop(
        columns=["_ticket_rank", "_config_rank"]
    )

    if len(out) > TARGET_MAX_ROWS:
        # Prefer keeping complete config sets for the earliest selected ticket IDs.
        ticket_ids = out["ticket_id"].drop_duplicates().head(TARGET_MAX_ROWS // 6).tolist()
        out = out[out["ticket_id"].isin(ticket_ids)]
    return out.reset_index(drop=True)


def build_sample(output_path: Path = OUTPUT_CSV) -> Path:
    df, source = load_source()
    if df.empty:
        raise FileNotFoundError(
            "No usable source found. Expected one of: "
            f"{GOLD_MATCHED_JSONL}, {BALANCED_FLAT_CSV}, {SAMPLE_ROWS_JSON}"
        )

    output = make_output(df, source)
    if len(output) < TARGET_MIN_ROWS:
        print(f"Warning: sample has only {len(output)} rows; expected at least {TARGET_MIN_ROWS}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(output):,} rows to {output_path}")
    print(f"Source: {source}")
    print(f"Size: {size_mb:.2f} MB")
    if size_mb > 10:
        print("Warning: sample exceeds the recommended 10 MB Streamlit/GitHub target.")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV path for the deployable Streamlit ticket sample.",
    )
    args = parser.parse_args()
    build_sample(args.output)


if __name__ == "__main__":
    main()
