"""Build a gold evaluation template JSONL from tickets.csv."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from RouterGym.label_space import canonicalize_label

DEFAULT_SEED = 42
TARGET_COUNTS: Dict[str, int] = {
    "Hardware": 60,
    "HR Support": 50,
    "Access": 40,
    "Storage": 30,
    "Purchase": 30,
    "Internal Project": 30,
    "Administrative rights": 30,
    "Miscellaneous": 30,
}

DEFAULT_TICKETS_PATH = Path(__file__).resolve().parents[1] / "data" / "tickets" / "tickets.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "gold_eval" / "gold_eval.jsonl"


def _normalize_dataframe(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected columns exist and drop empty rows."""
    df = tickets_df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required = {"document", "topic_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}; expected Document and Topic_group.")
    df = df.dropna(subset=["document", "topic_group"])
    df = df[df["document"].astype(str).str.strip() != ""]
    df = df[df["topic_group"].astype(str).str.strip() != ""]
    return df


def _build_record(ticket_index: int, topic_group: str, ticket_text: str) -> Dict[str, Any]:
    """Create a single gold eval record."""
    return {
        "ticket_index": int(ticket_index),
        "topic_group": topic_group,
        "ticket_text": str(ticket_text),
        "gold_resolution": {
            "summary": "TODO",
            "steps": [],
            "escalation_required": False,
            "escalation_reason": "",
            "kb_policies": [],
            "acceptance_criteria": [],
        },
    }


def build_gold_eval(
    tickets_df: pd.DataFrame,
    target_counts: Dict[str, int] | None = None,
    seed: int = DEFAULT_SEED,
    start: int | None = None,
    limit: int | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Sample tickets per label and build gold eval records."""
    targets = target_counts or TARGET_COUNTS
    df = _normalize_dataframe(tickets_df)
    rng = random.Random(seed)

    label_to_rows: Dict[str, List[Tuple[int, str]]] = {label: [] for label in targets}
    for idx, row in df.iterrows():
        try:
            label = canonicalize_label(row["topic_group"])
        except RuntimeError as err:
            raise RuntimeError(f"Failed to canonicalize label for row {idx}: {err}") from err
        if label not in label_to_rows:
            continue
        label_to_rows[label].append((int(idx), str(row["document"])))

    records: List[Dict[str, Any]] = []
    actual_counts: Dict[str, int] = {}
    for label, target in targets.items():
        rows = label_to_rows.get(label, [])
        if len(rows) <= target:
            if len(rows) < target:
                print(f"Warning: requested {target} rows for {label} but only {len(rows)} available; using all.")
            chosen = rows
        else:
            chosen = rng.sample(rows, target)
        actual_counts[label] = len(chosen)
        for row_index, text in chosen:
            records.append(_build_record(row_index, label, text))

    records.sort(key=lambda r: int(r["ticket_index"]))
    if start is not None or limit is not None:
        start_idx = start or 0
        end_idx = start_idx + limit if limit is not None else None
        records = records[start_idx:end_idx]

    final_counts: Dict[str, int] = {}
    for rec in records:
        lbl = str(rec.get("topic_group", ""))
        final_counts[lbl] = final_counts.get(lbl, 0) + 1
    return records, final_counts


def _write_jsonl(records: Iterable[Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold eval JSONL template from tickets.csv.")
    parser.add_argument(
        "--tickets-path",
        type=str,
        default=None,
        help="Optional override path to tickets.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional override path to gold_eval.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of sampled records after sorting.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Optional start offset into sampled records.",
    )
    args = parser.parse_args()

    tickets_path = Path(args.tickets_path) if args.tickets_path else DEFAULT_TICKETS_PATH
    output_path = Path(args.output_path) if args.output_path else DEFAULT_OUTPUT_PATH

    tickets_df = pd.read_csv(tickets_path)
    records, counts = build_gold_eval(
        tickets_df,
        TARGET_COUNTS,
        seed=DEFAULT_SEED,
        start=args.start,
        limit=args.limit,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(records, output_path)

    total = sum(counts.values())
    print(f"Wrote {total} gold eval tickets to {output_path}")
    print(f"Per-label counts: {counts}")


if __name__ == "__main__":
    main()
