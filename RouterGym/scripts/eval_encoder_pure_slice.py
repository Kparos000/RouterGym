"""Evaluate pure encoder-centroid classifier on a ticket slice (diagnostic)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from RouterGym.classifiers.encoder_classifier import EncoderClassifier
from RouterGym.label_space import CANONICAL_LABELS, canonicalize_label

DATA_PATH = Path("RouterGym/data/tickets/tickets.csv")
TEXT_COL = "Document"
LABEL_COL = "Topic_group"
CANONICALS = CANONICAL_LABELS


def _load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Expected columns '{TEXT_COL}' and '{LABEL_COL}' in {path}")
    return df[[TEXT_COL, LABEL_COL]].dropna()


def _normalize_label(label: str) -> str:
    return canonicalize_label(label)


def evaluate_slice(ticket_start: int, ticket_limit: int, head_mode: str) -> None:
    # Preserve centroid/pure behaviour when requested, but keep calibrated path aligned with training features.
    if head_mode == "centroid":
        os.environ["ROUTERGYM_ENCODER_USE_LEXICAL_PRIOR"] = "0"
    else:
        os.environ.pop("ROUTERGYM_ENCODER_USE_LEXICAL_PRIOR", None)
    df = _load_data(DATA_PATH)
    end = ticket_start + ticket_limit if ticket_limit >= 0 else len(df)
    slice_df = df.iloc[ticket_start:end]
    clf = EncoderClassifier(labels=CANONICAL_LABELS, use_lexical_prior=True, head_mode=head_mode)

    total_per: Dict[str, int] = {lbl: 0 for lbl in CANONICALS}
    correct_per: Dict[str, int] = {lbl: 0 for lbl in CANONICALS}
    correct = 0

    for _, row in slice_df.iterrows():
        gold = _normalize_label(str(row[LABEL_COL]))
        pred = canonicalize_label(clf.predict_label(str(row[TEXT_COL])))
        if gold not in total_per:
            gold = "Miscellaneous"
        total_per[gold] += 1
        if pred == gold:
            correct_per[gold] += 1
            correct += 1

    n = len(slice_df)
    overall = correct / n if n else 0.0
    print(f"Overall accuracy on slice (N={n}): {overall:.3f}\n")
    for lbl in CANONICALS:
        total = total_per.get(lbl, 0)
        corr = correct_per.get(lbl, 0)
        acc = corr / total if total else 0.0
        print(f"{lbl:13s}: {corr}/{total} = {acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate encoder classifier on a ticket slice.")
    parser.add_argument("--ticket-start", type=int, default=0, help="Start index (0-based) into tickets.csv")
    parser.add_argument("--ticket-limit", type=int, default=30, help="Max tickets to evaluate (default 30; -1 for all)")
    parser.add_argument(
        "--head-mode",
        type=str,
        default="auto",
        choices=["centroid", "calibrated", "auto"],
        help="Head mode to use for encoder classifier.",
    )
    args = parser.parse_args()
    evaluate_slice(args.ticket_start, args.ticket_limit, args.head_mode)


if __name__ == "__main__":
    main()
