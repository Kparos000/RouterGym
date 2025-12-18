"""Classification-only experiment using the calibrated encoder head.

Runs the frozen encoder_calibrated (E5 + MLP head) over the tickets dataset and
writes a CSV with per-ticket classification results. No routing, KB retrieval,
or generation is performed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from RouterGym.agents.generator import get_confidence_bucket
from RouterGym.classifiers.encoder_classifier import EncoderClassifier
from RouterGym.data.tickets import dataset_loader
from RouterGym.label_space import CANONICAL_LABELS

DEFAULT_OUTPUT = Path("RouterGym/results/encoder_classification_full.csv")


def classify_dataframe(df: pd.DataFrame, classifier: Any, start_offset: int = 0) -> List[Dict[str, Any]]:
    """Run classification over a dataframe of tickets and return per-row dicts."""
    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        ticket_id = start_offset + idx
        text = str(row.get("text", "") or "")
        gold_label = str(row.get("label", "") or "")
        t0 = time.perf_counter()
        probs = classifier.predict_proba(text)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        if not probs:
            raise RuntimeError("EncoderClassifier.predict_proba returned empty probabilities.")
        predicted_label = max(probs, key=probs.__getitem__)
        predicted_confidence = float(probs[predicted_label])
        confidence_bucket = get_confidence_bucket(predicted_confidence)
        correct = predicted_label == gold_label
        records.append(
            {
                "ticket_id": ticket_id,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "predicted_confidence": predicted_confidence,
                "confidence_bucket": confidence_bucket,
                "correct": correct,
                "runtime_ms": runtime_ms,
                "ticket_text": text,
                "classifier_backend": classifier.backend_name,
            }
        )
    return records


def _summaries(records: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    total = len(records)
    if total == 0:
        return 0.0, {}, {}
    correct = sum(1 for r in records if r["correct"])
    overall = correct / total
    per_label: Dict[str, Tuple[int, int]] = {lbl: (0, 0) for lbl in CANONICAL_LABELS}
    for rec in records:
        lbl = rec["gold_label"]
        if lbl not in per_label:
            per_label[lbl] = (0, 0)
        hits, cnt = per_label[lbl]
        per_label[lbl] = (hits + int(rec["correct"]), cnt + 1)
    per_label_acc = {lbl: (hits / cnt) if cnt else 0.0 for lbl, (hits, cnt) in per_label.items()}

    per_bucket: Dict[str, Tuple[int, int]] = {}
    for rec in records:
        b = rec["confidence_bucket"]
        hits, cnt = per_bucket.get(b, (0, 0))
        per_bucket[b] = (hits + int(rec["correct"]), cnt + 1)
    per_bucket_acc = {b: (hits / cnt) if cnt else 0.0 for b, (hits, cnt) in per_bucket.items()}
    return overall, per_label_acc, per_bucket_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated encoder classification over tickets.")
    parser.add_argument(
        "--ticket-path",
        type=Path,
        default=None,
        help="Path to tickets CSV (defaults to RouterGym/data/tickets/tickets.csv).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write classification results CSV.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of tickets.")
    parser.add_argument("--start", type=int, default=0, help="Optional start offset.")
    args = parser.parse_args()

    df = dataset_loader.load_dataset(n=args.limit, start=args.start, path=args.ticket_path)
    if df.empty:
        raise RuntimeError("No tickets loaded; check the ticket path or filters.")

    classifier = EncoderClassifier(head_mode="calibrated")
    if classifier.backend_name != "encoder_calibrated":
        raise RuntimeError(
            "Calibrated encoder head not active. Run `python -m RouterGym.scripts.train_encoder_calibrated_head` first."
        )

    records = classify_dataframe(df, classifier, start_offset=args.start)
    output_path = args.output_path or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)

    overall, per_label_acc, per_bucket_acc = _summaries(records)
    print(f"Processed {len(records)} tickets; overall accuracy={overall:.3f}")
    print("Per-label accuracy:")
    for lbl in sorted(per_label_acc):
        print(f"  {lbl}: {per_label_acc[lbl]:.3f}")
    print("Per-confidence-bucket accuracy:")
    for b, acc in sorted(per_bucket_acc.items()):
        print(f"  {b}: {acc:.3f}")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
