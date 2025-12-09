"""Train encoder centroids over tickets dataset using a frozen E5 encoder.

Run:
    python -m RouterGym.scripts.train_encoder_centroids

Uses the canonical 6-label space: access, administrative rights,
hardware, hr support, purchase, miscellaneous.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("sentence-transformers is required to train encoder centroids") from exc

from RouterGym.label_space import CANONICAL_LABELS, canonical_label

DEFAULT_MODEL = "intfloat/e5-small-v2"
CENTROID_PATH = Path(__file__).resolve().parents[1] / "classifiers" / "encoder_centroids.npz"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "tickets"


def _find_dataset() -> Path:
    candidates = list(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV dataset found under {DATA_DIR}")
    # Prefer files containing likely columns
    for cand in candidates:
        try:
            df_head = pd.read_csv(cand, nrows=1)
            cols = {c.lower() for c in df_head.columns}
            if {"document", "topic_group"} & cols or {"text", "category"} & cols:
                return cand
        except Exception:
            continue
    return candidates[0]


def _infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    col_map = {c.lower(): c for c in df.columns}
    text_col = None
    label_col = None
    for candidate in ["document", "text", "body", "ticket", "content"]:
        if candidate in col_map:
            text_col = col_map[candidate]
            break
    for candidate in ["topic_group", "category", "label", "tag"]:
        if candidate in col_map:
            label_col = col_map[candidate]
            break
    if text_col is None or label_col is None:
        raise ValueError(f"Could not infer text/label columns from columns={df.columns.tolist()}")
    return text_col, label_col


def _encode_batch(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return np.array(
        model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True),
        dtype="float32",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoder centroids on tickets dataset.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Encoder model name (default: intfloat/e5-small-v2)")
    args = parser.parse_args()

    dataset_path = _find_dataset()
    print(f"[Centroids] Using dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    text_col, label_col = _infer_columns(df)
    print(f"[Centroids] Using text column '{text_col}' and label column '{label_col}'")

    df = df[[text_col, label_col]].dropna()
    df["label_norm"] = df[label_col].apply(canonical_label)
    df = df[df["label_norm"].isin(CANONICAL_LABELS)]
    if df.empty:
        raise ValueError("No records after label normalization; check dataset columns.")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_norm"])
    print(f"[Centroids] Train size: {len(train_df)} | Val size: {len(val_df)}")

    model = SentenceTransformer(args.model)
    train_embeddings = _encode_batch(model, train_df[text_col].tolist())
    train_labels = train_df["label_norm"].tolist()

    centroids = []
    centroid_labels = []
    for label in CANONICAL_LABELS:
        mask = [lbl == label for lbl in train_labels]
        if not any(mask):
            continue
        label_embs = train_embeddings[mask]
        centroids.append(label_embs.mean(axis=0))
        centroid_labels.append(label)
    centroids_np = np.vstack(centroids)
    norms = np.linalg.norm(centroids_np, axis=1, keepdims=True) + 1e-9
    centroids_np = centroids_np / norms

    # Validation
    val_embeddings = _encode_batch(model, val_df[text_col].tolist())
    correct = 0
    per_label_total: Dict[str, int] = {lbl: 0 for lbl in centroid_labels}
    per_label_correct: Dict[str, int] = {lbl: 0 for lbl in centroid_labels}
    centroid_matrix = centroids_np.T
    for emb, lbl in zip(val_embeddings, val_df["label_norm"]):
        sim = emb @ centroid_matrix
        pred = centroid_labels[int(np.argmax(sim))]
        per_label_total[lbl] = per_label_total.get(lbl, 0) + 1
        if pred == lbl:
            correct += 1
            per_label_correct[lbl] = per_label_correct.get(lbl, 0) + 1
    acc = correct / max(len(val_df), 1)
    print(f"[Centroids] Validation accuracy: {acc:.3f}")
    for lbl in centroid_labels:
        total = per_label_total.get(lbl, 0)
        corr = per_label_correct.get(lbl, 0)
        if total:
            print(f"  {lbl:15s}: {corr}/{total} = {corr/total:.3f}")

    CENTROID_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CENTROID_PATH, labels=np.array(centroid_labels, dtype=str), centroids=centroids_np.astype("float32"))
    print(f"[Centroids] Saved centroids to {CENTROID_PATH}")


if __name__ == "__main__":
    main()
