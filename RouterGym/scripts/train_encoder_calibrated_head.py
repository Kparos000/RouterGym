"""Train a calibrated logistic head on top of encoder centroids + lexical priors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from RouterGym.classifiers.tfidf_classifier import TFIDFClassifier
from RouterGym.classifiers.utils import apply_lexical_prior
from RouterGym.label_space import CANONICAL_LABELS, canonical_label, canonicalize_label

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("sentence-transformers is required to train calibrated head") from exc

DEFAULT_TICKET_PATH = Path("RouterGym/data/tickets/tickets.csv")
DEFAULT_TEXT_COL = "Document"
DEFAULT_LABEL_COL = "Topic_group"
CENTROID_PATH = Path(__file__).resolve().parents[1] / "classifiers" / "encoder_centroids.npz"
HEAD_OUT_PATH = Path(__file__).resolve().parents[1] / "classifiers" / "encoder_calibrated_head.npz"
HEAD_VERSION = "1.0"
log = logging.getLogger(__name__)


def _load_dataset(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{text_col}' and '{label_col}' in {path}")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].apply(canonicalize_label)
    df = df[df[label_col].isin(CANONICAL_LABELS)]
    return df.reset_index(drop=True)


def _load_centroids() -> Tuple[np.ndarray, List[str]]:
    data = np.load(CENTROID_PATH, allow_pickle=True)
    labels = [canonical_label(lbl) for lbl in data["labels"].tolist()]
    cents = np.array(data["centroids"], dtype="float32")
    if cents.ndim != 2 or cents.shape[0] != len(labels):
        raise ValueError("Invalid centroid file")
    norms = np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    return (cents / norms), labels


def _compute_features(
    emb: np.ndarray,
    centroids: np.ndarray,
    labels: List[str],
    text: str,
    tfidf_clf: TFIDFClassifier,
) -> np.ndarray:
    emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
    sims = emb_norm @ centroids.T  # (num_labels,)
    base = {label: 1.0 / max(len(labels), 1) for label in labels}
    priors = apply_lexical_prior(text, base, alpha=0.0, beta=1.0)
    priors_vec = np.array([priors.get(lbl, 0.0) for lbl in labels], dtype="float32")
    tfidf_probs_dict = tfidf_clf.predict_proba(text)
    tfidf_vec = np.array([tfidf_probs_dict.get(lbl, 0.0) for lbl in labels], dtype="float32")
    return np.concatenate([sims.astype("float32"), priors_vec, tfidf_vec], axis=0)


def _encode_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return np.array(model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True), dtype="float32")


def _compute_class_weights(y_labels: np.ndarray) -> Dict[str, float]:
    """Compute balanced class weights and upweight minority labels like hr support."""
    unique_y = np.unique(y_labels)
    unexpected = set(unique_y) - set(CANONICAL_LABELS)
    if unexpected:
        raise RuntimeError(f"Unexpected labels in y_labels: {sorted(unexpected)}")
    base_weights = compute_class_weight(class_weight="balanced", classes=np.array(CANONICAL_LABELS), y=y_labels)
    weights: Dict[str, float] = {label: float(weight) for label, weight in zip(CANONICAL_LABELS, base_weights)}
    # Targeted boost for minority/underperforming classes to improve recall without collapsing overall accuracy.
    weights["hr support"] = weights.get("hr support", 1.0) * 1.3
    weights["purchase"] = weights.get("purchase", 1.0) * 1.1
    weights["administrative rights"] = weights.get("administrative rights", 1.0) * 1.1
    return weights


def train_head(
    ticket_path: Path,
    text_col: str,
    label_col: str,
    val_fraction: float,
    seed: int,
    C: float,
) -> None:
    df = _load_dataset(ticket_path, text_col, label_col)
    unique_labels = sorted(df[label_col].unique())
    unexpected = sorted(set(unique_labels) - set(CANONICAL_LABELS))
    if unexpected:
        raise RuntimeError(
            f"Unexpected labels in dataset: {unexpected}. Extend RouterGym/label_space.py mappings to canonical labels."
        )
    counts = df[label_col].value_counts().to_dict()
    print(f"[CalibratedHead] Label distribution: {counts}")
    label_to_idx = {lbl: idx for idx, lbl in enumerate(CANONICAL_LABELS)}
    y = df[label_col].map(label_to_idx).to_numpy(dtype="int64")
    texts = df[text_col].tolist()
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        texts, y, test_size=val_fraction, random_state=seed, stratify=y
    )

    centroids, cent_labels = _load_centroids()
    if cent_labels != CANONICAL_LABELS:
        raise ValueError("Centroid labels do not match canonical labels")

    tfidf_clf = TFIDFClassifier(labels=CANONICAL_LABELS)

    model = SentenceTransformer("intfloat/e5-small-v2")
    emb_train = _encode_texts(model, X_train_texts)
    emb_val = _encode_texts(model, X_val_texts)

    X_train_feat = np.stack(
        [
            _compute_features(emb, centroids, CANONICAL_LABELS, txt, tfidf_clf)
            for emb, txt in zip(emb_train, X_train_texts)
        ],
        axis=0,
    )
    X_val_feat = np.stack(
        [_compute_features(emb, centroids, CANONICAL_LABELS, txt, tfidf_clf) for emb, txt in zip(emb_val, X_val_texts)],
        axis=0,
    )

    mean = X_train_feat.mean(axis=0)
    std = X_train_feat.std(axis=0) + 1e-6
    X_train_std = (X_train_feat - mean) / std
    X_val_std = (X_val_feat - mean) / std

    class_weights = _compute_class_weights(y_train)
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        # Slightly upweight HR and other minority classes to raise recall without overpowering the majority.
        class_weight=class_weights,
        C=C,
    )
    clf.fit(X_train_std, y_train)

    feature_dim = X_train_feat.shape[1]
    train_preds = clf.predict(X_train_std)
    train_acc = float((train_preds == y_train).mean())

    val_preds = clf.predict(X_val_std)
    val_acc = float((val_preds == y_val).mean())
    print(f"[CalibratedHead] Validation accuracy: {val_acc:.3f}")
    for idx, lbl in enumerate(CANONICAL_LABELS):
        mask = y_val == idx
        acc = float((val_preds[mask] == y_val[mask]).mean()) if mask.any() else 0.0
        print(f"  {lbl:20s}: {acc:.3f}")

    HEAD_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        HEAD_OUT_PATH,
        labels=np.array(CANONICAL_LABELS, dtype=object),
        W=clf.coef_.astype("float32"),
        b=clf.intercept_.astype("float32"),
        feature_mean=mean.astype("float32"),
        feature_std=std.astype("float32"),
        feature_dim=np.array(feature_dim, dtype="int64"),
        version=np.array(HEAD_VERSION),
    )
    print(
        f"[CalibratedHead] Saved calibrated head to {HEAD_OUT_PATH} "
        f"(feature_dim={feature_dim}, version={HEAD_VERSION}) with "
        f"train_accuracy={train_acc:.3f} val_accuracy={val_acc:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated logistic head on encoder similarities + priors.")
    parser.add_argument("--ticket-path", type=Path, default=DEFAULT_TICKET_PATH, help="Path to tickets CSV.")
    parser.add_argument("--text-column", type=str, default=DEFAULT_TEXT_COL, help="Text column name.")
    parser.add_argument("--label-column", type=str, default=DEFAULT_LABEL_COL, help="Label column name.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--C", type=float, default=0.5, help="Inverse regularization strength for LogisticRegression.")
    args = parser.parse_args()

    train_head(
        ticket_path=args.ticket_path,
        text_col=args.text_column,
        label_col=args.label_column,
        val_fraction=args.val_fraction,
        seed=args.seed,
        C=args.C,
    )


if __name__ == "__main__":
    main()
