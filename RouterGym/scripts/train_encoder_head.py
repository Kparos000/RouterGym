"""Train a supervised linear softmax head on frozen E5 embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from RouterGym.label_space import CANONICAL_LABELS, canonical_label

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("sentence-transformers is required to train the encoder head") from exc


DEFAULT_TICKET_PATH = Path("RouterGym/data/tickets/tickets.csv")
DEFAULT_TEXT_COL = "Document"
DEFAULT_LABEL_COL = "Topic_group"
HEAD_PATH = Path(__file__).resolve().parents[1] / "classifiers" / "encoder_head.npz"


def _load_dataset(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{text_col}' and '{label_col}' in {path}")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].apply(canonical_label)
    df = df[df[label_col].isin(CANONICAL_LABELS)]
    return df.reset_index(drop=True)


def _encode_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return np.array(model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True), dtype="float32")


def _softmax_head_train(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    l2: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = np.zeros((num_classes, d), dtype="float32")
    b = np.zeros((num_classes,), dtype="float32")
    for _ in range(epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            Xb = X[batch_idx]
            yb = y[batch_idx]
            logits = Xb @ W.T + b  # (batch, classes)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits, dtype="float64")
            probs = exp / exp.sum(axis=1, keepdims=True)
            one_hot = np.eye(num_classes, dtype="float32")[yb]
            grad_logits = (probs - one_hot) / Xb.shape[0]
            grad_W = grad_logits.T @ Xb + l2 * W
            grad_b = grad_logits.sum(axis=0)
            W -= lr * grad_W.astype("float32")
            b -= lr * grad_b.astype("float32")
    return W.astype("float32"), b.astype("float32")


def _evaluate(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, labels: List[str]) -> Tuple[float, Dict[str, float]]:
    logits = X @ W.T + b
    preds = logits.argmax(axis=1)
    overall = float((preds == y).mean()) if len(y) else 0.0
    per_class: Dict[str, float] = {}
    for idx, lbl in enumerate(labels):
        mask = y == idx
        acc = float((preds[mask] == y[mask]).mean()) if mask.any() else 0.0
        per_class[lbl] = acc
    return overall, per_class


def train_head(
    ticket_path: Path,
    text_col: str,
    label_col: str,
    val_fraction: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2_weight: float,
    seed: int,
) -> None:
    df = _load_dataset(ticket_path, text_col, label_col)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(CANONICAL_LABELS)}
    y = df[label_col].map(label_to_idx).to_numpy(dtype="int64")
    texts = df[text_col].tolist()
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        texts, y, test_size=val_fraction, random_state=seed, stratify=y
    )

    model = SentenceTransformer("intfloat/e5-small-v2")
    X_train = _encode_texts(model, X_train_texts)
    X_val = _encode_texts(model, X_val_texts)

    W, b = _softmax_head_train(
        X_train,
        y_train,
        num_classes=len(CANONICAL_LABELS),
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        l2=l2_weight,
        seed=seed,
    )

    val_acc, per_class = _evaluate(X_val, y_val, W, b, CANONICAL_LABELS)
    print(f"[EncoderHead] Validation accuracy: {val_acc:.3f}")
    for lbl in CANONICAL_LABELS:
        print(f"  {lbl:20s}: {per_class.get(lbl, 0.0):.3f}")

    HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(HEAD_PATH, labels=np.array(CANONICAL_LABELS, dtype=object), W=W, b=b)
    print(f"[EncoderHead] Saved linear head to {HEAD_PATH} with val_accuracy={val_acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised linear head on frozen E5 embeddings.")
    parser.add_argument("--ticket-path", type=Path, default=DEFAULT_TICKET_PATH, help="Path to tickets CSV.")
    parser.add_argument("--text-column", type=str, default=DEFAULT_TEXT_COL, help="Text column name.")
    parser.add_argument("--label-column", type=str, default=DEFAULT_LABEL_COL, help="Label column name.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (default 0.2).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--l2-weight", type=float, default=0.01, help="L2 regularization weight.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_head(
        ticket_path=args.ticket_path,
        text_col=args.text_column,
        label_col=args.label_column,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2_weight=args.l2_weight,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
