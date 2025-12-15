"""Analyze calibrated encoder confidence vs accuracy on the validation split.

Outputs a CSV at RouterGym/results/analysis/encoder_confidence_curve.csv and prints a summary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from RouterGym.classifiers.tfidf_classifier import TFIDFClassifier
from RouterGym.label_space import CANONICAL_LABELS, LABEL_TO_ID, canonicalize_label
from RouterGym.scripts.train_encoder_calibrated_head import (
    _compute_features,
    _load_dataset,
    _train_val_split,
)

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("sentence-transformers is required for encoder analysis") from exc


RESULTS_PATH = Path("RouterGym/results/analysis/encoder_confidence_curve.csv")


def _load_calibrated_head(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Calibrated head not found at {path}. "
            "Train it first with `python -m RouterGym.scripts.train_encoder_calibrated_head` "
            "or pass --head-path pointing to the generated encoder_calibrated_head.npz."
        )
    data = np.load(path, allow_pickle=True)
    required = ["labels", "W", "b", "feature_mean", "feature_std", "feature_dim"]
    for key in required:
        if key not in data:
            raise RuntimeError(f"Missing key '{key}' in calibrated head file at {path}")
    head_type = str(data.get("head_type", "logreg"))
    head: Dict[str, Any] = {
        "labels": data["labels"],
        "feature_mean": np.asarray(data["feature_mean"], dtype="float32"),
        "feature_std": np.asarray(data["feature_std"], dtype="float32"),
        "feature_dim": int(data["feature_dim"]),
        "head_type": head_type,
        "weight_mode": data.get("weight_mode"),
    }
    if head_type == "mlp":
        if "layer_weights" not in data or "layer_biases" not in data:
            raise RuntimeError("Calibrated MLP head missing layer weights/biases.")
        head["layer_weights"] = [np.asarray(w, dtype="float32") for w in data["layer_weights"]]
        head["layer_biases"] = [np.asarray(bias, dtype="float32") for bias in data["layer_biases"]]
        head["alpha"] = data.get("alpha")
    else:
        head["W"] = np.asarray(data["W"], dtype="float32")
        head["b"] = np.asarray(data["b"], dtype="float32")
        head["C"] = data.get("C")
    return head


def _compute_metrics(
    probabilities: np.ndarray,
    gold_labels: np.ndarray,
    tau_list: Sequence[float],
) -> pd.DataFrame:
    preds = probabilities.argmax(axis=1)
    max_conf = probabilities.max(axis=1)
    rows = []
    for tau in tau_list:
        mask = max_conf >= tau
        n = int(mask.sum())
        coverage = float(n) / float(len(max_conf)) if len(max_conf) else 0.0
        if n == 0:
            acc = np.nan
        else:
            acc = float((preds[mask] == gold_labels[mask]).mean())
        rows.append(
            {
                "threshold": tau,
                "coverage": coverage,
                "accuracy": acc,
                "n_examples": n,
            }
        )
    return pd.DataFrame(rows)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype="float64")
    return exp / exp.sum(axis=-1, keepdims=True)


def _forward_head(head: Dict[str, Any], X_std: np.ndarray) -> np.ndarray:
    if head["head_type"] == "mlp":
        weights: List[np.ndarray] = head["layer_weights"]
        biases: List[np.ndarray] = head["layer_biases"]
        z = X_std
        for idx, (W, b) in enumerate(zip(weights, biases)):
            z = z @ W + b
            if idx < len(weights) - 1:
                z = np.maximum(0.0, z)
        return _softmax(z)
    # default logreg
    W = head["W"]
    b = head["b"]
    logits = X_std @ W + b
    return _softmax(logits)


def run_analysis(
    ticket_path: Path,
    text_col: str,
    label_col: str,
    val_fraction: float,
    seed: int,
    head_path: Path,
    output_path: Path,
) -> None:
    df = _load_dataset(ticket_path, text_col, label_col)
    texts = df[text_col].tolist()
    labels = df[label_col].to_numpy()
    _, val_texts, _, y_val = _train_val_split(texts, labels, val_fraction, seed)

    head = _load_calibrated_head(head_path)
    feature_dim = head["feature_dim"]
    label_order = [str(lbl) for lbl in head["labels"].tolist()]
    if label_order != CANONICAL_LABELS:
        raise RuntimeError("Calibrated head labels do not match canonical labels.")

    tfidf_clf = TFIDFClassifier(labels=CANONICAL_LABELS)
    model = SentenceTransformer("intfloat/e5-small-v2")
    embeddings = model.encode(val_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    X_val = np.stack(
        [
            _compute_features(np.array(emb, dtype="float32"), CANONICAL_LABELS, text, tfidf_clf)
            for emb, text in zip(embeddings, val_texts)
        ],
        axis=0,
    )
    if X_val.shape[1] != feature_dim:
        raise RuntimeError(f"Feature dimension mismatch: built {X_val.shape[1]}, expected {feature_dim}")

    # Normalize features
    mean = head["feature_mean"]
    std = head["feature_std"]
    std_safe = np.where(std > 1e-6, std, 1.0)
    X_val_std = (X_val - mean) / std_safe

    probs = _forward_head(head, X_val_std)

    y_val_ids = np.array([LABEL_TO_ID[canonicalize_label(lbl)] for lbl in y_val], dtype=int)
    tau_list = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    df_metrics = _compute_metrics(probs, y_val_ids, tau_list)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_path, index=False)

    print("Threshold\tCoverage\tAccuracy\tN_examples")
    for _, row in df_metrics.iterrows():
        acc_display = "nan" if pd.isna(row["accuracy"]) else f"{row['accuracy']:.3f}"
        print(f"{row['threshold']:.2f}\t\t{row['coverage']:.3f}\t\t{acc_display}\t{int(row['n_examples'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze calibrated encoder confidence vs accuracy on validation split.")
    parser.add_argument("--ticket-path", type=Path, default=Path("RouterGym/data/tickets/tickets.csv"), help="Tickets CSV.")
    parser.add_argument("--text-column", type=str, default="Document", help="Text column name.")
    parser.add_argument("--label-column", type=str, default="Topic_group", help="Label column name.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (must match training).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (must match training).")
    parser.add_argument("--head-path", type=Path, default=Path("RouterGym/classifiers/encoder_calibrated_head.npz"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=RESULTS_PATH,
        help="Where to write CSV (default RouterGym/results/analysis/encoder_confidence_curve.csv).",
    )
    args = parser.parse_args()
    run_analysis(
        ticket_path=args.ticket_path,
        text_col=args.text_column,
        label_col=args.label_column,
        val_fraction=args.val_fraction,
        seed=args.seed,
        head_path=args.head_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
