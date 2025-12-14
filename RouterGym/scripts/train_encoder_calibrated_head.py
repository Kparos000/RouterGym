"""Train a calibrated logistic head on top of encoder embeddings + lexical priors + TF-IDF scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from RouterGym.classifiers.tfidf_classifier import TFIDFClassifier
from RouterGym.classifiers.utils import apply_lexical_prior
from RouterGym.label_space import (
    CANONICAL_LABELS,
    ID_TO_LABEL,
    LABEL_NORMALIZATION_MAP,
    LABEL_TO_ID,
    canonical_label,
    canonicalize_label,
)

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("sentence-transformers is required to train calibrated head") from exc

DEFAULT_TICKET_PATH = Path("RouterGym/data/tickets/tickets.csv")
DEFAULT_TEXT_COL = "Document"
DEFAULT_LABEL_COL = "Topic_group"
HEAD_OUT_PATH = Path(__file__).resolve().parents[1] / "classifiers" / "encoder_calibrated_head.npz"
HEAD_VERSION = "1.0"
C_GRID = [0.5, 1.0, 2.0]
WEIGHT_MODES = ["balanced", "balanced_plus_boosts"]
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


def _train_val_split(
    texts: List[str],
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    return train_test_split(texts, labels, test_size=val_fraction, random_state=seed, stratify=labels)


def _compute_features(
    emb: np.ndarray,
    labels: List[str],
    text: str,
    tfidf_clf: TFIDFClassifier,
) -> np.ndarray:
    emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
    base = {label: 1.0 / max(len(labels), 1) for label in labels}
    priors = apply_lexical_prior(text, base, alpha=0.0, beta=1.0)
    priors_vec = np.array([priors.get(lbl, 0.0) for lbl in labels], dtype="float32")
    tfidf_probs_dict = tfidf_clf.predict_proba(text)
    tfidf_vec = np.array([tfidf_probs_dict.get(lbl, 0.0) for lbl in labels], dtype="float32")
    return np.concatenate([emb_norm.astype("float32"), priors_vec, tfidf_vec], axis=0)


def _encode_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return np.array(model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True), dtype="float32")


def _compute_class_weights(y_labels: np.ndarray, weight_mode: str = "balanced_plus_boosts") -> Dict[str, float]:
    """Compute balanced class weights from canonical string labels."""
    # Accept either canonical strings or integer IDs and normalize to strings for validation.
    normalized_labels_list: List[str] = []
    unexpected_raw: List[str] = []
    for lbl in y_labels:
        # Handle numeric labels even when dtype=object.
        if isinstance(lbl, (int, np.integer)):
            idx_int = int(lbl)
            if idx_int not in ID_TO_LABEL:
                raise RuntimeError(f"Unexpected label id {idx_int} in y_labels")
            normalized_labels_list.append(ID_TO_LABEL[idx_int])
        else:
            raw = str(lbl).strip().lower()
            norm = canonical_label(raw)
            if raw not in CANONICAL_LABELS and raw not in LABEL_NORMALIZATION_MAP:
                # Surface truly unknown labels instead of silently mapping to misc.
                unexpected_raw.append(raw)
            normalized_labels_list.append(norm)

    if unexpected_raw:
        raise RuntimeError(f"Unexpected labels in y_labels: {sorted(set(unexpected_raw))}")
    normalized_labels = np.array(normalized_labels_list, dtype=object)

    unique_y = np.unique(normalized_labels)
    base_weights = compute_class_weight(class_weight="balanced", classes=unique_y, y=normalized_labels)
    weights: Dict[str, float] = {label: float(weight) for label, weight in zip(unique_y, base_weights)}
    # Ensure every canonical label is present; default unseen classes to weight 1.0
    weights = {label: weights.get(label, 1.0) for label in CANONICAL_LABELS}

    if weight_mode == "balanced_plus_boosts":
        # Targeted boost for minority/underperforming classes to improve recall without collapsing overall accuracy.
        weights["hr support"] = weights.get("hr support", 1.0) * 1.3
        weights["purchase"] = weights.get("purchase", 1.0) * 1.1
        weights["administrative rights"] = weights.get("administrative rights", 1.0) * 1.1
    elif weight_mode == "balanced":
        pass
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    return weights


def _select_best_config(
    candidates: List[Dict[str, Any]],
    weight_mode_order: List[str] = WEIGHT_MODES,
) -> Dict[str, Any]:
    """Select the best candidate by val_acc, then macro_f1, then tie-breaker on C/weight order."""

    def weight_mode_rank(mode: str) -> int:
        return weight_mode_order.index(mode) if mode in weight_mode_order else len(weight_mode_order)

    candidates_sorted = sorted(
        candidates,
        key=lambda c: (
            -c["val_acc"],
            -c["macro_f1"],
            weight_mode_rank(c["weight_mode"]),
            c["C"],
        ),
    )
    return candidates_sorted[0]


def train_head(
    ticket_path: Path,
    text_col: str,
    label_col: str,
    val_fraction: float,
    seed: int,
    C: float | None,
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
    y_labels = df[label_col].to_numpy(dtype=object)
    y_ids = df[label_col].map(LABEL_TO_ID).to_numpy(dtype="int64")
    texts = df[text_col].tolist()
    X_train_texts, X_val_texts, y_train_ids, y_val_ids, y_train_labels, y_val_labels = train_test_split(
        texts, y_ids, y_labels, test_size=val_fraction, random_state=seed, stratify=y_ids
    )

    tfidf_clf = TFIDFClassifier(labels=CANONICAL_LABELS)

    model = SentenceTransformer("intfloat/e5-small-v2")
    emb_train = _encode_texts(model, X_train_texts)
    emb_val = _encode_texts(model, X_val_texts)

    X_train_feat = np.stack(
        [
            _compute_features(emb, CANONICAL_LABELS, txt, tfidf_clf)
            for emb, txt in zip(emb_train, X_train_texts)
        ],
        axis=0,
    )
    X_val_feat = np.stack(
        [_compute_features(emb, CANONICAL_LABELS, txt, tfidf_clf) for emb, txt in zip(emb_val, X_val_texts)],
        axis=0,
    )

    mean = X_train_feat.mean(axis=0)
    std = X_train_feat.std(axis=0) + 1e-6
    X_train_std = (X_train_feat - mean) / std
    X_val_std = (X_val_feat - mean) / std

    candidates: List[Dict[str, Any]] = []
    if C is not None:
        sweep_C = [C]
        sweep_weight_modes = ["balanced_plus_boosts"]
    else:
        sweep_C = C_GRID
        sweep_weight_modes = WEIGHT_MODES

    for c_value in sweep_C:
        for weight_mode in sweep_weight_modes:
            class_weights_label = _compute_class_weights(y_train_labels, weight_mode=weight_mode)
            class_weights = {LABEL_TO_ID[label]: weight for label, weight in class_weights_label.items()}
            clf = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                class_weight=class_weights,
                C=c_value,
            )
            clf.fit(X_train_std, y_train_ids)

            train_preds = clf.predict(X_train_std)
            val_preds = clf.predict(X_val_std)
            val_acc = float((val_preds == y_val_ids).mean())
            macro_f1 = float(f1_score(y_val_ids, val_preds, average="macro"))
            train_acc = float((train_preds == y_train_ids).mean())
            candidates.append(
                {
                    "C": c_value,
                    "weight_mode": weight_mode,
                    "clf": clf,
                    "class_weights_label": class_weights_label,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "macro_f1": macro_f1,
                    "val_preds": val_preds,
                }
            )

    print("[CalibratedHead] Hyperparameter sweep results:")
    print("C\tweight_mode\t\tval_acc\tmacro_F1")
    for cand in candidates:
        print(
            f"{cand['C']:.2f}\t{cand['weight_mode']:<20s}\t{cand['val_acc']:.3f}\t{cand['macro_f1']:.3f}"
        )

    best = _select_best_config(candidates)
    print(
        f"[CalibratedHead] Selected config: C={best['C']}, weight_mode={best['weight_mode']} "
        f"(val_acc={best['val_acc']:.3f}, macro_F1={best['macro_f1']:.3f})"
    )

    best_clf: LogisticRegression = best["clf"]
    val_preds = best["val_preds"]
    train_acc = best["train_acc"]
    val_acc = best["val_acc"]

    feature_dim = X_train_feat.shape[1]
    print(f"[CalibratedHead] Validation accuracy: {val_acc:.3f}")
    for idx, lbl in enumerate(CANONICAL_LABELS):
        mask = y_val_ids == idx
        acc = float((val_preds[mask] == y_val_ids[mask]).mean()) if mask.any() else 0.0
        print(f"  {lbl:20s}: {acc:.3f}")

    HEAD_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        HEAD_OUT_PATH,
        labels=np.array(CANONICAL_LABELS, dtype=object),
        W=best_clf.coef_.astype("float32"),
        b=best_clf.intercept_.astype("float32"),
        feature_mean=mean.astype("float32"),
        feature_std=std.astype("float32"),
        feature_dim=np.array(feature_dim, dtype="int64"),
        version=np.array(HEAD_VERSION),
        C=np.array(best["C"]),
        weight_mode=np.array(best["weight_mode"]),
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
    parser.add_argument("--C", type=float, default=None, help="Inverse regularization strength for LogisticRegression.")
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
