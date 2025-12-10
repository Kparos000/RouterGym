"""Tests for encoder classifier linear head support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np

import RouterGym.classifiers.encoder_classifier as enc_cls
from RouterGym.classifiers.encoder_classifier import EncoderClassifier


def _write_head(path: Path, labels: List[str], W: np.ndarray, b: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=np.array(labels, dtype=object), W=W, b=b)


def test_linear_head_loads_and_sets_mode(monkeypatch: Any, tmp_path: Path) -> None:
    labels = ["access", "administrative rights"]
    W = np.zeros((2, 3), dtype="float32")
    b = np.zeros((2,), dtype="float32")
    head_path = tmp_path / "encoder_head.npz"
    _write_head(head_path, labels, W, b)

    monkeypatch.setattr(enc_cls, "HEAD_PATH", head_path)
    monkeypatch.setattr(enc_cls, "SentenceTransformer", None)

    clf = EncoderClassifier(labels=labels, use_lexical_prior=False, head_mode="linear", embedding_dimension=3, model_name="dummy")
    assert clf._head_mode_active == "linear"
    assert clf._linear_W is not None and clf._linear_W.shape == (2, 3)
    assert clf._linear_b is not None and clf._linear_b.shape == (2,)


def test_linear_head_predicts_from_weights(monkeypatch: Any, tmp_path: Path) -> None:
    labels = ["access", "administrative rights"]
    W = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype="float32")
    b = np.zeros((2,), dtype="float32")
    head_path = tmp_path / "encoder_head.npz"
    _write_head(head_path, labels, W, b)

    monkeypatch.setattr(enc_cls, "HEAD_PATH", head_path)
    monkeypatch.setattr(enc_cls, "SentenceTransformer", None)

    clf = EncoderClassifier(labels=labels, use_lexical_prior=False, head_mode="linear", embedding_dimension=3, model_name="dummy")
    monkeypatch.setattr(clf, "_encode_text", lambda text: [1.0, 0.0, 0.0])

    probs = clf.predict_proba("any text")
    assert sum(probs.values()) == 1.0 or abs(sum(probs.values()) - 1.0) < 1e-3
    assert clf.predict_label("any text") == "access"
