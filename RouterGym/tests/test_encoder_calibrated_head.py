from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from RouterGym.classifiers import encoder_classifier as enc


class DummyEncoder:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:  # pragma: no cover - simple stub
        return self.vector


def _build_fake_head(path: Path, labels: list[str]) -> None:
    feature_dim = len(labels) * 3
    W = np.zeros((len(labels), feature_dim), dtype="float32")
    W[0, 0] = 2.0
    W[1, 0] = -2.0
    b = np.zeros(len(labels), dtype="float32")
    mean = np.zeros(feature_dim, dtype="float32")
    std = np.ones(feature_dim, dtype="float32")
    np.savez(
        path,
        labels=np.array(labels, dtype=object),
        W=W,
        b=b,
        feature_mean=mean,
        feature_std=std,
    )


def test_calibrated_head_probability_flow(monkeypatch: Any, tmp_path: Path) -> None:
    labels = ["access", "administrative rights"]
    head_path = tmp_path / "encoder_calibrated_head.npz"
    _build_fake_head(head_path, labels)

    monkeypatch.setattr(enc, "CALIBRATED_HEAD_PATH", head_path)
    monkeypatch.setattr(enc.EncoderClassifier, "_maybe_load_centroids", lambda self: None)
    monkeypatch.delenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", raising=False)

    classifier = enc.EncoderClassifier(labels=labels, use_lexical_prior=False, head_mode="calibrated", embedding_dimension=2)
    classifier._encoder = DummyEncoder(np.array([1.0, 0.0], dtype="float32"))  # type: ignore[attr-defined, assignment]
    classifier._centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")  # type: ignore[attr-defined, assignment]

    probs: Dict[str, float] = classifier.predict_proba("payroll ticket")
    assert classifier._head_mode_active == "calibrated"
    assert set(probs.keys()) == set(labels)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["access"] > probs["administrative rights"]


def test_head_mode_centroid(monkeypatch: Any, tmp_path: Path) -> None:
    labels = ["access", "administrative rights"]
    monkeypatch.setattr(enc, "CALIBRATED_HEAD_PATH", tmp_path / "missing.npz")
    monkeypatch.setattr(enc.EncoderClassifier, "_maybe_load_centroids", lambda self: None)
    monkeypatch.delenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", raising=False)

    classifier = enc.EncoderClassifier(labels=labels, use_lexical_prior=False, head_mode="centroid", embedding_dimension=2)
    assert classifier._head_mode_active == "centroid"


def test_compute_class_weights_upweights_hr(monkeypatch: Any) -> None:
    from RouterGym.scripts import train_encoder_calibrated_head as trainer
    import numpy as np

    # Simulate imbalanced labels where hr support is rare.
    y_labels = np.array([0, 0, 0, 1, 2, 3, 3, 4, 5, 5, 5], dtype=int)
    # Map indices to canonical labels
    label_mapping = {idx: label for idx, label in enumerate(trainer.CANONICAL_LABELS)}
    y_named = np.array([label_mapping[idx] for idx in y_labels])

    weights = trainer._compute_class_weights(y_named)
    assert weights["hr support"] > weights["access"]
    assert weights["hr support"] > weights.get("miscellaneous", 0.0)


def test_auto_mode_raises_without_calibrated_head(monkeypatch: Any, tmp_path: Path) -> None:
    labels = ["access", "administrative rights"]
    monkeypatch.setattr(enc, "CALIBRATED_HEAD_PATH", tmp_path / "missing.npz")
    monkeypatch.setattr(enc.EncoderClassifier, "_maybe_load_centroids", lambda self: None)
    monkeypatch.delenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", raising=False)
    with pytest.raises(RuntimeError):
        enc.EncoderClassifier(labels=labels, use_lexical_prior=False, head_mode="auto", embedding_dimension=2)


def test_auto_mode_allows_fallback_when_env_set(monkeypatch: Any, tmp_path: Path) -> None:
    labels = ["access", "administrative rights"]
    monkeypatch.setenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", "1")
    monkeypatch.setattr(enc, "CALIBRATED_HEAD_PATH", tmp_path / "missing.npz")
    monkeypatch.setattr(enc.EncoderClassifier, "_maybe_load_centroids", lambda self: None)
    classifier = enc.EncoderClassifier(labels=labels, use_lexical_prior=False, head_mode="auto", embedding_dimension=2)
    assert classifier._backend_name == "encoder_centroid"
    monkeypatch.delenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", raising=False)


def test_auto_mode_uses_calibrated_when_present(monkeypatch: Any, tmp_path: Path) -> None:
    labels = ["access", "administrative rights"]
    head_path = tmp_path / "encoder_calibrated_head.npz"
    _build_fake_head(head_path, labels)
    monkeypatch.delenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", raising=False)
    monkeypatch.setattr(enc, "CALIBRATED_HEAD_PATH", head_path)
    monkeypatch.setattr(enc.EncoderClassifier, "_maybe_load_centroids", lambda self: None)
    classifier = enc.EncoderClassifier(labels=labels, use_lexical_prior=False, head_mode="auto", embedding_dimension=2)
    assert classifier._backend_name == "encoder_calibrated"
