from __future__ import annotations

import numpy as np
import pandas as pd

from RouterGym.scripts.analyze_encoder_confidence_curve import _compute_metrics, _load_calibrated_head
from pathlib import Path


def test_compute_metrics_basic() -> None:
    probs = np.array(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.9, 0.1],
        ]
    )
    gold = np.array([0, 1, 0], dtype=int)
    df = _compute_metrics(probs, gold, tau_list=[0.5, 0.8])
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    # tau=0.5 covers all 3; accuracy = 2/3 (second item incorrect)
    row_05 = df.iloc[0]
    assert abs(row_05["coverage"] - 1.0) < 1e-6
    assert abs(row_05["accuracy"] - (2 / 3)) < 1e-6
    # tau=0.8 covers only third; correct -> accuracy 1.0
    row_08 = df.iloc[1]
    assert abs(row_08["coverage"] - (1 / 3)) < 1e-6
    assert abs(row_08["accuracy"] - 1.0) < 1e-6


def test_load_calibrated_head_mlp(tmp_path: Path) -> None:
    labels = np.array(["access", "hardware"], dtype=object)
    feature_dim = 4
    layer_weights = np.array(
        [np.ones((feature_dim, 2), dtype="float32"), np.ones((2, len(labels)), dtype="float32")],
        dtype=object,
    )
    layer_biases = np.array([np.zeros(2, dtype="float32"), np.zeros(len(labels), dtype="float32")], dtype=object)
    head_path = tmp_path / "head.npz"
    np.savez(
        head_path,
        labels=labels,
        feature_mean=np.zeros(feature_dim, dtype="float32"),
        feature_std=np.ones(feature_dim, dtype="float32"),
        feature_dim=np.array(feature_dim, dtype="int64"),
        head_type=np.array("mlp"),
        layer_weights=layer_weights,
        layer_biases=layer_biases,
    )
    head = _load_calibrated_head(head_path)
    assert head["head_type"] == "mlp"
    assert "layer_weights" in head and "layer_biases" in head
    assert head["feature_dim"] == feature_dim
