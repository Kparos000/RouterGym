from __future__ import annotations

import numpy as np
import pandas as pd

from RouterGym.scripts.analyze_encoder_confidence_curve import _compute_metrics


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
