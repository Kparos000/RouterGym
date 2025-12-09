"""Tests for analyze_embedding_centroids utility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from RouterGym.scripts import analyze_embedding_centroids as centroid_script


def test_cosine_matrix_output(monkeypatch: Any, capsys: Any, tmp_path: Path) -> None:
    labels = np.array(["a", "b"], dtype="<U1")
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    npz_path = tmp_path / "encoder_centroids.npz"
    np.savez(npz_path, labels=labels, centroids=centroids)

    monkeypatch.setattr(centroid_script, "ENCODER_CENTROIDS_PATH", npz_path)

    centroid_script.main()

    captured = capsys.readouterr().out
    assert "Labels: a, b" in captured
    assert "Cosine similarity matrix" in captured
    assert "1.00" in captured
    assert "0.00" in captured
    assert "Mean off-diagonal similarity" in captured
    assert "Min similarity" in captured
    assert "Max similarity" in captured
