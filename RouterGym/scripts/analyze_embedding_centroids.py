"""Analyze E5 embedding centroid separability via pairwise cosine similarity."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


ENCODER_CENTROIDS_PATH = Path(__file__).resolve().parents[1] / "classifiers" / "encoder_centroids.npz"


def load_centroids(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    labels = data["labels"]
    centroids = data["centroids"]
    return labels, centroids


def normalize_centroids(centroids: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    # Avoid division by zero for any degenerate vectors.
    norms = np.clip(norms, 1e-12, None)
    return centroids / norms


def cosine_similarity_matrix(centroids: np.ndarray) -> np.ndarray:
    normalized = normalize_centroids(centroids)
    return normalized @ normalized.T


def _format_matrix(labels: Iterable[str], matrix: np.ndarray) -> str:
    labels_list = list(labels)
    col_width = max(len(label) for label in labels_list + ["miscellaneous"]) + 2
    header = " " * (col_width + 1) + "".join(f"{label:{col_width}}" for label in labels_list)
    rows = [header]
    for label, row in zip(labels_list, matrix):
        formatted_values = "".join(f"{value:>{col_width}.2f}" for value in row)
        rows.append(f"{label:{col_width}} {formatted_values}")
    return "\n".join(rows)


def _off_diagonal_summary(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, tuple[str, str, float], tuple[str, str, float]]:
    n = matrix.shape[0]
    mean_val = float(np.sum(matrix) - np.trace(matrix)) / (n * (n - 1)) if n > 1 else float("nan")

    min_val = float("inf")
    max_val = float("-inf")
    min_pair = ("", "", float("nan"))
    max_pair = ("", "", float("nan"))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = float(matrix[i, j])
            if val < min_val:
                min_val = val
                min_pair = (str(labels[i]), str(labels[j]), val)
            if val > max_val:
                max_val = val
                max_pair = (str(labels[i]), str(labels[j]), val)

    return mean_val, min_pair, max_pair


def analyze_centroids(path: Path) -> None:
    labels, centroids = load_centroids(path)
    cos_matrix = cosine_similarity_matrix(centroids)

    labels_list = [str(label) for label in labels]
    print("Labels:", ", ".join(labels_list))
    print("\nCosine similarity matrix:")
    print(_format_matrix(labels_list, cos_matrix))

    mean_off, min_pair, max_pair = _off_diagonal_summary(cos_matrix, labels)
    print("\nSummary:")
    print(f"Mean off-diagonal similarity: {mean_off:.4f}")
    print(f"Min similarity: {min_pair[2]:.4f} ({min_pair[0]} vs {min_pair[1]})")
    print(f"Max similarity: {max_pair[2]:.4f} ({max_pair[0]} vs {max_pair[1]})")


def main() -> None:
    analyze_centroids(ENCODER_CENTROIDS_PATH)


if __name__ == "__main__":
    main()
