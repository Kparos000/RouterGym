"""Sentence-encoder style classifier using cosine similarity and optional centroids."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - fallback when dependency missing
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from RouterGym.classifiers.utils import (
    ClassifierMetadata,
    ClassifierProtocol,
    DEFAULT_LABELS,
    canonical_label,
    normalize_probabilities,
    apply_lexical_prior,
)

_PROTOTYPES: Dict[str, str] = {
    "access": "password reset login mfa sso otp locked out account",
    "administrative rights": "admin privilege elevated permission group membership entitlement role change",
    "hardware": "laptop printer dock monitor battery broken keyboard mouse device",
    "hr support": "benefits payroll leave hr question vacation",
    "purchase": "buy purchase order license invoice billing subscription renewal procurement payment quote",
    "miscellaneous": "general inquiry misc other",
}


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class EncoderClassifier(ClassifierProtocol):
    """Cosine similarity classifier using MiniLM/E5 embeddings or hash/centroid fallback."""

    def __init__(
        self,
        labels: Optional[Iterable[str]] = None,
        model_name: str = "intfloat/e5-small-v2",
        embedding_dimension: int = 16,
        use_lexical_prior: bool = True,
    ) -> None:
        self.labels = [canonical_label(lbl) for lbl in (labels or DEFAULT_LABELS)]
        self.metadata = ClassifierMetadata(
            name="Encoder",
            mode="encoder",
            provider="sentence-transformers",
            model_reference=model_name,
            token_cost=0.0002,
            latency_ms=12.0,
            description="MiniLM/E5-style embedding classifier (centroid-enhanced when available)",
        )
        self.embedding_dimension = embedding_dimension
        env_flag = os.getenv("ROUTERGYM_ENCODER_USE_LEXICAL_PRIOR", "").lower()
        if env_flag in {"0", "false"}:
            self.use_lexical_prior = False
        else:
            self.use_lexical_prior = use_lexical_prior
        self._encoder: SentenceTransformer | None = None
        self._centroid_labels: List[str] = []
        self._centroids = None
        if SentenceTransformer is not None:
            try:
                self._encoder = SentenceTransformer(model_name)  # type: ignore[arg-type]
            except Exception:
                self._encoder = None
        self._prototype_vectors = {
            canonical_label(label): self._encode_text(text)
            for label, text in _PROTOTYPES.items()
        }
        self._maybe_load_centroids()

    def _maybe_load_centroids(self) -> None:
        """Load centroid file if available; otherwise stay on prototype path."""
        path = Path(__file__).resolve().parent / "encoder_centroids.npz"
        if np is None or not path.exists():
            if not path.exists():
                print("[EncoderClassifier] Centroid file missing; run `python -m RouterGym.scripts.train_encoder_centroids` to improve accuracy.")
            return
        try:
            data = np.load(path, allow_pickle=True)
            labels = list(map(str, data["labels"].tolist()))
            centroids = np.array(data["centroids"], dtype="float32")
            if centroids.ndim != 2 or len(labels) != centroids.shape[0]:
                print("[EncoderClassifier] Invalid centroid file; falling back to prototypes.")
                return
            # Normalize centroids once for cosine similarity.
            norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
            self._centroids = centroids / norms
            self._centroid_labels = [canonical_label(lbl) for lbl in labels]
            self.labels = self._centroid_labels
            self.metadata.description += " (centroid mode)"
        except Exception:
            print("[EncoderClassifier] Failed to load centroids; falling back to prototypes.")
            self._centroids = None
            self._centroid_labels = []

    def _hash_vector(self, text: str) -> List[float]:
        vector = [0.0] * self.embedding_dimension
        if not text:
            return vector
        for idx, char in enumerate(text.lower()):
            if char.isalpha():
                bucket = (idx + ord(char)) % self.embedding_dimension
                vector[bucket] += 1.0
        return _normalize(vector)

    def _encode_text(self, text: str) -> List[float]:
        if self._encoder is not None:
            try:
                array = self._encoder.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
                return list(map(float, array))
            except Exception:
                return self._hash_vector(text)
        return self._hash_vector(text)

    def _cosine(self, a: List[float], b: List[float]) -> float:
        size = min(len(a), len(b))
        return sum(a[i] * b[i] for i in range(size))

    def predict_proba(self, text: str) -> Dict[str, float]:
        if self._centroids is not None and np is not None and self._encoder is not None:
            try:
                emb = self._encoder.encode(text or "", normalize_embeddings=True)  # type: ignore[attr-defined]
                emb_np = np.array(emb, dtype="float32")
                emb_norm = emb_np / (np.linalg.norm(emb_np) + 1e-9)
                sims = emb_norm @ self._centroids.T
                # Softmax probabilities
                logits = sims.astype("float64")
                logits = logits - logits.max()
                exp = np.exp(logits)
                probs = exp / exp.sum()
                centroid_probs = {lbl: float(probs[idx]) for idx, lbl in enumerate(self._centroid_labels)}
                if self.use_lexical_prior:
                    return apply_lexical_prior(text, centroid_probs)
                # Pure centroid path (no lexical prior)
                return normalize_probabilities(centroid_probs, self._centroid_labels)
            except Exception:
                print("[EncoderClassifier] Warning: centroid inference failed, falling back to prototypes.")

        text_vec = self._encode_text(text or "")
        scores = {
            label: max(0.0, self._cosine(text_vec, self._prototype_vectors.get(label, [])))
            for label in self.labels
        }
        base = normalize_probabilities(scores, self.labels)
        return apply_lexical_prior(text, base) if self.use_lexical_prior else base

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.__getitem__)


__all__ = ["EncoderClassifier"]
