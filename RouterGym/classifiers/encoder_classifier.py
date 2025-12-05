"""Sentence-encoder style classifier using cosine similarity."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - fallback when dependency missing
    SentenceTransformer = None  # type: ignore

from RouterGym.classifiers.utils import (
    ClassifierMetadata,
    ClassifierProtocol,
    DEFAULT_LABELS,
    canonical_label,
    normalize_probabilities,
)

_PROTOTYPES: Dict[str, str] = {
    "access": "password reset login mfa sso otp locked out",
    "administrative rights": "admin privilege elevated permission install software",
    "hardware": "laptop printer dock monitor battery broken",
    "hr support": "benefits payroll leave hr question",
    "internal project": "internal repository analytics project access",
    "miscellaneous": "general inquiry misc other",
    "purchase": "buy purchase order license invoice",
    "storage": "shared drive storage quota full",
}


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class EncoderClassifier(ClassifierProtocol):
    """Cosine similarity classifier using MiniLM/E5 embeddings or hash fallback."""

    def __init__(
        self,
        labels: Optional[Iterable[str]] = None,
        model_name: str = "intfloat/e5-small-v2",
        embedding_dimension: int = 16,
    ) -> None:
        self.labels = [canonical_label(lbl) for lbl in (labels or DEFAULT_LABELS)]
        self.metadata = ClassifierMetadata(
            name="Encoder",
            mode="encoder",
            provider="sentence-transformers",
            model_reference=model_name,
            token_cost=0.0002,
            latency_ms=12.0,
            description="MiniLM/E5-style embedding classifier",
        )
        self.embedding_dimension = embedding_dimension
        self._encoder: SentenceTransformer | None = None
        if SentenceTransformer is not None:
            try:
                self._encoder = SentenceTransformer(model_name)  # type: ignore[arg-type]
            except Exception:
                self._encoder = None
        self._prototype_vectors = {
            canonical_label(label): self._encode_text(text)
            for label, text in _PROTOTYPES.items()
        }

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
        text_vec = self._encode_text(text or "")
        scores = {
            label: max(0.0, self._cosine(text_vec, self._prototype_vectors.get(label, [])))
            for label in self.labels
        }
        return normalize_probabilities(scores, self.labels)

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.__getitem__)


__all__ = ["EncoderClassifier"]
