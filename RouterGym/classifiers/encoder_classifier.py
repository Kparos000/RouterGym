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

HEAD_PATH = Path(__file__).resolve().parent / "encoder_head.npz"

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
        head_mode: str = "auto",
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
        self._head_mode_config = head_mode
        self._head_mode_active = "centroid"
        self._linear_W: Optional[np.ndarray] = None if np is not None else None
        self._linear_b: Optional[np.ndarray] = None if np is not None else None
        self._linear_labels: Optional[List[str]] = None
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
        self._resolve_head_mode()

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

    def _try_load_linear_head(self) -> None:
        """Load linear softmax head if present and valid."""
        if np is None:
            return
        path = HEAD_PATH
        if not path.exists():
            return
        try:
            data = np.load(path, allow_pickle=True)
            labels = [canonical_label(lbl) for lbl in data["labels"].tolist()]
            W = np.asarray(data["W"], dtype="float32")
            b = np.asarray(data["b"], dtype="float32")
            if W.ndim != 2 or b.ndim != 1 or W.shape[0] != len(labels) or b.shape[0] != len(labels):
                return
            if list(self.labels) != labels:
                # Align classifier labels to the head order.
                self.labels = labels
            self._linear_W = W
            self._linear_b = b
            self._linear_labels = labels
            self._head_mode_active = "linear"
            self.metadata.description += " (linear head)"
        except Exception:
            return

    def _resolve_head_mode(self) -> None:
        """Select active head based on config and file availability."""
        self._head_mode_active = "centroid"
        mode = (self._head_mode_config or "centroid").lower()
        if mode == "centroid":
            self._head_mode_active = "centroid"
            return
        if mode in {"linear", "auto"}:
            self._try_load_linear_head()
            if mode == "linear" and self._head_mode_active != "linear":
                print("[EncoderClassifier] Requested linear head but none loaded; using centroid.")
        else:
            print(f"[EncoderClassifier] Unknown head_mode '{mode}', defaulting to centroid.")

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

    def _predict_proba_centroid(self, emb_norm: np.ndarray) -> np.ndarray:
        if self._centroids is None:
            raise ValueError("Centroids not loaded")
        sims = emb_norm @ self._centroids.T
        logits = sims.astype("float64")
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return probs

    def _predict_proba_linear(self, emb: np.ndarray) -> Optional[np.ndarray]:
        if self._linear_W is None or self._linear_b is None:
            return None
        logits = self._linear_W @ emb + self._linear_b
        logits = logits - float(logits.max())
        exp = np.exp(logits, dtype="float64")
        probs = exp / float(exp.sum())
        return probs

    def _predict_with_embedding(self, emb_vec: List[float]) -> Dict[str, float]:
        if np is None:
            base = {label: max(0.0, self._cosine(emb_vec, self._prototype_vectors.get(label, []))) for label in self.labels}
            return normalize_probabilities(base, self.labels)
        emb_np = np.array(emb_vec, dtype="float32")
        norm = np.linalg.norm(emb_np) + 1e-9
        emb_norm = emb_np / norm
        if self._head_mode_active == "linear":
            linear_probs = self._predict_proba_linear(emb_np)
            if linear_probs is not None:
                return {lbl: float(linear_probs[idx]) for idx, lbl in enumerate(self.labels)}
        if self._centroids is not None:
            centroid_probs = self._predict_proba_centroid(emb_norm)
            return {lbl: float(centroid_probs[idx]) for idx, lbl in enumerate(self._centroid_labels or self.labels)}
        # Prototype fallback
        scores = {
            label: max(0.0, self._cosine(emb_vec, self._prototype_vectors.get(label, [])))
            for label in self.labels
        }
        return normalize_probabilities(scores, self.labels)

    def predict_proba(self, text: str) -> Dict[str, float]:
        text_vec = self._encode_text(text or "")
        base_probs = self._predict_with_embedding(text_vec)
        if self.use_lexical_prior:
            return apply_lexical_prior(text, base_probs)
        return base_probs

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.__getitem__)


__all__ = ["EncoderClassifier"]
