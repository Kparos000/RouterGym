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
    apply_lexical_prior,
    canonical_label,
    normalize_probabilities,
)
from RouterGym.classifiers.tfidf_classifier import TFIDFClassifier

_PROTOTYPES: Dict[str, str] = {
    "access": "password reset login mfa sso otp locked out account",
    "administrative rights": "admin privilege elevated permission group membership entitlement role change",
    "hardware": "laptop printer dock monitor battery broken keyboard mouse device",
    "hr support": "benefits payroll leave hr question vacation",
    "purchase": "buy purchase order license invoice billing subscription renewal procurement payment quote",
    "miscellaneous": "general inquiry misc other",
}

CALIBRATED_HEAD_PATH = Path(__file__).resolve().parent / "encoder_calibrated_head.npz"


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
        self._backend_name = "encoder_centroid"
        self._calib_W: Optional[np.ndarray] = None if np is not None else None
        self._calib_b: Optional[np.ndarray] = None if np is not None else None
        self._calib_mean: Optional[np.ndarray] = None if np is not None else None
        self._calib_std: Optional[np.ndarray] = None if np is not None else None
        self._calib_feature_dim: Optional[int] = None
        self._tfidf_classifier: Optional[TFIDFClassifier] = None
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

    def _try_load_calibrated_head(self, allow_fallback: bool) -> bool:
        """Load calibrated logistic head; optionally allow fallback to centroid if missing or invalid."""
        if np is None:
            if allow_fallback:
                print("[EncoderClassifier] Warning: numpy unavailable; falling back to centroids.")
                return False
            raise RuntimeError(
                "Encoder calibrated head missing or incompatible. Run `python -m RouterGym.scripts.train_encoder_calibrated_head` to regenerate encoder_calibrated_head.npz."
            )
        path = CALIBRATED_HEAD_PATH
        if not path.exists():
            if allow_fallback:
                print("[EncoderClassifier] Warning: calibrated head file missing; falling back to centroids.")
                return False
            raise RuntimeError(
                "Encoder calibrated head missing or incompatible. Run `python -m RouterGym.scripts.train_encoder_calibrated_head` to regenerate encoder_calibrated_head.npz."
            )
        try:
            data = np.load(path, allow_pickle=True)
            labels = [canonical_label(lbl) for lbl in data["labels"].tolist()]
            W = np.asarray(data["W"], dtype="float32")
            b = np.asarray(data["b"], dtype="float32")
            mean = np.asarray(data.get("feature_mean", np.zeros(W.shape[1], dtype="float32")), dtype="float32")
            std = np.asarray(data.get("feature_std", np.ones(W.shape[1], dtype="float32")), dtype="float32")
            feature_dim = int(data.get("feature_dim", W.shape[1] if W.ndim == 2 else 0))
            if (
                W.ndim != 2
                or b.ndim != 1
                or W.shape[0] != len(labels)
                or b.shape[0] != len(labels)
                or feature_dim <= 0
                or feature_dim != W.shape[1]
            ):
                if allow_fallback:
                    print("[EncoderClassifier] Warning: calibrated head incompatible; falling back to centroids.")
                    return False
                raise RuntimeError(
                    "Encoder calibrated head missing or incompatible. Run `python -m RouterGym.scripts.train_encoder_calibrated_head` to regenerate encoder_calibrated_head.npz."
                )
            if list(self.labels) != labels:
                if allow_fallback:
                    print("[EncoderClassifier] Warning: calibrated head labels mismatch; falling back to centroids.")
                    return False
                raise RuntimeError(
                    "Encoder calibrated head missing or incompatible. Run `python -m RouterGym.scripts.train_encoder_calibrated_head` to regenerate encoder_calibrated_head.npz."
                )
            self._calib_W = W
            self._calib_b = b
            self._calib_mean = mean
            self._calib_std = std
            self._calib_feature_dim = feature_dim
            try:
                self._tfidf_classifier = TFIDFClassifier(labels=self.labels)
            except Exception:
                print("[EncoderClassifier] Warning: failed to initialize TF-IDF classifier for calibrated head; continuing without TF-IDF features.")
                self._tfidf_classifier = None
            self._head_mode_active = "calibrated"
            self._backend_name = "encoder_calibrated"
            self.metadata.description += " (calibrated head)"
            return True
        except Exception:
            if allow_fallback:
                print("[EncoderClassifier] Warning: calibrated head load failed; falling back to centroids.")
                return False
            raise RuntimeError(
                "Encoder calibrated head missing or incompatible. Run `python -m RouterGym.scripts.train_encoder_calibrated_head` to regenerate encoder_calibrated_head.npz."
            )

    def _resolve_head_mode(self) -> None:
        """Select active head based on config and file availability."""
        self._head_mode_active = "centroid"
        allow_fallback = os.getenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", "") == "1"
        mode = (self._head_mode_config or "centroid").lower()
        if mode == "centroid":
            return
        if mode in {"calibrated", "auto"}:
            success = self._try_load_calibrated_head(allow_fallback=allow_fallback)
            if not success and not allow_fallback:
                raise RuntimeError(
                    "Encoder calibrated head missing or incompatible. Run `python -m RouterGym.scripts.train_encoder_calibrated_head` to regenerate encoder_calibrated_head.npz."
                )
        else:
            print(f"[EncoderClassifier] Unknown head_mode '{mode}', defaulting to centroid.")
        if self._head_mode_active != "calibrated":
            self._backend_name = "encoder_centroid"

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

    def _compute_prior_vector(self, text: str) -> Dict[str, float]:
        base = {label: 1.0 / max(len(self.labels), 1) for label in self.labels}
        return apply_lexical_prior(text, base, alpha=0.0, beta=1.0)

    def _compute_calibrated_features(self, text: str, emb_np: np.ndarray) -> Optional[np.ndarray]:
        if np is None:
            return None
        emb_norm = emb_np / (np.linalg.norm(emb_np) + 1e-9)
        priors_dict = self._compute_prior_vector(text)
        priors = np.array([priors_dict.get(lbl, 0.0) for lbl in self.labels], dtype="float32")
        if self._tfidf_classifier is None:
            return None
        tfidf_probs_dict = self._tfidf_classifier.predict_proba(text)
        tfidf_probs = np.array([tfidf_probs_dict.get(lbl, 0.0) for lbl in self.labels], dtype="float32")
        features = np.concatenate([emb_norm.astype("float32"), priors, tfidf_probs], axis=0)
        if self._calib_feature_dim is not None and features.shape[0] != self._calib_feature_dim:
            raise RuntimeError(
                f"Calibrated head feature length mismatch: built {features.shape[0]}, expected {self._calib_feature_dim}"
            )
        return features

    @property
    def backend_name(self) -> str:
        """Expose which backend is active for debugging/analytics."""
        return self._backend_name

    def predict_proba(self, text: str) -> Dict[str, float]:
        if np is not None and self._encoder is not None:
            try:
                emb = self._encoder.encode(text or "", normalize_embeddings=True)  # type: ignore[attr-defined]
                emb_np = np.array(emb, dtype="float32")
                if self._head_mode_active == "calibrated" and self._calib_W is not None and self._calib_b is not None:
                    feats = self._compute_calibrated_features(text, emb_np)
                    if feats is not None:
                        mean = self._calib_mean if self._calib_mean is not None else np.zeros_like(feats)
                        std = self._calib_std if self._calib_std is not None else np.ones_like(feats)
                        std_safe = np.where(std > 1e-6, std, 1.0)
                        feats_std = (feats - mean) / std_safe
                        logits = self._calib_W @ feats_std + self._calib_b
                        logits = logits - float(logits.max())
                        exp = np.exp(logits, dtype="float64")
                        probs = exp / float(exp.sum())
                        return {lbl: float(probs[idx]) for idx, lbl in enumerate(self.labels)}

                if self._centroids is not None and self._centroid_labels:
                    emb_norm = emb_np / (np.linalg.norm(emb_np) + 1e-9)
                    sims = emb_norm @ self._centroids.T
                    logits = sims.astype("float64")
                    logits = logits - logits.max()
                    exp = np.exp(logits)
                    probs = exp / exp.sum()
                    centroid_probs = {lbl: float(probs[idx]) for idx, lbl in enumerate(self._centroid_labels)}
                    if self.use_lexical_prior:
                        return apply_lexical_prior(text, centroid_probs)
                    return normalize_probabilities(centroid_probs, self._centroid_labels)
            except Exception:
                print("[EncoderClassifier] Warning: centroid or calibrated inference failed, falling back to prototypes.")

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
