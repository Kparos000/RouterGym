"""Lightweight text classifier for ticket categories with optional confidence."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

try:
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SKLEARN_AVAILABLE = False
    Pipeline = None  # type: ignore
    TfidfVectorizer = None  # type: ignore
    LogisticRegression = None  # type: ignore
    pd = None  # type: ignore

from RouterGym.data.tickets import dataset_loader


@lru_cache(maxsize=1)
def _load_training_data(limit: int = 5000):
    if not SKLEARN_AVAILABLE:
        return None
    try:
        df = dataset_loader.load_dataset(limit)
        if not isinstance(df, pd.DataFrame):
            return None
        if "text" not in df.columns or "label" not in df.columns:
            return None
        return df[["text", "label"]].dropna()
    except Exception:
        return None


@lru_cache(maxsize=1)
def _train_pipeline(limit: int = 5000):
    if not SKLEARN_AVAILABLE:
        return None
    data = _load_training_data(limit)
    if data is None or data.empty:
        return None
    try:
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=300, n_jobs=1, multi_class="auto")),
            ]
        )
        pipeline.fit(data["text"], data["label"])
        return pipeline
    except Exception:
        return None


def predict_label_with_confidence(text: str, threshold: float = 0.5) -> Tuple[str, float]:
    """Return (label, confidence) using a light supervised classifier if available."""
    if not text:
        return "miscellaneous", 0.0

    model = _train_pipeline()
    if model is not None:
        try:
            proba = model.predict_proba([text])[0]
            labels = list(model.classes_)
            top_idx = int(proba.argmax())
            label = labels[top_idx] if labels else "miscellaneous"
            confidence = float(proba[top_idx])
            return label, confidence
        except Exception:
            pass

    # Fallback heuristic if sklearn is unavailable or training failed
    lower = text.lower()
    keyword_map = [
        ({"login", "password", "account", "access", "credential"}, "access"),
        ({"admin", "administrator", "permission", "privilege", "rights"}, "administrative rights"),
        ({"laptop", "printer", "device", "hardware", "dock", "keyboard", "mouse"}, "hardware"),
        ({"hr", "benefit", "leave", "vacation", "payroll"}, "hr support"),
        ({"project", "repo", "repository", "internal"}, "internal project"),
        ({"buy", "purchase", "order", "procure", "invoice", "billing"}, "purchase"),
        ({"storage", "quota", "space", "drive", "share"}, "storage"),
    ]
    for keywords, label in keyword_map:
        if any(k in lower for k in keywords):
            return label, 0.4
    if "misc" in lower or "general" in lower or "other" in lower:
        return "miscellaneous", 0.35
    return "miscellaneous", 0.2


__all__ = ["predict_label_with_confidence"]
