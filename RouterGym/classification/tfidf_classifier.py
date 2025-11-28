"""TF-IDF + Logistic Regression classifier with confidence outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class TFIDFClassifier:
    """Wrapper around a TF-IDF + calibrated logistic regression pipeline."""

    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None

    def train(self, df: pd.DataFrame, test_size: float = 0.0, random_state: int = 42) -> float:
        """Train the classifier; optionally return held-out accuracy if test_size > 0."""
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("DataFrame must contain 'text' and 'label' columns")

        texts = df["text"].astype(str)
        labels = df["label"].astype(str)

        if test_size and test_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)
        else:
            X_train, y_train = texts, labels
            X_val, y_val = None, None

        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
        base_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        min_class = int(labels.value_counts().min())
        cv_folds = max(2, min(5, min_class))
        clf = CalibratedClassifierCV(base_clf, cv=cv_folds)

        self.pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        self.pipeline.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            preds = self.pipeline.predict(X_val)
            acc = float(np.mean(preds == y_val))
            return acc
        return 0.0

    def predict_with_confidence(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Return (label, confidence, full_probs)."""
        if self.pipeline is None:
            raise ValueError("Classifier not trained or loaded")
        probs = self.pipeline.predict_proba([text or ""])[0]
        labels = self.pipeline.classes_
        top_idx = int(np.argmax(probs))
        label = str(labels[top_idx])
        confidence = float(probs[top_idx])
        full_probs = {str(lbl): float(p) for lbl, p in zip(labels, probs)}
        return label, confidence, full_probs

    def save(self, path: Path | str) -> None:
        """Persist the trained pipeline."""
        if self.pipeline is None:
            raise ValueError("Classifier not trained or loaded")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path | str) -> "TFIDFClassifier":
        """Load a persisted pipeline."""
        path = Path(path)
        pipeline = joblib.load(path)
        inst = cls()
        inst.pipeline = pipeline
        return inst
